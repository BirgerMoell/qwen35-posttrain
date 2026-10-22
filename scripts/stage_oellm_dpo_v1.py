#!/usr/bin/env python3
"""Stage the Stage-4 DPO text mix: an exact replication of the threeway DPO data, with every source pinned.

Replicates `stage_threeway_dpo_full.py` from birgermoell/oellm-9b-threeway-dpo-full-v1 (the reference run):
- nvidia/HelpSteer3 `preference`: the sign of `overall_preference` picks the chosen response (positive ->
  response2); ties are dropped; the native validation split is kept.
- openeurollm/Dolci-Instruct-DPO-translated, 12 languages: Parquet shards when present, else JSONL; string
  responses become one assistant message; every 100th source row (per language) is held out for validation.
- argilla/ultrafeedback-binarized-preferences-cleaned: a duplicated leading user turn is removed from chosen and
  rejected; every 100th row is held out.

Differences from the reference run: HelpSteer3 and UltraFeedback are read from pinned Hub revisions (the
reference used unpinned copies), and every record gets a stable `id`. Output: gzipped JSONL with
prompt / chosen / rejected message lists, source, language, id; plus manifest.json with counts and revisions.
Two phases: `--prefetch-only` downloads the pinned files into the HF cache (login node, network only);
the staging pass then reads them offline (batch node, HF_HUB_OFFLINE=1). Tokenization and length filtering
happen in build_oellm_dpo_v1_pretok.py, never here.
"""
from __future__ import annotations

import argparse
import gzip
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

DOLCI_REPO = "openeurollm/Dolci-Instruct-DPO-translated"
DOLCI_REVISION = "a57841a6bbabfed8ca2597438995c2991e5b73b7"
DOLCI_LANGUAGES = ("cs", "de", "el", "en", "es", "fi", "fr", "it", "pl", "ro", "sv", "uk")
HELPSTEER_REPO = "nvidia/HelpSteer3"
HELPSTEER_REVISION = "f6d145777bcbde96137596340fab89793acd1031"
ULTRA_REPO = "argilla/ultrafeedback-binarized-preferences-cleaned"
ULTRA_REVISION = "770076f077c4c5e298498fa32f804857f46d5134"


def messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"expected message list, got {type(value).__name__}")
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in value]


def response_messages(value: Any) -> list[dict[str, str]]:
    """Dolci JSONL has response strings; its Parquet shards have messages."""
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    return messages(value)


def local(repo: str, path: str, revision: str) -> str:
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision)


def dolci_files() -> dict[str, tuple[str, list[str]]]:
    import os
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        cached = Path(local(DOLCI_REPO, "en/shard00-of-25.parquet", DOLCI_REVISION)).parents[1]
        paths = [str(p.relative_to(cached)) for p in cached.rglob("*") if p.is_file()]
    else:
        paths = HfApi().list_repo_files(DOLCI_REPO, repo_type="dataset", revision=DOLCI_REVISION)
    out = {}
    for language in DOLCI_LANGUAGES:
        parquet = sorted(p for p in paths if p.startswith(f"{language}/shard") and p.endswith(".parquet"))
        jsonl = [p for p in paths if p == f"{language}/translated_dpo.jsonl"]
        builder, selected = ("parquet", parquet) if parquet else ("json", jsonl)
        if not selected:
            raise FileNotFoundError(f"no Dolci files for {language} at {DOLCI_REVISION}")
        out[language] = (builder, selected)
    return out


def prefetch() -> None:
    for split in ("train", "validation"):
        print(local(HELPSTEER_REPO, f"preference/{split}.jsonl.gz", HELPSTEER_REVISION), flush=True)
    for language, (_, selected) in dolci_files().items():
        for path in selected:
            local(DOLCI_REPO, path, DOLCI_REVISION)
        print("dolci", language, len(selected), "files cached", flush=True)
    print(local(ULTRA_REPO, "data/train-00000-of-00001.parquet", ULTRA_REVISION), flush=True)


def write(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prefetch-only", action="store_true")
    a = ap.parse_args()
    if a.prefetch_only:
        prefetch()
        return
    a.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {}

    with ExitStack() as stack:
        out = {name: stack.enter_context(gzip.open(a.out / f"{name}.jsonl.gz", "wt", encoding="utf-8"))
               for name in ("helpsteer3-train", "helpsteer3-validation", "dolci-train", "dolci-validation",
                            "ultrafeedback-train", "ultrafeedback-validation")}

        # --- HelpSteer3 preference (native splits, ties dropped)
        hs = {"train": 0, "validation": 0, "dropped_ties": 0}
        for split in ("train", "validation"):
            path = local(HELPSTEER_REPO, f"preference/{split}.jsonl.gz", HELPSTEER_REVISION)
            for index, row in enumerate(load_dataset("json", data_files={split: path}, split=split, streaming=True)):
                preference = int(row["overall_preference"])
                if preference == 0:
                    hs["dropped_ties"] += 1
                    continue
                chosen, rejected = ((row["response2"], row["response1"]) if preference > 0
                                    else (row["response1"], row["response2"]))
                write(out[f"helpsteer3-{split}"], {
                    "id": f"helpsteer3-{split}-{index:06d}",
                    "prompt": messages(row["context"]),
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                    "source": HELPSTEER_REPO, "language": row.get("language") or "unknown"})
                hs[split] += 1
        counts[HELPSTEER_REPO] = {**hs, "revision": HELPSTEER_REVISION}
        print("helpsteer3", hs, flush=True)

        # --- Dolci translated, 12 languages, every 100th row held out per language
        dolci, by_language = {"train": 0, "validation": 0}, {}
        for language, (builder, selected) in dolci_files().items():
            urls = [local(DOLCI_REPO, p, DOLCI_REVISION) for p in selected]
            lang = {"train": 0, "validation": 0}
            for index, row in enumerate(load_dataset(builder, data_files={"train": urls}, split="train", streaming=True)):
                split = "validation" if index % 100 == 0 else "train"
                write(out[f"dolci-{split}"], {
                    "id": f"dolci-{language}-{index:07d}",
                    "prompt": messages(row["prompt"]),
                    "chosen": response_messages(row["chosen"]),
                    "rejected": response_messages(row["rejected"]),
                    "source": DOLCI_REPO, "language": language})
                lang[split] += 1
                dolci[split] += 1
            by_language[language] = lang
            print("dolci", language, builder, len(urls), lang, flush=True)
        counts[DOLCI_REPO] = {**dolci, "revision": DOLCI_REVISION, "by_language": by_language}

        # --- UltraFeedback cleaned, every 100th row held out
        uf = {"train": 0, "validation": 0}
        url = local(ULTRA_REPO, "data/train-00000-of-00001.parquet", ULTRA_REVISION)
        for index, row in enumerate(load_dataset("parquet", data_files={"train": url}, split="train", streaming=True)):
            chosen, rejected = messages(row["chosen"]), messages(row["rejected"])
            split = "validation" if index % 100 == 0 else "train"
            write(out[f"ultrafeedback-{split}"], {
                "id": f"ultrafeedback-{index:06d}",
                "prompt": [{"role": "user", "content": str(row["prompt"])}],
                "chosen": chosen[1:] if chosen and chosen[0]["role"] == "user" else chosen,
                "rejected": rejected[1:] if rejected and rejected[0]["role"] == "user" else rejected,
                "source": ULTRA_REPO, "language": "en"})
            uf[split] += 1
        counts[ULTRA_REPO] = {**uf, "revision": ULTRA_REVISION}
        print("ultrafeedback", uf, flush=True)

    manifest = {"kind": "oellm_dpo_v1_staged_text",
                "reference": "birgermoell/oellm-9b-threeway-dpo-full-v1 stage_threeway_dpo_full.py",
                "split_policy": "native HelpSteer3 validation; every 100th source row held out for Dolci (per language) "
                                "and UltraFeedback",
                "sources": counts}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
