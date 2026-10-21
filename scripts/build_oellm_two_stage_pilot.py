#!/usr/bin/env python3
"""Build a small, auditable 36-language Instruct pilot and Reasoning candidates.

This is intentionally a *pilot* builder. It never silently promotes mixed-license
source rows into a production/release manifest. All output is un-packed JSONL
with source IDs, and every selected row is rendered with the actual tokenizer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import heapq
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from audit_two_stage_sft import source_kind, verify_render


EXCELLENT_REPO = "openeurollm/OpenEuroLLM-Excellent-SFT"
EXCELLENT_REVISION = "c4ee55804f256c5f8cc4a6ef3fdecbe4a4ea95e0"
AYA_REPO = "CohereLabs/aya_dataset"
AYA_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"

AYA_CODES = {
    "dan": "da", "deu": "de", "ell": "el", "eng": "en", "eus": "eu",
    "fin": "fi", "fra": "fr", "gle": "ga", "hun": "hu", "ita": "it",
    "lit": "lt", "nld": "nl", "pol": "pl", "por": "pt", "spa": "es",
    "sqi": "sq", "srp": "sr", "swe": "sv", "tur": "tr", "ukr": "uk",
}

# A small, heterogeneous selection for engineering and behavior checks only.
DIRECT_SOURCES = {
    "dolci/train", "smoltalk2/smoltalk_multilingual_8languages_lang_5_no_think",
    "smoltalk2/tulu_3_sft_personas_instruction_following_no_think",
}
REASONING_SOURCES = {
    "smoltalk2/OpenThoughts3_1.2M_think", "smoltalk2/smoltalk_multilingual8_Qwen3_32B_think",
    "nemotron/multilingual_de", "nemotron/multilingual_es",
    "nemotron/multilingual_fr", "nemotron/multilingual_it",
}


def stable_score(seed: int, key: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest(), "big"
    )


def keep_smallest(heaps: dict, group: str, item: dict, key: str, limit: int, seed: int) -> None:
    score = stable_score(seed, key)
    heap = heaps[group]
    entry = (-score, key, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def rows_from_parquets(paths: list[Path], columns: list[str]):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            yield from batch.to_pylist()
        del batch, parquet
        gc.collect()
        pa.default_memory_pool().release_unused()


def normalize_prompt(messages: list[dict]) -> str:
    return "\n".join(
        " ".join(str(message.get("content", "")).split()).casefold()
        for message in messages if message.get("role") == "user"
    )


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excellent-dir", type=Path, required=True)
    parser.add_argument("--aya-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-aya-language", type=int, default=32)
    parser.add_argument("--max-per-excellent-source", type=int, default=64)
    parser.add_argument("--max-per-reasoning-source", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if min(args.max_per_aya_language, args.max_per_excellent_source,
           args.max_per_reasoning_source, args.max_length) < 1:
        parser.error("limits must be positive")

    shards = sorted((args.excellent_dir / "data").glob("train-*.parquet"))
    if len(shards) != 62:
        raise ValueError(f"Expected all 62 pinned Excellent shards, found {len(shards)}")
    if not args.aya_parquet.is_file():
        raise FileNotFoundError(args.aya_parquet)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.chat_template = args.template.read_text(encoding="utf-8")
    instruct_heaps: dict[str, list] = defaultdict(list)
    reasoning_heaps: dict[str, list] = defaultdict(list)
    seen = Counter()

    for index, row in enumerate(rows_from_parquets([args.aya_parquet],
                                                     ["inputs", "targets", "language_code", "annotation_type"])):
        seen["aya_total"] += 1
        language = AYA_CODES.get(row.get("language_code"))
        if not language:
            continue
        prompt, answer = row.get("inputs"), row.get("targets")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str) or not answer.strip():
            seen["aya_empty"] += 1
            continue
        item = {
            "id": f"aya-train-{index}", "source": AYA_REPO,
            "source_revision": AYA_REVISION, "language": language,
            "annotation_type": row.get("annotation_type"),
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
        }
        keep_smallest(instruct_heaps, f"aya:{language}", item, item["id"],
                      args.max_per_aya_language, args.seed)

    for row in rows_from_parquets(shards, ["id", "messages", "source", "tools"]):
        seen["excellent_total"] += 1
        source = row.get("source")
        if source not in DIRECT_SOURCES | REASONING_SOURCES or row.get("tools"):
            continue
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        try:
            kind = source_kind(messages)
        except (KeyError, TypeError, ValueError):
            seen["excellent_malformed"] += 1
            continue
        item = {
            "id": row["id"], "source": f"{EXCELLENT_REPO}:{source}",
            "source_revision": EXCELLENT_REVISION, "language": "unverified",
            "messages": messages,
        }
        if source in DIRECT_SOURCES and kind == "direct":
            keep_smallest(instruct_heaps, f"excellent:{source}", item, item["id"],
                          args.max_per_excellent_source, args.seed)
        elif source in REASONING_SOURCES and kind == "reasoning":
            keep_smallest(reasoning_heaps, f"excellent:{source}", item, item["id"],
                          args.max_per_reasoning_source, args.seed)

    # Check the exact source text *and* rendered assistant labels; reject, never truncate.
    selected = {"instruct": [], "reasoning": []}
    for stage, heaps in (("instruct", instruct_heaps), ("reasoning", reasoning_heaps)):
        for group in sorted(heaps):
            for _, _, item in sorted(heaps[group], key=lambda entry: (-entry[0], entry[1])):
                try:
                    checked = verify_render(tokenizer, item["messages"], stage, args.max_length)
                except (KeyError, TypeError, ValueError) as exc:
                    seen[f"{stage}_render_rejected"] += 1
                    seen[f"{stage}_render_rejected:{str(exc).split(':', 1)[0][:80]}"] += 1
                    continue
                item["rendered_tokens"] = checked["tokens"]
                item["supervised_tokens"] = checked["supervised_tokens"]
                selected[stage].append(item)

    deduped: dict[str, list[dict]] = {"instruct": [], "reasoning": []}
    prompt_seen = set()
    for stage in ("instruct", "reasoning"):
        # Aya wins exact prompt collisions against Excellent; ensure an identical
        # prompt is not both direct Stage 1 and a Stage 2 reasoning example.
        ordered = sorted(selected[stage], key=lambda row: (not row["source"].startswith("CohereLabs/"), row["id"]))
        for item in ordered:
            prompt = normalize_prompt(item["messages"])
            if not prompt:
                seen[f"{stage}_empty_prompt"] += 1
                continue
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if digest in prompt_seen:
                seen[f"{stage}_duplicate_prompt"] += 1
                continue
            prompt_seen.add(digest)
            item["prompt_sha256"] = digest
            deduped[stage].append(item)

    instruct_train, instruct_eval = [], []
    for item in deduped["instruct"]:
        # Deterministic internal pilot validation; not a benchmark or release test.
        if stable_score(args.seed + 1, item["prompt_sha256"]) % 10 == 0:
            instruct_eval.append(item)
        else:
            instruct_train.append(item)
    if not instruct_train or not instruct_eval or not deduped["reasoning"]:
        raise RuntimeError("insufficient audited Instruct train/eval or Reasoning candidate rows: "
                           + json.dumps({"train": len(instruct_train), "eval": len(instruct_eval),
                                         "reasoning": len(deduped["reasoning"]), "seen": dict(seen)},
                                        sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "instruct_train": (args.output_dir / "instruct_train.jsonl", instruct_train),
        "instruct_eval": (args.output_dir / "instruct_eval.jsonl", instruct_eval),
        "reasoning_candidates": (args.output_dir / "reasoning_candidates.jsonl", deduped["reasoning"]),
    }
    files = {name: {"path": str(path), "rows": len(rows), "sha256": write_jsonl(path, rows),
                    "supervised_tokens": sum(row["supervised_tokens"] for row in rows)}
             for name, (path, rows) in paths.items()}
    manifest = {
        "kind": "internal_engineering_pilot_not_production_or_release",
        "seed": args.seed, "max_length": args.max_length,
        "sources": {EXCELLENT_REPO: EXCELLENT_REVISION, AYA_REPO: AYA_REVISION},
        "template_sha256": hashlib.sha256(args.template.read_bytes()).hexdigest(),
        "files": files, "seen": dict(seen),
        "instruct_train_by_source": dict(Counter(row["source"] for row in instruct_train)),
        "instruct_train_by_language": dict(Counter(row["language"] for row in instruct_train)),
        "reasoning_by_source": dict(Counter(row["source"] for row in deduped["reasoning"])),
        "release_blockers": [
            "Excellent-SFT is a mixed-license aggregate; source-level release review is pending",
            "Native-language quality review and benchmark decontamination are pending",
            "No full-corpus tokenizer, length, language, or supervised-token distribution has been audited",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
