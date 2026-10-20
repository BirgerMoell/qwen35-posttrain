#!/usr/bin/env python3
"""Build the pre-tokenized Stage-1 Instruct mixture with the v2 chat template.

Sources (all direct-answer, text-only):
  excellent  audited Excellent-SFT direct rows (keeps that build's train/validation split)
  dolci_tr   translated Dolci, 12 languages; every source-family id is assigned to exactly
             one language by hash, so no prompt family is repeated across languages
  aya        human-authored Aya rows in non-English OpenEuroLLM target languages, capped
  identity   model-identity train rows at a fixed number of exposures

Every row is rendered and tokenized with the exact training tokenizer/template. The
outputs carry `input_ids` + `assistant_masks`, so TRL skips its own tokenization and
trains on exactly the audited tokens. A supervised <think>/</think> token is fatal;
overlength rows are rejected, never truncated. Per-unit reports make the build resumable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from jinja2 import TemplateError
import pyarrow as pa
import pyarrow.parquet as pq

from audit_two_stage_sft import source_kind
from build_oellm_excellent_stage1 import canonical_prompt, file_sha256, get_tokenizer, THINK_IDS

DOLCI_LANGS = ("cs", "de", "el", "es", "fi", "fr", "it", "nl", "pl", "ro", "sv", "uk")
# Aya ISO 639-3 codes for the OpenEuroLLM non-English target languages.
AYA_TARGETS = {
    "bul": "bg", "ces": "cs", "dan": "da", "deu": "de", "ell": "el", "spa": "es", "est": "et",
    "fin": "fi", "fra": "fr", "gle": "ga", "hrv": "hr", "hun": "hu", "ita": "it", "lit": "lt",
    "lav": "lv", "lvs": "lv", "mlt": "mt", "nld": "nl", "pol": "pl", "por": "pt", "ron": "ro",
    "slk": "sk", "slv": "sl", "swe": "sv", "cat": "ca", "eus": "eu", "glg": "gl", "bos": "bs",
    "kat": "ka", "mkd": "mk", "sqi": "sq", "als": "sq", "srp": "sr", "tur": "tr", "ukr": "uk",
    "isl": "is", "nob": "no", "nno": "no", "nor": "no",
}
SCHEMA = pa.schema([
    ("input_ids", pa.list_(pa.int32())), ("assistant_masks", pa.list_(pa.int8())),
    ("source", pa.string()), ("language", pa.string()), ("id", pa.string()),
    ("supervised_tokens", pa.int32()),
])


class SplitWriter:
    """Stream rows to `<split>-<unit>.parquet` in small row groups; no file if empty."""

    def __init__(self, out: Path, split: str, name: str, batch_rows: int = 2000, schema: pa.Schema = SCHEMA):
        self.schema = schema
        self.path = out / f"{split}-{name}.parquet"
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.batch_rows, self.rows, self.buffer, self.writer = batch_rows, 0, [], None

    def add(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_rows:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.partial, self.schema, compression="zstd")
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
        self.rows += len(self.buffer)
        self.buffer = []

    def finish(self) -> dict | None:
        self._flush()
        if self.writer is None:
            return None
        self.writer.close()
        os.replace(self.partial, self.path)
        return {"path": str(self.path), "rows": self.rows, "sha256": file_sha256(self.path)}


def h64(text: str, part: int = 0) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[16 * part:16 * part + 16], 16)


def iter_unit(unit: dict, config: dict):
    """Yield (split, source, language, id, messages) for one work unit."""
    kind, path = unit["kind"], Path(unit["path"])
    if kind == "excellent":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                yield unit["split"], row["source"], "mixed", str(row["id"]), row["messages"]
    elif kind == "dolci_tr":
        lang = unit["language"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["id", "messages"]):
            for row in batch.to_pylist():
                family = str(row["id"])
                if DOLCI_LANGS[h64(family, 1) % len(DOLCI_LANGS)] != lang:
                    continue
                split = "validation" if h64(family) % 1000 == 0 else "train"
                yield split, "dolci_tr", lang, f"{lang}:{family}", row["messages"]
    elif kind == "aya":
        kept: Counter[str] = Counter()
        rows = pq.read_table(path, columns=["inputs", "targets", "language_code"]).to_pylist()
        # Deterministic per-language cap: order by prompt hash, not file order.
        rows.sort(key=lambda r: h64(r["inputs"] or ""))
        for row in rows:
            lang = AYA_TARGETS.get(row["language_code"])
            prompt, answer = (row["inputs"] or "").strip(), (row["targets"] or "").strip()
            if lang is None or not prompt or len(answer) < config["aya_min_answer_chars"]:
                continue
            split = "validation" if h64(prompt) % 200 == 0 else "train"
            if split == "train":
                if kept[lang] >= config["aya_cap_per_language"]:
                    continue
                kept[lang] += 1
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            yield split, "aya", lang, f"aya:{h64(prompt):016x}", messages
    elif kind == "identity":
        repeats = config["identity_exposures"] if unit["split"] == "train" else 1
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        for exposure in range(repeats):
            for row in rows:
                yield unit["split"], "identity", row.get("language", "mixed"), f"{row['id']}#{exposure}", row["messages"]
    else:
        raise ValueError(f"unknown unit kind {kind}")


def process_unit(unit: dict, output_dir: str, config: dict) -> dict:
    out = Path(output_dir)
    report_path = out / f"{unit['name']}.report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("config") == config and previous.get("unit") == unit:
            return previous
    tokenizer = get_tokenizer(config["tokenizer"], config["template"])
    writers = {"train": SplitWriter(out, "train", unit["name"]),
               "validation": SplitWriter(out, "validation", unit["name"])}
    counts: Counter[str] = Counter()
    by_group: dict[str, Counter[str]] = {}
    for split, source, language, row_id, messages in iter_unit(unit, config):
        if config["max_rows_per_unit"] and counts["seen"] >= config["max_rows_per_unit"]:
            break
        group = by_group.setdefault(f"{source}|{language}", Counter())
        counts["seen"] += 1
        try:
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            if source_kind(messages) != "direct":
                raise ValueError("not a direct-answer row")
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_assistant_tokens_mask=True,
            )
            ids = list(rendered["input_ids"])
            mask = rendered.get("assistant_masks")
            if mask is None:
                mask = rendered.get("assistant_tokens_mask")
            if mask is None or len(ids) != len(mask) or not 0 < sum(mask) < len(mask) or mask[0]:
                raise ValueError("invalid assistant-only loss mask")
            target_ids = [token for token, active in zip(ids, mask) if active]
            if any(token in THINK_IDS for token in target_ids):
                raise RuntimeError(f"FATAL: {row_id} from {source} has a supervised think token")
            if len(ids) > config["max_length"]:
                counts["overlength"] += 1
                group["overlength"] += 1
                continue
            target = tokenizer.decode(target_ids, skip_special_tokens=False)
            if messages[-1]["content"].strip() not in target or "<|im_start|>user" in target:
                raise ValueError("final answer missing or user text in assistant labels")
        except RuntimeError:
            raise
        except (KeyError, TypeError, ValueError, TemplateError) as exc:
            counts["rejected"] += 1
            group["rejected"] += 1
            if counts["rejected"] <= 3:
                print(f"{unit['name']} rejected {row_id}: {exc}", flush=True)
            continue
        writers[split].add({
            "input_ids": ids, "assistant_masks": [int(v) for v in mask], "source": source,
            "language": language, "id": row_id, "supervised_tokens": len(target_ids),
        })
        group[f"{split}_rows"] += 1
        group[f"{split}_rendered_tokens"] += len(ids)
        group[f"{split}_supervised_tokens"] += len(target_ids)
    files = {split: info for split, writer in writers.items() if (info := writer.finish())}
    report = {"unit": unit, "config": config, "counts": dict(counts),
              "by_group": {k: dict(v) for k, v in sorted(by_group.items())}, "files": files}
    partial = report_path.with_name(report_path.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, report_path)
    return report


def plan_units(args) -> list[dict]:
    units = []
    for split in ("train", "validation"):
        for path in sorted(args.excellent_dir.glob(f"{split}-*.jsonl")):
            if path.stat().st_size:
                units.append({"kind": "excellent", "split": split, "path": str(path),
                              "name": f"excellent-{path.stem}"})
    for lang in DOLCI_LANGS:
        paths = sorted((args.dolci_dir / lang).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no translated Dolci shards for {lang}")
        for path in paths:
            units.append({"kind": "dolci_tr", "language": lang, "path": str(path),
                          "name": f"dolci_tr-{lang}-{path.stem}"})
    aya = sorted((args.aya_dir / "data").glob("train-*.parquet"))
    if len(aya) != 1:
        raise FileNotFoundError("expected one Aya train parquet")
    units.append({"kind": "aya", "path": str(aya[0]), "name": "aya-train"})
    for split, name in (("train", "identity_train.jsonl"), ("validation", "identity_validation.jsonl")):
        units.append({"kind": "identity", "split": split, "path": str(args.identity_dir / name),
                      "name": f"identity-{split}"})
    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excellent-dir", type=Path, required=True)
    parser.add_argument("--dolci-dir", type=Path, required=True)
    parser.add_argument("--aya-dir", type=Path, required=True)
    parser.add_argument("--identity-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--aya-cap-per-language", type=int, default=8000)
    parser.add_argument("--aya-min-answer-chars", type=int, default=50)
    parser.add_argument("--identity-exposures", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="2 units per source, 300 rows each")
    args = parser.parse_args()
    excellent_manifest = json.loads((args.excellent_dir / "manifest.json").read_text(encoding="utf-8"))
    if excellent_manifest["kind"] != "full_stage1_direct_audited":
        raise ValueError("Excellent input is not the full audited Stage-1 build")
    get_tokenizer(str(args.tokenizer), str(args.template))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "tokenizer": str(args.tokenizer), "template": str(args.template),
        "template_sha256": file_sha256(args.template),
        "tokenizer_json_sha256": file_sha256(args.tokenizer / "tokenizer.json"),
        "max_length": args.max_length, "aya_cap_per_language": args.aya_cap_per_language,
        "aya_min_answer_chars": args.aya_min_answer_chars,
        "identity_exposures": args.identity_exposures,
        "excellent_manifest_sha256": file_sha256(args.excellent_dir / "manifest.json"),
        "max_rows_per_unit": 300 if args.smoke else 0,
    }
    units = plan_units(args)
    if args.smoke:
        per_kind: Counter[str] = Counter()
        units = [u for u in units if (per_kind.update([u["kind"]]) or per_kind[u["kind"]] <= 2)]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_unit, unit, str(args.output_dir), config) for unit in units]
        for done, future in enumerate(as_completed(jobs), 1):
            reports.append(future.result())
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} units done", flush=True)
    counts: Counter[str] = Counter()
    by_group: dict[str, Counter[str]] = {}
    files = {"train": [], "validation": []}
    for report in sorted(reports, key=lambda r: r["unit"]["name"]):
        counts.update(report["counts"])
        for group, values in report["by_group"].items():
            by_group.setdefault(group, Counter()).update(values)
        for split, info in report["files"].items():
            files[split].append(info)
    totals = {
        split: {key: sum(v.get(f"{split}_{key}", 0) for v in by_group.values())
                for key in ("rows", "rendered_tokens", "supervised_tokens")}
        for split in ("train", "validation")
    }
    by_source: dict[str, Counter[str]] = {}
    for group, values in by_group.items():
        by_source.setdefault(group.split("|")[0], Counter()).update(values)
    manifest = {
        "kind": "engineering_smoke_not_production" if args.smoke else "stage1_instruct_mix_pretokenized",
        "version": "v1", "config": config,
        "counts": dict(counts), "totals": totals,
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "by_source_language": {k: dict(v) for k, v in sorted(by_group.items())},
        "files": files,
        "invariant": "Every admitted assistant mask has zero <think>/</think> token IDs and a preserved final answer",
        "status": "candidate_internal_training_data_not_release_clearance",
    }
    path = args.output_dir / "manifest.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    print(json.dumps({"totals": totals, "by_source": manifest["by_source"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
