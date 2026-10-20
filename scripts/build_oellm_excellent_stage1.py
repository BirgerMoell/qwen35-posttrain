#!/usr/bin/env python3
"""Build a full direct-answer Stage-1 corpus from pinned Excellent-SFT.

Every admitted row is tokenized with the *training* chat template. A masked
<think> or </think> token is fatal. Long rows are rejected, never truncated.
The per-shard outputs make a large CPU build resumable without touching the
original Parquet shards. This does not start model training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from audit_two_stage_sft import source_kind
from stage_excellent_sft_leonardo import (
    EXPECTED_BYTES, EXPECTED_ROWS, EXPECTED_SHARDS, REPO, REVISION,
)


_TOKENIZER = None
_TEMPLATE_HASH = None
THINK_IDS = frozenset((17, 18))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_tokenizer(tokenizer_path: str, template_path: str):
    global _TOKENIZER, _TEMPLATE_HASH
    template_hash = file_sha256(Path(template_path))
    if _TOKENIZER is None or _TEMPLATE_HASH != template_hash:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokenizer.chat_template = Path(template_path).read_text(encoding="utf-8")
        for tag, expected in (("<think>", 18), ("</think>", 17)):
            if tokenizer.encode(tag, add_special_tokens=False) != [expected]:
                raise ValueError(f"Unexpected tokenizer ID for {tag}")
        # Fail before reading millions of rows if generation itself inserts tags.
        generation_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "NO_THINK_SENTINEL"}],
            tokenize=False, add_generation_prompt=True,
        )
        if "<think>" in generation_prompt or "</think>" in generation_prompt:
            raise ValueError("The training generation prompt injects think tags")
        _TOKENIZER, _TEMPLATE_HASH = tokenizer, template_hash
    return _TOKENIZER


def canonical_prompt(messages: list[dict]) -> str:
    text = "\n".join(
        " ".join(message["content"].split()).casefold()
        for message in messages if message["role"] == "user"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.partial = path.with_name(path.name + ".partial")
        self.stream = self.partial.open("wb")
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def write(self, row: dict) -> None:
        encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self.stream.write(encoded)
        self.digest.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)

    def finish(self) -> dict:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.partial, self.path)
        return {"path": str(self.path), "rows": self.rows,
                "bytes": self.bytes, "sha256": self.digest.hexdigest()}


def report_is_valid(report_path: Path, config: dict) -> dict | None:
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("config") != config:
        raise ValueError(f"Existing shard report uses different configuration: {report_path}")
    for info in report["files"].values():
        path = Path(info["path"])
        if not path.is_file() or path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
            raise ValueError(f"Existing shard output is missing or damaged: {path}")
    return report


def process_shard(index: int, shard_path: str, output_dir: str, config: dict) -> dict:
    destination = Path(output_dir)
    report_path = destination / f"shard-{index:05d}.report.json"
    previous = report_is_valid(report_path, config)
    if previous:
        return previous
    tokenizer = get_tokenizer(config["tokenizer"], config["template"])
    outputs = {
        kind: JsonlWriter(destination / f"{kind}-{index:05d}.jsonl")
        for kind in ("train", "validation", "warmup")
    }
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    parquet = pq.ParquetFile(shard_path)
    if not {"id", "messages", "source", "tools"}.issubset(parquet.schema_arrow.names):
        raise ValueError(f"Required columns missing from {shard_path}")
    for batch in parquet.iter_batches(
        batch_size=128, columns=["id", "messages", "source", "tools"]
    ):
        for row in batch.to_pylist():
            if config["max_rows_per_shard"] and counts["source_rows"] >= config["max_rows_per_shard"]:
                break
            counts["source_rows"] += 1
            source = str(row.get("source") or "<missing>")
            source_counts = by_source.setdefault(source, Counter())
            if row.get("tools"):
                counts["tool_declared"] += 1
                source_counts["tool_declared"] += 1
                continue
            messages = row.get("messages")
            try:
                kind = source_kind(messages)
            except (KeyError, TypeError, ValueError):
                counts["malformed_or_unsupported"] += 1
                source_counts["malformed_or_unsupported"] += 1
                continue
            counts[kind] += 1
            source_counts[kind] += 1
            if kind != "direct":
                continue
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_assistant_tokens_mask=True,
                )
                ids = rendered["input_ids"]
                mask = rendered.get("assistant_masks")
                if mask is None:
                    mask = rendered.get("assistant_tokens_mask")
                if mask is None or len(ids) != len(mask) or not 0 < sum(mask) < len(mask) or mask[0]:
                    raise ValueError("invalid assistant-only loss mask")
                target_ids = [token for token, active in zip(ids, mask) if active]
                if any(token in THINK_IDS for token in target_ids):
                    # This is the historical bug. Never quarantine and proceed.
                    raise RuntimeError(
                        f"FATAL: direct row {row.get('id')} from {source} has a masked think token"
                    )
                if len(ids) > config["max_length"]:
                    counts["overlength"] += 1
                    source_counts["overlength"] += 1
                    continue
                target = tokenizer.decode(target_ids, skip_special_tokens=False)
                if messages[-1]["content"].strip() not in target or "<|im_start|>user" in target:
                    raise ValueError("final answer missing or user text in assistant labels")
            except RuntimeError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                counts["render_rejected"] += 1
                source_counts["render_rejected"] += 1
                if counts["render_rejected"] <= 5:
                    print(f"shard {index} render-rejected {row.get('id')}: {exc}", flush=True)
                continue
            digest = canonical_prompt(messages)
            split = "validation" if int(digest[:16], 16) % 1000 == 0 else "train"
            item = {
                "id": row.get("id"), "source": f"{REPO}:{source}",
                "source_revision": REVISION, "prompt_sha256": digest,
                "rendered_tokens": len(ids), "supervised_tokens": len(target_ids),
                "messages": messages,
            }
            outputs[split].write(item)
            counts[f"{split}_supervised_tokens"] += len(target_ids)
            source_counts[f"{split}_admitted"] += 1
            if split == "train" and int(digest[16:32], 16) % 16 == 0:
                outputs["warmup"].write(item)
                counts["warmup_supervised_tokens"] += len(target_ids)
        if config["max_rows_per_shard"] and counts["source_rows"] >= config["max_rows_per_shard"]:
            break
    files = {kind: writer.finish() for kind, writer in outputs.items()}
    report = {
        "shard": index, "input": shard_path, "config": config,
        "counts": dict(counts),
        "by_source": {name: dict(value) for name, value in by_source.items()},
        "files": files,
    }
    partial = report_path.with_name(report_path.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excellent-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-rows-per-shard", type=int, default=0)
    args = parser.parse_args()
    if args.max_length < 1 or args.workers < 1 or args.max_shards < 0 or args.max_rows_per_shard < 0:
        parser.error("length/workers must be positive and limits nonnegative")
    shards = sorted((args.excellent_dir / "data").glob("train-*.parquet"))
    if len(shards) != EXPECTED_SHARDS:
        raise ValueError(f"Expected {EXPECTED_SHARDS} Excellent shards, got {len(shards)}")
    if sum(path.stat().st_size for path in shards) != EXPECTED_BYTES:
        raise ValueError("Pinned Excellent shard byte total differs")
    if sum(pq.ParquetFile(path).metadata.num_rows for path in shards) != EXPECTED_ROWS:
        raise ValueError("Pinned Excellent metadata row count differs")
    if not args.tokenizer.is_dir() or not args.template.is_file():
        raise FileNotFoundError("tokenizer or template missing")
    # Check the exact template and tokenizer before launching worker processes.
    get_tokenizer(str(args.tokenizer), str(args.template))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = shards[:args.max_shards] if args.max_shards else shards
    config = {
        "repo": REPO, "revision": REVISION,
        "tokenizer": str(args.tokenizer), "template": str(args.template),
        "template_sha256": file_sha256(args.template),
        "tokenizer_json_sha256": file_sha256(args.tokenizer / "tokenizer.json"),
        "max_length": args.max_length, "max_rows_per_shard": args.max_rows_per_shard,
    }
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(process_shard, index, str(path), str(args.output_dir), config): index
            for index, path in enumerate(selected)
        }
        for future in as_completed(jobs):
            report = future.result()
            reports.append(report)
            print(f"Finished shard {report['shard']}: {report['counts']}", flush=True)
    reports.sort(key=lambda item: item["shard"])
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    files = {"train": [], "validation": [], "warmup": []}
    for report in reports:
        counts.update(report["counts"])
        for source, values in report["by_source"].items():
            by_source.setdefault(source, Counter()).update(values)
        for kind, info in report["files"].items():
            files[kind].append(info)
    full_run = not args.max_shards and not args.max_rows_per_shard
    if full_run and counts["source_rows"] != EXPECTED_ROWS:
        raise ValueError(f"Scanned {counts['source_rows']} rows, expected {EXPECTED_ROWS}")
    if full_run and counts["direct"] != 2_649_634:
        raise ValueError(f"Direct count changed: {counts['direct']}")
    if not sum(item["rows"] for item in files["train"]) or not sum(item["rows"] for item in files["validation"]):
        raise ValueError("No train or validation examples admitted")
    if counts["render_rejected"] or counts["overlength"]:
        print(f"Filtered {counts['render_rejected']} render failures and {counts['overlength']} overlength rows", flush=True)
    manifest = {
        "kind": "full_stage1_direct_audited" if full_run else "engineering_smoke_not_production",
        "config": config, "counts": dict(counts),
        "by_source": {name: dict(value) for name, value in sorted(by_source.items())},
        "files": files,
        "invariant": "Every admitted assistant mask has zero <think>/</think> token IDs and a preserved final answer",
        "status": "candidate_internal_training_data_not_release_clearance",
    }
    path = args.output_dir / "manifest.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    print(json.dumps({"kind": manifest["kind"], "counts": manifest["counts"],
                      "files": {k: sum(i["rows"] for i in v) for k, v in files.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
