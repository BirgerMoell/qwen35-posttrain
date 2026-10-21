#!/usr/bin/env python3
"""Fail-closed validation for build_long_context_sft.py output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--chat-template", type=Path, default=None)
    parser.add_argument("--retokenize-samples", type=int, default=32)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    if not args.parquet.is_file():
        raise SystemExit(f"missing parquet: {args.parquet}")
    manifest_path = args.manifest or args.parquet.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        if args.chat_template:
            tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")

    errors: list[str] = []
    task_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    language_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    rows = 0
    sampled = 0

    parquet = pq.ParquetFile(args.parquet)
    required = {
        "messages",
        "lang",
        "task",
        "target_tokens",
        "actual_tokens",
        "reasoning_trace",
        "evidence_depths",
        "example_id",
    }
    # Parquet's physical schema exposes nested leaf names (role/content/element);
    # use the Arrow schema for the logical top-level training columns.
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            rows += 1
            prefix = row.get("example_id") or f"row-{rows}"
            messages = row.get("messages") or []
            if len(messages) != 2:
                errors.append(f"{prefix}: expected exactly two messages")
                continue
            if [message.get("role") for message in messages] != ["user", "assistant"]:
                errors.append(f"{prefix}: roles must be user,assistant")
            if not str(messages[0].get("content") or "").strip():
                errors.append(f"{prefix}: empty prompt")
            if not str(messages[1].get("content") or "").strip():
                errors.append(f"{prefix}: empty answer")
            target = int(row["target_tokens"])
            actual = int(row["actual_tokens"])
            if actual > target:
                errors.append(f"{prefix}: actual_tokens {actual} exceeds target {target}")
            if actual < int(target * 0.75):
                errors.append(f"{prefix}: actual_tokens {actual} is below 75% of target {target}")
            evidence_depths = row.get("evidence_depths") or []
            if not evidence_depths or any(not 0 <= value <= 1 for value in evidence_depths):
                errors.append(f"{prefix}: invalid evidence depths {evidence_depths}")
            if len(evidence_depths) >= 3 and (
                min(evidence_depths) > 0.20 or max(evidence_depths) < 0.80
            ):
                errors.append(
                    f"{prefix}: multi-evidence row does not cover early and late context"
                )
            if prefix in seen_ids:
                errors.append(f"{prefix}: duplicate example id")
            seen_ids.add(prefix)
            task_counts[row["task"]] += 1
            length_counts[target] += 1
            language_counts[row["lang"]] += 1

            if tokenizer is not None and sampled < args.retokenize_samples:
                exact = len(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )
                )
                if exact != actual:
                    errors.append(
                        f"{prefix}: stored actual_tokens={actual}, retokenized={exact}"
                    )
                sampled += 1

    if rows != int(manifest["rows"]):
        errors.append(f"manifest rows={manifest['rows']} but parquet rows={rows}")
    expected_lengths = {int(key): value for key, value in manifest["counts"]["target_tokens"].items()}
    if dict(length_counts) != expected_lengths:
        errors.append(
            f"length counts differ: parquet={dict(length_counts)}, manifest={expected_lengths}"
        )
    if dict(task_counts) != manifest["counts"]["task"]:
        errors.append("task counts differ from manifest")
    if dict(language_counts) != manifest["counts"]["language"]:
        errors.append("language counts differ from manifest")

    summary = {
        "rows": rows,
        "tasks": dict(sorted(task_counts.items())),
        "lengths": dict(sorted(length_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "retokenized_samples": sampled,
        "errors": errors[:50],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"validation failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
