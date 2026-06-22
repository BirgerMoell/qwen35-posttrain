#!/usr/bin/env python3
"""Validate the OpenEuroLLM EU eval holdout JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DIR = Path("data/eval_holdouts/oellm-eu-eval-holdouts-v1")

REQUIRED_FIELDS = {
    "id",
    "dataset_id",
    "version",
    "split",
    "language",
    "language_name",
    "bucket",
    "task_type",
    "source_type",
    "source_url",
    "source_doc_id",
    "source_license",
    "template_id",
    "canary",
    "prompt",
    "context",
    "expected_answer",
    "scoring",
    "rubric",
    "denylist",
    "contamination_signature",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def validate(rows: list[dict], min_per_language: int, expected_buckets: int) -> dict:
    errors = []
    ids = set()
    signatures = set()
    by_lang = Counter()
    by_bucket = Counter()
    by_split = Counter()
    by_lang_bucket = defaultdict(Counter)

    for row in rows:
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append(f"{row.get('id', '<missing id>')}: missing fields {missing}")

        row_id = row.get("id")
        if row_id in ids:
            errors.append(f"{row_id}: duplicate id")
        ids.add(row_id)

        signature = row.get("contamination_signature")
        if signature in signatures:
            errors.append(f"{row_id}: duplicate contamination_signature")
        signatures.add(signature)

        if row.get("split") == "train":
            errors.append(f"{row_id}: holdout dataset must not contain train split")
        if row.get("source_type") != "synthetic_canary":
            errors.append(f"{row_id}: public v1 rows must be synthetic_canary")
        if not str(row.get("source_url", "")).startswith("synthetic://"):
            errors.append(f"{row_id}: source_url must be synthetic:// for public v1")

        denylist = row.get("denylist") or {}
        if row.get("source_doc_id") not in denylist.get("source_doc_ids", []):
            errors.append(f"{row_id}: denylist must include source_doc_id")
        if row.get("template_id") not in denylist.get("template_ids", []):
            errors.append(f"{row_id}: denylist must include template_id")
        if denylist.get("canary") != row.get("canary"):
            errors.append(f"{row_id}: denylist canary mismatch")

        if row.get("bucket") == "long_context_retrieval":
            materialization = (row.get("metadata") or {}).get("materialization")
            if not materialization:
                errors.append(f"{row_id}: long-context row missing materialization metadata")
            elif materialization.get("target_context_tokens", 0) < 32768:
                errors.append(f"{row_id}: long-context target too short")

        by_lang[row.get("language")] += 1
        by_bucket[row.get("bucket")] += 1
        by_split[row.get("split")] += 1
        by_lang_bucket[row.get("language")][row.get("bucket")] += 1

    for lang, count in by_lang.items():
        if count < min_per_language:
            errors.append(f"{lang}: only {count} rows, expected at least {min_per_language}")
        if len(by_lang_bucket[lang]) != expected_buckets:
            errors.append(f"{lang}: has {len(by_lang_bucket[lang])} buckets, expected {expected_buckets}")

    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        raise SystemExit(1)

    return {
        "rows": len(rows),
        "languages": len(by_lang),
        "buckets": len(by_bucket),
        "splits": dict(sorted(by_split.items())),
        "by_language": dict(sorted(by_lang.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--min-per-language", type=int, default=200)
    parser.add_argument("--expected-buckets", type=int, default=10)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset_dir / "data" / "all.jsonl")
    summary = validate(rows, args.min_per_language, args.expected_buckets)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
