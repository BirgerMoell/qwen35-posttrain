#!/usr/bin/env python3
"""Validate oellm-eu-exam-mcq-v1 GRPO/DPO JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_DIR = Path("data/exam_mcq/oellm-eu-exam-mcq-v1")
LETTER_RE = re.compile(r"^[A-Z]$")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: {exc}") from exc


def validate_grpo_file(path: Path, ids: set[str], summary: dict, errors: list[str]) -> None:
    for row in iter_jsonl(path):
        row_id = row.get("id")
        if row_id in ids:
            errors.append(f"duplicate GRPO id: {row_id}")
        ids.add(row_id)
        for field in [
            "prompt",
            "question",
            "options",
            "answer",
            "language",
            "source_id",
            "source_license",
            "license_id",
            "license_category",
            "redistribution_status",
            "provenance_hash",
        ]:
            if not row.get(field):
                errors.append(f"{row_id}: missing {field}")
        if not LETTER_RE.match(str(row.get("answer", ""))):
            errors.append(f"{row_id}: invalid answer {row.get('answer')}")
        labels = [opt.get("label") for opt in row.get("options") or []]
        if row.get("answer") not in labels:
            errors.append(f"{row_id}: answer not in option labels")
        if row.get("reward_type") != "mcq_letter_exact":
            errors.append(f"{row_id}: unsupported reward_type {row.get('reward_type')}")
        summary["grpo_rows"] += 1
        summary["languages"][row["language"]] += 1
        summary["sources"][row["source_id"]] += 1
        summary["splits"][row["split"]] += 1
        summary["licenses"][row["license_id"]] += 1
        summary["redistribution_statuses"][row["redistribution_status"]] += 1


def validate_dpo_file(path: Path, ids: set[str], summary: dict, errors: list[str]) -> None:
    for row in iter_jsonl(path):
        row_id = row.get("id")
        if row_id in ids:
            errors.append(f"duplicate DPO id: {row_id}")
        ids.add(row_id)
        for field in [
            "prompt",
            "chosen",
            "rejected",
            "language",
            "source_id",
            "source_license",
            "license_id",
            "redistribution_status",
            "provenance_hash",
        ]:
            if not row.get(field):
                errors.append(f"{row_id}: missing {field}")
        if row.get("chosen") == row.get("rejected"):
            errors.append(f"{row_id}: chosen equals rejected")
        if not LETTER_RE.match(str(row.get("chosen", ""))) or not LETTER_RE.match(str(row.get("rejected", ""))):
            errors.append(f"{row_id}: invalid chosen/rejected")
        summary["dpo_rows"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    summary = {
        "grpo_rows": 0,
        "dpo_rows": 0,
        "languages": Counter(),
        "sources": Counter(),
        "splits": Counter(),
        "licenses": Counter(),
        "redistribution_statuses": Counter(),
    }
    errors: list[str] = []
    grpo_ids: set[str] = set()
    dpo_ids: set[str] = set()
    for split in ["train", "validation", "test"]:
        validate_grpo_file(args.dataset_dir / "grpo" / f"{split}.jsonl", grpo_ids, summary, errors)
        validate_dpo_file(args.dataset_dir / "dpo" / f"{split}.jsonl", dpo_ids, summary, errors)

    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        raise SystemExit(1)

    printable = {
        "grpo_rows": summary["grpo_rows"],
        "dpo_rows": summary["dpo_rows"],
        "languages": dict(sorted(summary["languages"].items())),
        "sources": dict(sorted(summary["sources"].items())),
        "splits": dict(sorted(summary["splits"].items())),
        "licenses": dict(sorted(summary["licenses"].items())),
        "redistribution_statuses": dict(sorted(summary["redistribution_statuses"].items())),
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
