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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: {exc}") from exc
    return rows


def validate_grpo(rows: list[dict]) -> list[str]:
    errors = []
    ids = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in ids:
            errors.append(f"duplicate GRPO id: {row_id}")
        ids.add(row_id)
        for field in ["prompt", "question", "options", "answer", "language", "source_id", "provenance_hash"]:
            if not row.get(field):
                errors.append(f"{row_id}: missing {field}")
        if not LETTER_RE.match(str(row.get("answer", ""))):
            errors.append(f"{row_id}: invalid answer {row.get('answer')}")
        labels = [opt.get("label") for opt in row.get("options") or []]
        if row.get("answer") not in labels:
            errors.append(f"{row_id}: answer not in option labels")
        if row.get("reward_type") != "mcq_letter_exact":
            errors.append(f"{row_id}: unsupported reward_type {row.get('reward_type')}")
    return errors


def validate_dpo(rows: list[dict]) -> list[str]:
    errors = []
    ids = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in ids:
            errors.append(f"duplicate DPO id: {row_id}")
        ids.add(row_id)
        for field in ["prompt", "chosen", "rejected", "language", "source_id", "provenance_hash"]:
            if not row.get(field):
                errors.append(f"{row_id}: missing {field}")
        if row.get("chosen") == row.get("rejected"):
            errors.append(f"{row_id}: chosen equals rejected")
        if not LETTER_RE.match(str(row.get("chosen", ""))) or not LETTER_RE.match(str(row.get("rejected", ""))):
            errors.append(f"{row_id}: invalid chosen/rejected")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    all_grpo = []
    all_dpo = []
    for split in ["train", "validation", "test"]:
        all_grpo.extend(read_jsonl(args.dataset_dir / "grpo" / f"{split}.jsonl"))
        all_dpo.extend(read_jsonl(args.dataset_dir / "dpo" / f"{split}.jsonl"))

    errors = validate_grpo(all_grpo) + validate_dpo(all_dpo)
    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        raise SystemExit(1)

    summary = {
        "grpo_rows": len(all_grpo),
        "dpo_rows": len(all_dpo),
        "languages": dict(sorted(Counter(row["language"] for row in all_grpo).items())),
        "sources": dict(sorted(Counter(row["source_id"] for row in all_grpo).items())),
        "splits": dict(sorted(Counter(row["split"] for row in all_grpo).items())),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
