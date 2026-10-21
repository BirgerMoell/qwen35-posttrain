#!/usr/bin/env python3
"""Count structural Stage-1 eligibility across pinned Excellent-SFT shards.

This is a fast source-level survey, not a tokenization, license, quality, or
decontamination audit. It never copies conversation text into the report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_two_stage_sft import source_kind
from stage_excellent_sft_leonardo import EXPECTED_ROWS, EXPECTED_SHARDS, REVISION, REPO


def category(row: dict) -> str:
    if row.get("tools"):
        return "tool_declared"
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "malformed_messages"
    try:
        return source_kind(messages)
    except (KeyError, TypeError, ValueError):
        return "malformed_or_unsupported"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 surveys all rows; positive values are a dry run")
    args = parser.parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")

    import pyarrow.parquet as pq

    paths = sorted((args.dataset_dir / "data").glob("train-*.parquet"))
    if len(paths) != EXPECTED_SHARDS:
        raise ValueError(f"expected {EXPECTED_SHARDS} shards, got {len(paths)}")
    metadata_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    if metadata_rows != EXPECTED_ROWS:
        raise ValueError(f"metadata row count {metadata_rows} != {EXPECTED_ROWS}")

    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    row_count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        columns = [name for name in ("source", "messages", "tools") if name in available]
        if "source" not in columns or "messages" not in columns:
            raise ValueError(f"{path} lacks required source/messages columns")
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                source = str(row.get("source") or "<missing>")
                label = category(row)
                counts[label] += 1
                by_source[source][label] += 1
                row_count += 1
                if args.max_rows and row_count >= args.max_rows:
                    break
            if args.max_rows and row_count >= args.max_rows:
                break
        if args.max_rows and row_count >= args.max_rows:
            break
    if args.max_rows == 0 and row_count != EXPECTED_ROWS:
        raise ValueError(f"scanned {row_count} rows, expected {EXPECTED_ROWS}")

    report = {
        "dataset": REPO,
        "revision": REVISION,
        "surveyed_rows": row_count,
        "metadata_rows": metadata_rows,
        "structural_only": True,
        "counts": dict(sorted(counts.items())),
        "by_source": {name: dict(sorted(values.items()))
                      for name, values in sorted(by_source.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("dataset", "revision", "surveyed_rows", "counts")}, indent=2))


if __name__ == "__main__":
    main()
