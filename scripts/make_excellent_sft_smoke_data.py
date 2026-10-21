#!/usr/bin/env python3
"""Fetch a few public Excellent-SFT rows for preflight, never a training mix.

The dataset viewer serves the current revision. We require its SHA to match the
pinned revision before using these rows. This script is only for a tiny smoke;
full training must use pinned Parquet shards and a separate curation pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from audit_two_stage_sft import source_kind


DATASET = "openeurollm/OpenEuroLLM-Excellent-SFT"
API = "https://huggingface.co/api/datasets/" + DATASET
ROWS = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stage", type=int, default=4)
    parser.add_argument("--rows-per-offset", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.per_stage <= 16 or not 1 <= args.rows_per_offset <= 100:
        raise ValueError("this is a small smoke only (per-stage <=16, rows-per-offset <=100)")

    session = requests.Session()
    metadata = session.get(API, timeout=30)
    metadata.raise_for_status()
    actual_sha = metadata.json()["sha"]
    if actual_sha != args.revision:
        raise ValueError(f"dataset HEAD changed: {actual_sha} != pinned {args.revision}")

    selected: dict[str, list[dict]] = {"instruct": [], "reasoning": []}
    sources: dict[str, dict[str, int]] = {"instruct": {}, "reasoning": {}}
    for offset in (1000, 5000000, 1000000, 3000000):
        reply = session.get(ROWS, params={
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": args.rows_per_offset,
        }, timeout=60)
        reply.raise_for_status()
        for item in reply.json()["rows"]:
            row = item["row"]
            if row.get("tools"):
                continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            try:
                kind = source_kind(messages)
            except (KeyError, TypeError, ValueError):
                continue
            stage = "reasoning" if kind == "reasoning" else "instruct"
            if len(selected[stage]) >= args.per_stage:
                continue
            if any(len(message["content"]) > 12000 for message in messages):
                continue
            selected[stage].append({"id": row["id"], "source": row["source"], "messages": messages})
            source = row["source"]
            sources[stage][source] = sources[stage].get(source, 0) + 1
    if any(len(rows) < args.per_stage for rows in selected.values()):
        raise ValueError({stage: len(rows) for stage, rows in selected.items()})

    args.output.mkdir(parents=True, exist_ok=True)
    for stage, rows in selected.items():
        with (args.output / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": DATASET, "revision": actual_sha,
                      "counts": {k: len(v) for k, v in selected.items()},
                      "sources": sources, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
