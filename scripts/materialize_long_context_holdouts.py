#!/usr/bin/env python3
"""Materialize deterministic long-context holdout rows.

This expands rows with metadata.materialization into full contexts. It is
intentionally tokenizer-free: `target_context_tokens` is approximated by whitespace
tokens so the output is stable in any environment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_IN = Path("data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/all.jsonl")


def token_count(text: str) -> int:
    return len(text.split())


def make_record(seed: str, idx: int, language: str, canary: str) -> str:
    rng = random.Random(f"{seed}:{idx}")
    code = rng.randrange(100000, 999999)
    return (
        f"record={idx:06d} language={language} canary={canary} "
        f"section={rng.choice(['alpha', 'beta', 'gamma', 'delta'])} "
        f"reference={code} status=background note=padding "
    )


def materialize(row: dict) -> dict:
    materialization = (row.get("metadata") or {}).get("materialization")
    if not materialization:
        return row

    target = int(materialization["target_context_tokens"])
    position = materialization["answer_position"]
    needle = materialization["needle"]
    seed = materialization["filler_seed"]

    prefix = f"{row['language']} long-context synthetic document. canary={row['canary']}\n"
    needle_text = (
        ""
        if position == "needle_absent"
        else f"\nIMPORTANT AUDIT REFERENCE: {needle}\n"
    )

    if position == "early":
        early_ratio = 0.08
    elif position == "middle":
        early_ratio = 0.50
    elif position == "late":
        early_ratio = 0.88
    else:
        early_ratio = 0.50

    before_target = int(target * early_ratio)
    chunks = [prefix]
    current_tokens = token_count(prefix)
    while current_tokens < before_target:
        chunk = make_record(seed, len(chunks), row["language"], row["canary"])
        chunks.append(chunk)
        current_tokens += token_count(chunk)

    if needle_text:
        chunks.append(needle_text)
        current_tokens += token_count(needle_text)

    idx = 0
    while current_tokens < target:
        chunk = make_record(seed, 100000 + idx, row["language"], row["canary"])
        chunks.append(chunk)
        current_tokens += token_count(chunk)
        idx += 1

    out = dict(row)
    out["context"] = "".join(chunks)
    out["metadata"] = dict(row.get("metadata") or {})
    out["metadata"]["materialized_context_tokens_approx"] = current_tokens
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-long-context", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if args.only_long_context and row.get("bucket") != "long_context_retrieval":
                continue
            dst.write(json.dumps(materialize(row), ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
