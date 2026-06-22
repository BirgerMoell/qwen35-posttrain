#!/usr/bin/env python3
"""Tabulate lm-eval results across pipeline stages (base vs SFT vs DPO).

lm-evaluation-harness writes a results JSON under each eval/results/<tag>/ dir. This script
reads the headline metric per task for each tag and prints a comparison table with deltas
vs the base, so you can see the EU delta AND catch regressions at a glance.

Usage:
  python eval/compare.py --base base --stages sft,dpo
  python eval/compare.py --results-dir eval/results --base base --stages sft,dpo
"""
from __future__ import annotations

import argparse
import json
import pathlib

# Headline metric per task (lm-eval reports several; pick the standard one).
PREFERRED_METRICS = [
    "exact_match,strict-match",      # gsm8k
    "exact_match,flexible-extract",
    "prompt_level_strict_acc,none",  # ifeval
    "inst_level_strict_acc,none",
    "acc_norm,none",                 # arc, hellaswag
    "acc,none",                      # mmlu and most acc tasks
]


def find_results_json(tag_dir: pathlib.Path) -> pathlib.Path | None:
    # lm-eval writes results_*.json (sometimes nested under a model-name subdir)
    cands = sorted(tag_dir.rglob("results_*.json")) + sorted(tag_dir.rglob("results.json"))
    return cands[-1] if cands else None


def load_scores(tag_dir: pathlib.Path) -> dict[str, float]:
    p = find_results_json(tag_dir)
    if not p:
        return {}
    data = json.load(open(p))
    results = data.get("results", {})
    scores: dict[str, float] = {}
    for task, metrics in results.items():
        for key in PREFERRED_METRICS:
            if key in metrics and isinstance(metrics[key], (int, float)):
                scores[task] = float(metrics[key])
                break
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="eval/results")
    ap.add_argument("--base", default="base")
    ap.add_argument("--stages", default="sft,dpo", help="comma-separated tags to compare vs base")
    a = ap.parse_args()

    root = pathlib.Path(a.results_dir)
    tags = [a.base] + [s.strip() for s in a.stages.split(",") if s.strip()]
    by_tag = {t: load_scores(root / t) for t in tags}

    all_tasks = sorted({t for s in by_tag.values() for t in s})
    if not all_tasks:
        print(f"No results found under {root}/. Run eval_lmeval_lumi.sbatch first.")
        return

    base = by_tag.get(a.base, {})
    header = f"{'task':<22} " + " ".join(f"{t:>14}" for t in tags)
    print(header)
    print("-" * len(header))
    for task in all_tasks:
        row = f"{task:<22} "
        for t in tags:
            v = by_tag[t].get(task)
            if v is None:
                row += f"{'—':>14} "
                continue
            cell = f"{100*v:.1f}"
            if t != a.base and task in base:
                d = 100 * (v - base[task])
                cell += f"({'+' if d >= 0 else ''}{d:.1f})"
            row += f"{cell:>14} "
        print(row)

    print()
    print("Reading: each cell is score% with (delta vs base). Positive = improved.")
    print("Gate:    EU/instruction tasks should rise; reasoning/knowledge must not drop > ~2pts.")


if __name__ == "__main__":
    main()
