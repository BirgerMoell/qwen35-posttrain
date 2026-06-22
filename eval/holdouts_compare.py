#!/usr/bin/env python3
"""Tabulate EU-holdout results across stages (base vs SFT vs DPO) — the European delta.

Reads the per-tag JSON written by scripts/eval_eu_holdouts.py (overall_acc, by_bucket,
by_language) and prints overall, per-bucket, and per-language accuracy with deltas vs base.
Positive delta = the post-training improved that language/task.

Usage:
  python eval/holdouts_compare.py --base base --stages sft,dpo
  python eval/holdouts_compare.py --results-dir eval/results/eu-holdouts --base base --stages sft
"""
from __future__ import annotations

import argparse
import json
import pathlib


def load(results_dir: pathlib.Path, tag: str) -> dict:
    p = results_dir / f"{tag}.json"
    return json.load(open(p)) if p.exists() else {}


def section(title: str, key: str, tags: list[str], data: dict, base_tag: str):
    base = data.get(base_tag, {}).get(key, {})
    others = [t for t in tags if t != base_tag]
    rows = sorted({k for t in tags for k in data.get(t, {}).get(key, {})})
    if not rows:
        return
    print(f"\n{title}")
    hdr = f"  {'':<22}{base_tag:>10}" + "".join(f"{t+' (Δ)':>16}" for t in others)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        b = base.get(r)
        cell_b = f"{b:.1f}" if b is not None else "—"
        line = f"  {r:<22}{cell_b:>10}"
        for t in others:
            v = data.get(t, {}).get(key, {}).get(r)
            if v is None:
                line += f"{'—':>16}"
            elif b is None:
                line += f"{v:.1f}{'':>11}"
            else:
                d = v - b
                line += f"{v:.1f} ({'+' if d >= 0 else ''}{d:.1f}){'':>4}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="eval/results/eu-holdouts")
    ap.add_argument("--base", default="base")
    ap.add_argument("--stages", default="sft,dpo")
    a = ap.parse_args()

    rdir = pathlib.Path(a.results_dir)
    tags = [a.base] + [s.strip() for s in a.stages.split(",") if s.strip()]
    data = {t: load(rdir, t) for t in tags}
    have = [t for t in tags if data[t]]
    if a.base not in have:
        print(f"No base results at {rdir}/{a.base}.json — run the base holdouts eval first.")
        return

    print("=" * 60)
    print("EU HOLDOUTS — accuracy %, (Δ vs base). Positive = improved.")
    print("=" * 60)
    print("\nOVERALL")
    for t in have:
        o = data[t].get("overall_acc")
        d = "" if t == a.base else f"  (Δ {o - data[a.base]['overall_acc']:+.1f})"
        print(f"  {t:<10} {o:.1f}%{d}")

    section("BY BUCKET (task type)", "by_bucket", have, data, a.base)
    section("BY LANGUAGE", "by_language", have, data, a.base)
    print("\nGate: EU tasks/languages should rise; reasoning_math must not collapse.")


if __name__ == "__main__":
    main()
