#!/usr/bin/env python3
"""Eval gate: exit non-zero if a stage regressed, to halt the Slurm dependency chain.

Reads this stage's metrics.json and (optionally) the baseline tag's metrics, and enforces:
  - long-context (RULER) must not drop more than RULER_TOL points  (protect the 262K context)
  - core score (MMLU/IFEval avg) must not drop more than CORE_TOL points
Tune thresholds per project policy. SKELETON — wire to real metric keys from run_eval.py.
"""
import argparse, json, sys, pathlib

RULER_TOL = 2.0   # pts
CORE_TOL = 1.0    # pts

def load(p):
    try: return json.load(open(p))
    except Exception: return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--baseline-tag", default=None)
    a = ap.parse_args()
    cur = load(a.metrics)
    if not cur:
        print(f"[gate] no metrics at {a.metrics} — failing closed"); sys.exit(1)
    if a.baseline_tag:
        base = load(pathlib.Path(a.metrics).parent.parent / a.baseline_tag / "metrics.json")
        for key, tol in (("ruler", RULER_TOL), ("core", CORE_TOL)):
            if key in cur and key in base and cur[key] < base[key] - tol:
                print(f"[gate] REGRESSION {key}: {cur[key]:.2f} < {base[key]:.2f}-{tol}"); sys.exit(1)
    print("[gate] PASS", cur); sys.exit(0)

if __name__ == "__main__":
    main()
