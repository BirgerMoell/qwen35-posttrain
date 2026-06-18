#!/usr/bin/env python3
"""Run the eval suite for a checkpoint and write metrics.json. SKELETON.

Intended: serve $model with vLLM (ROCm) and run lm-evaluation-harness tasks + RULER (long-ctx)
+ OneRuler (multilingual). Aggregate into {core, ruler, ...} for the gate (eval/check_gate.py).
"""
import argparse, json

SUITES = {
    "core":  ["mmlu", "ifeval", "gsm8k", "math", "humaneval"],
    "ruler": ["ruler_16k", "ruler_64k", "ruler_128k"],          # long-context — must not regress
    "multi": ["oneruler"],                                       # if multilingual focus on
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="core,ruler")
    a = ap.parse_args()
    tasks = [t for s in a.suite.split(",") for t in SUITES.get(s, [])]
    print(f"[eval] model={a.model} tasks={tasks}")
    # TODO: launch vLLM (ROCm) server + run lm-eval-harness/RULER; collect real scores.
    metrics = {"core": None, "ruler": None, "_tasks": tasks, "_TODO": "wire lm-eval + RULER via vLLM-ROCm"}
    json.dump(metrics, open(a.out, "w"), indent=2)
    print(f"[eval] wrote {a.out}")

if __name__ == "__main__":
    main()
