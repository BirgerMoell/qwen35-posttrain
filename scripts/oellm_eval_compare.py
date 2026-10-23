#!/usr/bin/env python3
"""Compare oellm-eval runs: per-benchmark means over the tasks every model ran (usage: oellm_eval_compare.py results.csv [...])."""
import csv, glob, json, sys, collections
fams = {"belebele": "acc", "global_mmlu": "acc", "include": "acc", "global_mgsm": "exact_match", "multiblimp": "acc_norm"}
def fam(t):
    return next((f for f in fams if t.startswith(f)), None)
models = collections.defaultdict(dict)
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        f = fam(r["task"])
        if f and fams[f] == r["metric_name"]:
            models[r["model_name"].rstrip("/").split("/")[-1]][(r["task"], r["n_shot"])] = float(r["performance"])
common = {f: set.intersection(*[{k for k in m if fam(k[0]) == f} for m in models.values()]) for f in fams}
print("tasks per benchmark (common to all models):", {f: len(v) for f, v in common.items()})
print(f"{'model':32s}" + "".join(f"{f:>13s}" for f in fams))
for name, m in sorted(models.items()):
    print(f"{name:32s}" + "".join(f"{sum(m[k] for k in common[f]) / len(common[f]):13.3f}" if common[f] else f"{'-':>13s}" for f in fams))
