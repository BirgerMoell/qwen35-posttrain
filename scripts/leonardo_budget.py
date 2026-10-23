#!/usr/bin/env python3
"""Local-h spent on Leonardo by the current user since a date, by job-name category (billing-weighted ElapsedRaw).

Run on a login node (python 3.6 compatible): python3 leonardo_budget.py
"""
import re, collections, subprocess, os
out = subprocess.run(["sacct", "-u", os.environ["USER"], "-A", os.environ["OELLM_ACCOUNT"], "-S", "2026-10-20", "-X", "-P", "-n",
                      "-o", "JobID,JobName,ElapsedRaw,AllocTRES"], stdout=subprocess.PIPE, universal_newlines=True).stdout
rules = (("stage1", "Stage 1 SFT"), ("stage2-reason", "Stage 2 SFT"), ("stage3-tooluse", "Stage 3 SFT"),
         ("stage3-post", "Stage 3 post/export"), ("dpo", "Stage 4 DPO"), ("think", "RL runs"), ("grpo", "RL runs"),
         ("oellm-eval", "oellm-eval"), ("reasoning-quick", "quick reasoning eval"), ("europa", "EUROPA evals"),
         ("translate", "translation"), ("tool", "tool data/eval"), ("evalplus", "code eval"), ("repair", "tag repair"),
         ("build", "data builds"), ("stage", "data builds"))
total, by = 0.0, collections.Counter()
for line in out.splitlines():
    if not line.strip():
        continue
    jid, name, el, tres = line.split("|")[:4]
    m = re.search(r"billing=(\d+)", tres)
    lh = int(el or 0) / 3600 * (int(m.group(1)) if m else 0)
    total += lh
    by[next((v for k, v in rules if k in name.lower()), "other: " + name[:24])] += lh
print(f"TOTAL local-h since 2026-10-20: {total:,.0f}")
for k, v in by.most_common():
    print(f"  {k:30s} {v:8,.0f}")
