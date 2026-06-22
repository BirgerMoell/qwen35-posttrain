# eval/

Run after **every** stage to track the EU delta and catch regressions. Baseline the
**base model** first, then diff each post-training stage (SFT, DPO) against it.

## What to measure

| Purpose | Tasks | Direction |
|---|---|---|
| **Regression guard** | `mmlu`, `gsm8k`, `arc_challenge`, `hellaswag` | must NOT drop > ~2 pts |
| **Instruction following** | `ifeval` | should improve |
| **EU multilingual** | `m_mmlu_{sv,de,fr,…}`, ArenaHard-EU (judge) | should improve |
| **Long context** | RULER / OneRuler | must NOT regress |

## How to run (LUMI)

lm-evaluation-harness via the transformers 5.x overlay (HF backend — loads qwen3_5 / gemma4;
vLLM 0.15.1 may not support the 2026 architectures yet).

```bash
# 1. Baseline the base model
sbatch --export=ALL,MODEL=/scratch/.../models/Qwen3.5-9B,TAG=base \
  slurm/eval_lmeval_lumi.sbatch

# 2. After SFT
sbatch --export=ALL,MODEL=/scratch/.../output/qwen35-9b-sft,TAG=sft \
  slurm/eval_lmeval_lumi.sbatch

# 3. After DPO
sbatch --export=ALL,MODEL=/scratch/.../output/qwen35-9b-dpo,TAG=dpo \
  slurm/eval_lmeval_lumi.sbatch

# Quick smoke (cap examples): add LIMIT=100 to --export

# 4. Compare
python eval/compare.py --base base --stages sft,dpo
```

Results land in `eval/results/<tag>/`. `compare.py` prints a table of score% with deltas
vs base so the EU gains and any regressions are visible at a glance.

## Files
- `slurm/eval_lmeval_lumi.sbatch` — run lm-eval on one checkpoint (MODEL/TAG/TASKS/LIMIT)
- `compare.py` — tabulate base vs stages with deltas
- `check_gate.py` — pass/fail gate for the orchestrated pipeline (regression thresholds)
- `run_eval.py` — legacy stub (superseded by the sbatch + lm-eval)

## Next (EU delta judge)
ArenaHard-EU (`openeurollm/ArenaHard-EU-v0`, eng/fin subsets staged on LUMI) needs an
LLM-judge head-to-head vs base. Add as a second eval once the standard suite is green.
