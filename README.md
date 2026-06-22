# qwen35-posttrain

End-to-end **post-training** of a useful 2026 base model on **LUMI** (AMD/ROCm), starting
from a long-context base and running the full modern pipeline: **SFT → DPO → RLVR (GRPO) →
tool-use/agentic → safety → eval/merge**.

> **▶ Reproducing this? Start with [`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — the single source
> of truth for environment, exact commands, and every gotcha (written for humans and AI agents).

- **Base model:** `Qwen3.5-9B-Base` (dense, native 262K context — long context is already in
  the base, so post-training must *preserve* it). Pipeline is validated on `Qwen3.5-2B-Base`
  first, then scaled to 9B.
- **Cluster:** LUMI (AMD MI250X, ROCm). Slurm. The hard constraint shaping every choice.
- **Goal:** a genuinely useful open artifact — strong general instruct + reasoning, with
  long-context preserved (optional multilingual/European and tool-use stages — see `docs/PLAN.md`).

> Sister project: long-context extension of the OELLM baby model
> ([`openeuro-longctx-datamix`](https://github.com/BirgerMoell/openeuro-longctx-datamix)).
> This repo is the *post-training-after-long-context* track.

## The pipeline (what runs, in order)

| Stage | Tool | Purpose | Status |
|------|------|---------|--------|
| 0. Pipeline smoke (2B) | TRL | prove the LUMI/ROCm stack end-to-end | TODO |
| 1. SFT | TRL `SFTTrainer` | instruction + chat + CoT + tool + long-ctx data | TODO |
| 2. DPO | TRL `DPOTrainer` | preference alignment (no reward model/rollouts) | TODO |
| 3. RLVR / GRPO | veRL (or TRL+vLLM) | math/code/logic with verifiable rewards | TODO |
| 4. Tool-use / agentic | TRL/veRL | function calling, multi-step trajectories | optional |
| 5. Safety | woven into 1–3 | refusals, harmlessness | ongoing |
| 6. Eval + merge | lm-eval/vLLM | MMLU, GSM8K/MATH, HumanEval, IFEval, RULER | each stage |

See **`docs/PLAN.md`** for the full design and **`docs/lumi_rocm_runbook.md`** for the
cluster specifics.

## Why these tools (LUMI/ROCm reality)
- **TRL** for SFT + DPO — simplest, ROCm-proven, low risk.
- **veRL** for GRPO/RLVR — AMD-Instinct-optimized, scales, official ROCm build. Needs a
  validation pass on **MI250X** (AMD's RL guides target the newer MI300X + ROCm 7.0).
- **vLLM (ROCm)** for RL rollouts and eval generation.

## Repo layout
```
configs/   per-stage configs (sft, dpo, grpo) + model/data refs
slurm/     LUMI ROCm sbatch templates (one per stage)
stages/    stage-specific code/launchers
data/      data-mix specs (SFT, preference, RLVR)
eval/      eval harness configs + scripts
docs/      PLAN.md (design), lumi_rocm_runbook.md (cluster)
```

## Principle: validate tiny first
Every stage is proven on **2B** with a few steps before committing real compute at 9B —
the approach that de-risked the long-context work. Compute on LUMI is scarce; experiments
are cheap, full runs are not.
