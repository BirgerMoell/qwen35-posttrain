# Post-Training Plan — Qwen3.5 base on LUMI

**Created:** 2026-06-18
**Owner:** Birger (assigned the post-training-after-long-context run)
**Base:** `Qwen3.5-9B-Base` (validate on `Qwen3.5-2B-Base` first)

## Goal
Take a strong 2026 long-context base and run the **full modern post-training pipeline** on
LUMI, producing a genuinely useful open artifact with long context preserved.

## Why Qwen3.5 base
- Released Feb–Mar 2026; dense sizes 0.8B/2B/4B/9B + MoE 27B/35B-A3B/122B-A10B + 397B.
- **Native 262K context** (extensible ~1M) → long context already present; post-training
  must not erode it (include long-context data in SFT/RL).
- 9B-Base = best usefulness/compute tradeoff on LUMI; 2B-Base = cheap stack validation.

## Pipeline (ordered)

### Stage 0 — Pipeline smoke (2B)
Run SFT→DPO→(tiny)GRPO on `Qwen3.5-2B-Base` with a few steps each to prove the LUMI/ROCm
stack works end-to-end (container, TRL, vLLM rollouts, checkpoint I/O, eval). Cheap.

### Stage 1 — SFT (TRL `SFTTrainer`)
- Data: general instruction + multi-turn chat + **CoT/reasoning traces** + **tool-use traces**
  + **long-context instruction data** (so 262K survives). Candidate open mixes: Tülu 3 SFT mix,
  OpenHermes, plus reasoning distillations.
- Keep a held-out eval set; checkpoint + eval.

### Stage 2 — DPO (TRL `DPOTrainer`)
- Preference pairs (e.g. UltraFeedback-style + on-policy pairs). Offline, no reward
  model/rollouts → cheapest strong alignment. Variants to consider: SimPO/KTO.

### Stage 3 — RLVR / GRPO (veRL, fallback TRL+vLLM)
- Verifiable rewards: math (GSM8K/MATH/Numina + verifier), code (tests), logic.
- GRPO (group-normalized, no value net). Needs **vLLM rollouts** on ROCm — the main
  MI250X validation risk. Interleave with rejection-sampling SFT if useful.

### Stage 4 — Tool-use / agentic (optional)
- SFT on tool/agent trajectories + RL with environment feedback (function calling).

### Stage 5 — Safety (woven through 1–3)
- Refusals, harmlessness preference/SFT data; red-team eval.

### Stage 6 — Eval + merge
- **Eval after every stage:** MMLU, GSM8K/MATH, HumanEval, IFEval, **RULER (long-ctx)**,
  + multilingual (OneRuler) if the multilingual focus is on.
- Optional **model merge** of specialized checkpoints (TIES/SLERP).

## Frameworks (LUMI/ROCm)
| Need | Choice | Risk |
|------|--------|------|
| SFT, DPO | **TRL** | low (ROCm-proven) |
| GRPO/RLVR | **veRL** (official ROCm build); fallback TRL+vLLM | medium — AMD RL guides target MI300X+ROCm7.0, LUMI is MI250X+ROCm6.4 |
| Rollouts/eval gen | **vLLM (ROCm)** | medium on MI250X |
| Container | extend LUMI laif-rocm container, or build a TRL/veRL/vLLM ROCm image | — |

## Open decisions
- **Artifact focus** (multi): general+reasoning (default) / multilingual-European (OELLM) /
  tool-use. Drives data mix + which RL stages are prioritized.
- Final size: 9B confirmed; consider 27B if compute allows after the stack is proven.
- Data licensing for an open artifact (use permissively-licensed SFT/preference sets).

## Compute budget & timeline
- **~300,000 GPU-hours** available; **hard deadline 2026-08-02** (~6.5 weeks from 2026-06-18).
- Rough costs (MI250X GCD ≈ "1 GPU", ~120 TFLOP/s realistic): SFT 9B @ ~5B tokens ≈ **~600 GPU-h**;
  DPO cheaper; **GRPO/RLVR is the cost driver** (rollouts) — budget a few thousand GPU-h.
- **Full pipeline at 9B ≈ low-thousands GPU-h; even 27B fits comfortably** in 300k.
- **Conclusion:** compute is *not* the binding constraint — **time and a working ROCm
  RL/vLLM stack are.** Front-load container + vLLM-on-MI250X validation; reserve the bulk of
  the clock for SFT/DPO (low-risk) and de-risk GRPO early so it's not a last-week scramble.

## Milestones (target before 2026-08-02)
1. LUMI ROCm post-training container with TRL+vLLM(+veRL) working (`docs/lumi_rocm_runbook.md`).
2. Stage-0 smoke on 2B passes end-to-end (SFT→DPO→tiny GRPO + eval).
3. SFT 9B + eval.
4. DPO 9B + eval.
5. GRPO/RLVR 9B + eval (gated on vLLM-ROCm working at scale).
6. Release artifact + model card; preserve long-context (RULER) throughout.
