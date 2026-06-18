# Post-Training Plan — Qwen3.5 base on LUMI

**Created:** 2026-06-18  
**Owner:** Birger / AI Sweden (T4.6 — long context extension, task management, delta-learning)  
**Base:** `Qwen3.5-9B-Base` (validate on `Qwen3.5-2B-Base` first)

## T4.6 Organisational context

T4.6 takes over from the pre-training part of OELLM after general pre-training. The pipeline:

```
General pre-training → [T4.6] Context Extension (128K) → SFT → DPO → RLVR
```

**Partner responsibilities (as of 2026-06):**
| Partner | Focus |
|---|---|
| AI Sweden (Birger) | **Long context extension**, task management, delta-learning / translationese |
| Silo AI (Jouni) | Long context extension (Megatron), post-training data translation |
| LightOn (Yakine/Kai) | Function calling, multilingual reasoning + RL |
| ELLIST (Arjun/Hannan) | SFT+DPO post-training framework, ablations, eval |

**Reference pipeline: Nemotron 3** (stronger published results, equally open recipe).  
Dolci suite retained as the data backbone (already translated, validated by ELLIST). Nemotron 3 drives pipeline *structure*: reasoning SFT first, then general SFT, DPO, RLVR with verifiable rewards.

## Goal
Run the full T4.6 post-training pipeline on LUMI using Qwen3.5 as a strong 2026 base,
producing an open artifact with multilingual capability and long context preserved.

## Why Qwen3.5 base
- Released Feb–Mar 2026; dense sizes 0.8B/2B/4B/9B + MoE 27B/35B-A3B/122B-A10B + 397B.
- **Native 262K context** (extensible ~1M) → long context already present; post-training
  must not erode it (include long-context data in SFT/RL).
- 9B-Base = best usefulness/compute tradeoff on LUMI; 2B-Base = cheap stack validation.

## Pipeline (ordered) — following Nemotron 3 structure

### Stage 0 — Smoke (2B, dev-g)
20-step SFT → 20-step DPO → 10-step GRPO on `Qwen3.5-2B-Base`. Proves the LUMI/ROCm
stack end-to-end before spending real compute.

### Stage 1 — Reasoning SFT  *(Nemotron 3: thinking SFT before general SFT)*
- **Data:** Dolci-Think-SFT-32B + OpenThoughts-114k + LightOn multilingual reasoning traces.
  Mix: ~50% Dolci-Think, ~30% OpenThoughts, ~20% code/math (Dolci-Think-Python, OpenMathInstruct).
- **Max seq:** 16K+ (reasoning traces are long). Packing on.
- **Why first:** Nemotron 3 shows reasoning capability is better established before general
  instruction tuning dilutes it. OLMo3 does the same think-SFT pass first.

### Stage 2 — General SFT  (TRL `SFTTrainer`)
- **Data:** Dolci-Instruct-SFT (70 langs) + long-ctx SFT (ChatQA2 + synthesized from project corpus).
- Keep a held-out eval set; checkpoint + eval after.

### Stage 3 — DPO  (TRL `DPOTrainer`)
- **Data:** Dolci-Instruct-DPO-translated (11 langs) + Dolci-Think-DPO for reasoning preference.
- Offline, no rollouts → cheapest strong alignment. Evaluate SimPO/KTO as variants.

### Stage 4 — RLVR / GRPO  (veRL; fallback TRL+vLLM)
- **Verifiable rewards:** math (GSM8K/MATH + verifier), code (unit tests), IF constraints.
- **Data:** RLVR-GSM-MATH-IF-Mixed-Constraints + Dolci-RL-Zero-Mix.
- Needs vLLM rollouts on ROCm — the main MI250X risk. Front-load container validation.

### Stage 5 — Tool-use / agentic  (optional, LightOn lead)
- SFT on tool trajectories + RL with environment feedback.

### Stage 6 — Eval + merge
- **After every stage:** MMLU, GSM8K/MATH, HumanEval, IFEval, **RULER** (long-ctx must not regress), OneRuler (multilingual).
- Optional checkpoint merge (TIES/SLERP) of specialised checkpoints.

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
