# Post-Training Techniques — 2026 landscape & our choices

Survey of modern DPO/GRPO variants (June 2026) and what we adopt for Qwen3.5-9B on LUMI/ROCm.
Drives `configs/` and the `scripts/*_train.py` entrypoints.

---

## Preference optimization — reference-free wins

The big practical point: **DPO needs a frozen reference model = 2 models in memory.** On one
LUMI node that repeatedly OOMs (the ref copy). **Reference-free methods load one model** and
also tend to match or beat DPO. So we switch.

| Method | Ref model? | Notes |
|---|---|---|
| DPO | **yes (2 models)** | the OOM source on our setup |
| IPO / sDPO | yes | not worth the memory cost for us |
| **SimPO** | **no** | length-normalized implicit reward + target margin; **beats DPO** on AlpacaEval-2/Arena-Hard. **Our choice.** |
| ORPO | no | one-stage (SFT+pref combined), no ref; good fallback |
| CPO | no | contrastive; SimPO is CPO+gamma in TRL |
| KTO | no | binary good/bad labels (no pairs needed) |

**Our choice: SimPO** via TRL `CPOTrainer(loss_type="simpo")` — `scripts/simpo_train.py`,
`configs/simpo_qwen35_9b.yaml`. One model → no OOM → and it's an upgrade over DPO, not a
workaround. Same `{prompt,chosen,rejected}` data works unchanged.

---

## RL with verifiable rewards — GSPO for Qwen

| Method | What | For us |
|---|---|---|
| GRPO | group-relative advantages, no value net | baseline; TRL `GRPOTrainer` (stable) |
| **GSPO** (Qwen, 2025) | **sequence-level** importance ratio + clipping | **best for Qwen / hybrid-MoE** — avoids GRPO's token-level variance & MoE "routing replay". Not yet in TRL → custom loss or **verl** |
| DAPO | decoupled clip + dynamic sampling | strong for long CoT; GRPOTrainer + custom |
| Dr.GRPO | removes length/std bias | cheap correctness fix |
| CISPO / RLOO / REINFORCE++ | lower-variance / value-free variants | situational |

**Our plan:** start with TRL **GRPOTrainer** on verifiable RLVR data (math/IF — already on
LUMI), then layer **GSPO-style sequence clipping** (better for Qwen). Consider **verl** if we
want GSPO/DAPO out of the box.

**Why GRPO is attractive here:** no preference pairs needed (just verifiers) → **sidesteps the
translationese problem** of translated DPO data, and optimizes EU-language *correctness*
directly. Cost is rollout generation, which is the GRPO gate ↓.

---

## The GRPO gate: vLLM rollouts for qwen3_5 on ROCm

GRPO needs fast generation. Findings:
- Container **vLLM 0.15.1 has NO native `qwen3_5`** (has Qwen3/Qwen3Next/Qwen3Moe/Qwen3VL); gemma4 also absent.
- Two ways forward, both being tested/available:
  1. **vLLM transformers backend** (`model_impl=transformers`) — runs qwen3_5 via our
     transformers-5.x overlay with vLLM's PagedAttention/batching. (`slurm/test_vllm_qwen35.sbatch`)
  2. **Newer vLLM** with native qwen3_5 + ROCm AITER attention (`VLLM_ROCM_USE_AITER=1`) —
     upstream added day-0 Qwen3.5 + first-class ROCm; install into overlay or newer container.
- Fallback for first validation: TRL GRPO with HF generation (slow but no vLLM dependency) —
  proves the loop, then swap in vLLM for throughput.

---

## Beyond GRPO (2026)
- **On-policy distillation (OPD)** — teacher supervises student on the student's own samples;
  dense token-level signal. Pairs with our distillation plan (`docs/DISTILLATION.md`).
- **Process reward models (PRMs)** — step-level rewards for reasoning/agents. Research-stage.
- No paradigm shift past GRPO; the field optimizes within RL (GSPO/DAPO) or pivots to OPD.

---

## Decisions adopted
- **Preference stage → SimPO** (reference-free; fixes DPO OOM; SOTA). `scripts/simpo_train.py`.
- **RL stage → GRPO now, GSPO next**, on verifiable RLVR data; gated on vLLM-ROCm rollouts.
- **Distillation** (`docs/DISTILLATION.md`) complements both: SFT on verified teacher traces,
  optionally OPD later.
