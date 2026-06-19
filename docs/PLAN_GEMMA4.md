# Gemma 4 Fine-Tuning Plan — EU Model

**Model family:** Google Gemma 4 (released April 2026)  
**License:** Apache 2.0 — fine-tuned derivatives can be redistributed freely  
**Target:** Best open European language model in the Gemma 4 family

---

## Why Gemma 4

- **Strongest EU multilingual pretraining** at this scale — Google's data pipeline has
  the broadest European language coverage of any open family
- **Native `<|think|>` reasoning** built into the instruct model — no separate reasoning
  SFT stage needed if starting from Gemma-4-Instruct
- **Full size ladder in one family:** 2B → 4B → 9B → 27B — iterate cheap, scale without
  changing pipeline or retuning hyperparameters
- **Apache 2.0** — cleanest license for open redistribution of fine-tuned models
- **AMD Day 0 support** — confirmed for ROCm 6.4, our LUMI container works

---

## Sizes and iteration strategy

| Model | Params | VRAM (QLoRA) | Use |
|---|---|---|---|
| `google/gemma-4-e2b-it` | 2B (5B w/ embeddings) | ~6 GB | Pipeline debugging, minutes per run |
| `google/gemma-4-e4b-it` | 4B (8B w/ embeddings) | ~10 GB | EU delta ablations, cheap sweeps |
| `google/gemma-4-9b-it` (approx) | ~9B | ~20 GB | **Main experiment** |
| `google/gemma-4-27b-it` | 27B | ~50 GB (QLoRA) | Production artifact |

---

## Key technical constraints

### DPO is NOT supported on Gemma in TRL
TRL's DPOTrainer has a known incompatibility with Gemma models. Use instead:
- **SimPO** (`trl.SimPOTrainer`) — simpler, no reference model, works on Gemma
- **KTO** (`trl.KTOTrainer`) — binary feedback, works on Gemma
- Both are in TRL 0.28 (our container version)

### Chat template
Gemma 4 uses a new `<|turn>` delimiter format. Always use:
```python
tokenizer.apply_chat_template(messages, tokenize=False)
```
Never hand-roll the prompt format — easy to get wrong and silently breaks training.

### Reasoning is already in the instruct model
Gemma-4-Instruct has `<|think|>` reasoning built in. Starting from the instruct model
means the reasoning SFT stage can be skipped or significantly reduced — we only need to
add EU-language reasoning traces, not establish reasoning from scratch.

### Full fine-tuning on LUMI (not QLoRA)
LUMI nodes have 8 MI250X GCDs × 64 GB HBM2e = **512 GB per node**. Full fine-tuning
with FSDP ZeRO-3 is completely feasible and preferred over QLoRA:

| Model | Full FT memory | GCDs needed | Nodes |
|---|---|---|---|
| 4B | ~32 GB total | 1 GCD (comfortable) | 1 |
| 9B | ~108 GB (weights + optimizer + grads) | 2 GCDs | 1 |
| 27B | ~324 GB | 6 GCDs | 1 |
| 72B+ | ~800 GB+ | 13+ GCDs | 2+ |

Full fine-tuning advantages over QLoRA:
- No quantization error in the base model
- Can update all parameters including embeddings (important for EU language tokens)
- Better performance, especially for multilingual coverage
- No adapter merging step before inference

Use **LoRA only** when doing very rapid hyperparameter sweeps on dev-g (30-min limit).

---

## Pipeline

```
Gemma-4-{size}-Instruct  (start from instruct, not base)
        │
        ▼
Stage 1 — EU Multilingual SFT  (TRL SFTTrainer + QLoRA)
  Data: Dolci-Instruct-SFT (70 langs, Propella-filtered)
        + LightOn multilingual reasoning SFT (EU-language think traces)
        + project longctx corpus (native EU PDFs — SFT on real EU text)
        + ~15% English replay (prevent forgetting)
  Seq length: 8192 (raise to 32K for longctx subset)
  LR: 2e-5, cosine scheduler, warmup 0.03, bfloat16
  Max steps: 1000 (smoke: 20)
        │
        ▼
Stage 2 — EU Preference Alignment  (TRL SimPOTrainer)
  Data: Dolci-Instruct-DPO-translated (11 langs) formatted for SimPO
        + anti-translationese pairs (AI Sweden — prefer native over roundtrip)
        + on-policy rejected responses generated from Stage 1 checkpoint
  Note: NOT DPO — use SimPO (no reference model, works on Gemma)
        │
        ▼
Stage 3 — Light RLVR  (optional, if ROCm vLLM stack stable)
  Data: RLVR-GSM-MATH-IF + EU-language IF constraints
  Method: GRPO via TRL or veRL
        │
        ▼
Eval + release
```

---

## Recommended hyperparameters (full fine-tuning, FSDP ZeRO-3)

| Param | SFT | SimPO |
|---|---|---|
| Learning rate | 2e-5 | 5e-7 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |
| Global batch size | 128 | 64 |
| Per-device batch size | 2–4 | 2 |
| Gradient accumulation | varies | varies |
| Precision | bfloat16 | bfloat16 |
| Gradient checkpointing | true | true |
| Max seq length | 8192 | 4096 |
| Weight decay | 0.001 | 0.01 |
| FSDP sharding | FULL_SHARD (ZeRO-3) | FULL_SHARD |
| SimPO gamma | — | 0.5 |
| SimPO beta | — | 2.0 |

---

## Data mix (Stage 1 SFT)

| Source | Weight | Notes |
|---|---|---|
| Dolci-Instruct-SFT (Propella-filtered) | 50% | Core EU instruction quality |
| LightOn multilingual reasoning traces | 20% | EU-language `<|think|>` traces |
| Project longctx corpus (native EU PDFs) | 15% | Real EU text — key differentiator |
| English replay (original Dolci-Instruct EN) | 15% | Anti-forgetting |

---

## Eval gates (Gemma 4 baseline = zero-shot before fine-tuning)

| Metric | Gate |
|---|---|
| ArenaHard-EU | Must improve vs base |
| LCB (LiveCodeBench) | Must not drop > 3 pts |
| GSM8K / MATH | Must not drop > 2 pts |
| RULER-32K (EU langs) | Must improve vs base |
| Per-language checks (sv/fi/fr/de) | All must be ≥ base |

---

## LUMI submission

```bash
# Stage model to LUMI first (login node)
huggingface-cli download google/gemma-4-e2b-it \
  --local-dir /scratch/project_465002530/users/bmoell/models/gemma-4-e2b-it

# Run smoke (2B, 20 steps)
SMOKE=1 BASE_MODEL=/scratch/project_465002530/users/bmoell/models/gemma-4-e2b-it \
  bash pipeline/run_pipeline.sh 2b

# Run main (use e4b or larger once smoke passes)
BASE_MODEL=/scratch/project_465002530/users/bmoell/models/gemma-4-e4b-it \
  bash pipeline/run_pipeline.sh 4b
```

---

## Known gotchas

1. **DPO → SimPO** — do not use DPOTrainer with Gemma, use SimPOTrainer
2. **Chat template** — always use `apply_chat_template()`, never manual
3. **bfloat16 only** — other precisions can cause instability on ROCm
4. **Liger kernels** — can cause illegal memory access; disable if issues arise
5. **Packing** — set `packing=False` for stability; padding overhead is acceptable
6. **Vision layers** — if using multimodal variant, keep `finetune_vision_layers=False`
   initially to avoid forgetting visual capabilities
