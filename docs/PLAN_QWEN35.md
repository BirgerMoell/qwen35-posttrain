# Qwen 3.5 Fine-Tuning Plan — EU Model

**Model family:** Alibaba Qwen 3.5 (2026)  
**License:** Apache 2.0 — fine-tuned derivatives can be redistributed freely  
**Role in multi-family strategy:** Reasoning fallback — strongest reasoning ceiling,
201-language pretraining, proven fine-tuning ecosystem

---

## Why Qwen 3.5

- **Strongest reasoning** of any open model at 9B scale (2026)
- **201 languages** in pretraining — broadest multilingual coverage of any candidate
- **Proven fine-tuning ecosystem** — most community fine-tuning work, well-documented HPs
- **Native 262K context** — long-context is already there, no extension needed
- **AMD Day 0 support** for ROCm on AMD Instinct (confirmed for Qwen 3.6, retroactive)
- **Dense sizes** at 0.8B / 2B / 4B / 9B / 27B — same iteration ladder as Gemma 4
- **MoE variants** (35B-A3B, 122B-A10B) if we want to scale efficiently beyond 27B

---

## Sizes and iteration strategy

| Model | Params | Memory (full FT, FSDP ZeRO-3) | Use |
|---|---|---|---|
| `Qwen/Qwen3.5-2B-Instruct` | 2B | ~24 GB | Pipeline debugging |
| `Qwen/Qwen3.5-4B-Instruct` | 4B | ~48 GB | EU delta ablations |
| `Qwen/Qwen3.5-9B-Instruct` | 9B | ~108 GB | **Main experiment** |
| `Qwen/Qwen3.5-27B-Instruct` | 27B | ~324 GB | Production artifact |
| `Qwen/Qwen3.5-35B-A3B-Instruct` | 35B (3B active) | ~420 GB total | Efficient large-scale |

On LUMI (512 GB HBM2e per node): all sizes up to 27B fit on a single node with FSDP ZeRO-3.

---

## Key technical notes

### DPO works on Qwen (unlike Gemma)
TRL's DPOTrainer is compatible with Qwen models. Use standard DPO pipeline:
- `trl.DPOTrainer` with Dolci-Instruct-DPO-translated
- Consider **SimPO** as well (no reference model needed, often stronger)

### Native reasoning via thinking mode
Qwen3.5-Instruct has a built-in thinking mode (toggle with `enable_thinking=True`).
Starting from Instruct means reasoning SFT is optional — focus SFT on EU-language
instruction quality, not establishing reasoning from scratch.

### Chat template
Qwen uses the standard ChatML format:
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```
Always use `tokenizer.apply_chat_template()`.

### MoE fine-tuning caveat
For MoE variants (35B-A3B), do **not** fine-tune the router layers — only the expert FFN
weights and attention. Set `modules_to_save` appropriately in PEFT config if using LoRA
for rapid sweeps.

---

## Pipeline

```
Qwen3.5-{size}-Instruct  (start from instruct)
        │
        ▼
Stage 1 — EU Multilingual SFT  (TRL SFTTrainer, full fine-tuning, FSDP ZeRO-3)
  Data: Dolci-Instruct-SFT (70 langs, Propella-filtered)
        + LightOn multilingual reasoning SFT
        + project longctx corpus (native EU PDFs)
        + ~15% English replay
  LR: 2e-5, cosine, warmup 0.03, bfloat16, seq_len 8192
        │
        ▼
Stage 2 — EU DPO / SimPO  (TRL DPOTrainer or SimPOTrainer)
  Data: Dolci-Instruct-DPO-translated (11 langs)
        + anti-translationese pairs (AI Sweden)
        + on-policy rejected responses from Stage 1 checkpoint
  Use SimPO for simplicity (no reference model); DPO as fallback
        │
        ▼
Stage 3 — RLVR  (optional, TRL GRPOTrainer or veRL)
  Data: RLVR-GSM-MATH-IF + Dolci-RL-Zero-Mix
  Verifiable rewards: math, IF constraints, EU-language code
        │
        ▼
Eval + release
```

---

## Recommended hyperparameters (full fine-tuning, FSDP ZeRO-3)

| Param | SFT | DPO | SimPO |
|---|---|---|---|
| Learning rate | 2e-5 | 5e-7 | 5e-7 |
| LR scheduler | cosine | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 | 0.03 |
| Global batch size | 128 | 64 | 64 |
| Per-device batch size | 2–4 | 2 | 2 |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| Gradient checkpointing | true | true | true |
| Max seq length | 8192 | 4096 | 4096 |
| Weight decay | 0.001 | 0.01 | 0.01 |
| DPO beta | — | 0.1 | — |
| SimPO gamma | — | — | 0.5 |
| SimPO beta | — | — | 2.0 |

Start with LR 2e-5 for SFT — Qwen is somewhat sensitive to LR; if loss is unstable, drop
to 1e-5. ELLIST has BO-swept DPO HPs for Qwen3-8B; request from Hannan if available.

---

## Data mix (Stage 1 SFT)

| Source | Weight | Notes |
|---|---|---|
| Dolci-Instruct-SFT (Propella-filtered) | 50% | Core EU instruction quality |
| LightOn multilingual reasoning traces | 20% | EU-language thinking traces |
| Project longctx corpus (native EU PDFs) | 15% | Real EU text |
| English replay | 15% | Anti-forgetting |

---

## Eval gates

| Metric | Gate |
|---|---|
| ArenaHard-EU | Must improve vs Qwen3.5 base |
| LCB / LiveCodeBench | Must not drop > 3 pts (Qwen is strong here — protect it) |
| GSM8K / MATH | Must not drop > 2 pts |
| RULER-32K (EU langs) | Must improve |
| Per-language (sv/fi/fr/de) | All must be ≥ base |

**Note:** Qwen3.5 is strong on coding/math. These are the hardest to not regress — monitor
LCB closely throughout training.

---

## Comparison with Gemma 4

| Aspect | Qwen 3.5 | Gemma 4 |
|---|---|---|
| Reasoning ceiling | ★★★★★ | ★★★★ |
| EU multilingual pretraining | ★★★★ (201 langs) | ★★★★★ |
| DPO support in TRL | ✅ Yes | ❌ No → SimPO |
| Fine-tuning ecosystem | Largest | Growing |
| Context length | 262K native | 128K–256K |
| MoE option | ✅ 35B-A3B | ✅ 26B-A4B |
