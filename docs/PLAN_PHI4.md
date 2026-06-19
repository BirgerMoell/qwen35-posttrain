# Phi-4 Fine-Tuning Plan — EU Model

**Model family:** Microsoft Phi-4 (2025–2026)  
**License:** MIT — fully permissive, no restrictions on redistribution  
**Role in multi-family strategy:** Efficiency candidate — best reasoning-per-parameter,
smallest footprint for cheap ablations and many-language sweeps

---

## Why Phi-4

- **Best reasoning-per-parameter** at the 4B–14B scale — outperforms Llama 3.3 70B on
  some reasoning benchmarks despite being 5× smaller
- **MIT license** — most permissive of the three families
- **Small sizes** (3.8B / 14B) — extremely cheap to iterate, a full EU SFT run on 4B
  costs <50 GPU-h. Ideal for rapid ablations of data mix, LR, seq length
- **Reasoning variants:** `Phi-4-reasoning` and `Phi-4-mini-reasoning` — thinking-mode
  models that can be used as drop-in replacements without a reasoning SFT stage
- **Dense architecture** — no routing complexity, straightforward fine-tuning
- **100K vocab** with multilingual padding — good foundation for EU language tokens

---

## Sizes and iteration strategy

| Model | Params | Memory (full FT, 1 node) | Use |
|---|---|---|---|
| `microsoft/Phi-4-mini` | ~3.8B | ~46 GB | Ultra-fast ablations (minutes) |
| `microsoft/Phi-4-mini-reasoning` | ~3.8B | ~46 GB | Fast ablations with thinking |
| `microsoft/Phi-4` | 14B | ~168 GB | Main experiment |
| `microsoft/Phi-4-reasoning` | 14B | ~168 GB | Main experiment with thinking |

All sizes fit comfortably on a single LUMI node (512 GB HBM2e) with FSDP ZeRO-3.

**Primary use of Phi-4 in this project:** Run data mix and hyperparameter ablations on
Phi-4-mini (3.8B) cheaply, then transfer winning configs to Gemma 4 and Qwen 3.5 at 9B.
This saves significant GPU-hours vs tuning directly on 9B models.

---

## Key technical notes

### Strongest for ablations, not necessarily for production
Phi-4's EU language coverage is weaker than Gemma 4 or Qwen 3.5 — it is English-optimized
with multilingual padding rather than native multilingual pretraining. This means:
- EU delta will need more data to close the gap
- Results on Phi-4 may not perfectly predict Gemma 4 / Qwen 3.5 results
- Use Phi-4 for **relative comparisons** (which data mix is better?) not absolute results

### DPO works on Phi-4
TRL DPOTrainer is compatible with Phi-4. Standard DPO pipeline applies.

### Chat template
Phi-4 uses the standard ChatML format with `<|im_start|>` / `<|im_end|>` delimiters —
same as Qwen, easy to reuse data pipelines between the two families.

### Reasoning variants
`Phi-4-reasoning` has thinking-mode traces enabled. Start from the reasoning variant for
EU multilingual reasoning experiments — avoids a separate reasoning SFT stage.

---

## Pipeline (same structure as Gemma 4 / Qwen 3.5)

```
Phi-4-{variant}-Instruct
        │
        ▼
Stage 1 — EU Multilingual SFT  (full fine-tuning, FSDP ZeRO-3)
  Data: same mix as Gemma 4 / Qwen 3.5 — enables direct comparison
  LR: 2e-5, cosine, warmup 0.03, bfloat16, seq_len 4096
        │
        ▼
Stage 2 — EU DPO  (TRL DPOTrainer)
  Data: Dolci-Instruct-DPO-translated + anti-translationese pairs
        │
        ▼
Eval + compare with Gemma 4 / Qwen 3.5
```

---

## Recommended hyperparameters (full fine-tuning, FSDP ZeRO-3)

| Param | SFT | DPO |
|---|---|---|
| Learning rate | 1e-5 | 5e-7 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |
| Global batch size | 64 | 32 |
| Per-device batch size | 4–8 | 2–4 |
| Precision | bfloat16 | bfloat16 |
| Gradient checkpointing | true | true |
| Max seq length | 4096 | 2048 |
| Weight decay | 0.001 | 0.01 |
| DPO beta | — | 0.1 |

Phi-4 uses a lower LR than Qwen/Gemma (1e-5 vs 2e-5) — it is more sensitive to
over-training due to its data-efficient pretraining approach.

---

## Primary role: data mix ablations

Run these cheaply on Phi-4-mini (3.8B) before committing to 9B runs on Gemma 4 / Qwen 3.5:

| Ablation | Question |
|---|---|
| Dolci-only vs Dolci + longctx | Does native EU text improve ArenaHard-EU? |
| With vs without anti-translationese DPO | How much does translationese hurt? |
| Propella-filtered vs unfiltered | What's the quality filtering gain? |
| 15% vs 25% English replay | Optimal anti-forgetting weight? |
| seq_len 4096 vs 8192 vs 16384 | Long-ctx SFT impact on RULER? |

Each ablation: ~50–100 GPU-h on Phi-4-mini (3.8B). Same ablation on 9B: ~300–500 GPU-h.
Running 5 ablations on 3.8B first saves ~2,000 GPU-h.

---

## Eval gates

| Metric | Gate |
|---|---|
| ArenaHard-EU | Must improve vs Phi-4 base |
| MMLU / GSM8K | Must not drop > 3 pts (weaker base than Qwen/Gemma) |
| Per-language (sv/fi/fr/de) | All must be ≥ base |
| LCB | Track but lower bar — Phi-4 weaker on coding than Qwen |

---

## Comparison across three families

| Aspect | Gemma 4 | Qwen 3.5 | Phi-4 |
|---|---|---|---|
| EU multilingual pretraining | ★★★★★ | ★★★★ | ★★★ |
| Reasoning ceiling | ★★★★ | ★★★★★ | ★★★★ |
| Iteration speed (4B) | Fast | Fast | **Fastest** |
| DPO support in TRL | ❌ SimPO | ✅ | ✅ |
| License | Apache 2.0 | Apache 2.0 | **MIT** |
| Fine-tuning ecosystem | Growing | Largest | Medium |
| Best use | EU production | Reasoning-heavy | Ablations |
