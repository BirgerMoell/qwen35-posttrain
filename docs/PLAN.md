# Post-Training Plan — European Model Improvement (T4.6)

**Created:** 2026-06-18 · **Revised:** 2026-06-18  
**Owner:** Birger / AI Sweden (T4.6 — long context, task management, delta-learning)

---

## Goal

Take a strong existing instruct model and make it **the best European model** by improving it
on European languages, long-context EU documents, and EU-relevant tasks — without regressing
its existing English/reasoning capabilities.

This is **not** a full post-training from scratch. We apply a targeted EU delta on top of a
model that already knows how to reason, follow instructions, and use tools.

---

## What we can realistically improve

| Gap | Expected delta | Approach |
|---|---|---|
| EU language quality (sv, fi, fr, de, …) | **Large** — SOTA models are English-heavy | SFT on translated Dolci + DPO on EU preference pairs |
| Long context *in EU languages* | **Medium** — English long-ctx works, EU langs degrade | Long-ctx SFT from project corpus (multilingual, natively long PDFs) |
| EU-domain tasks (legal, government, science) | **Medium** — base models see little EU official text | SFT on domain data mined from the longctx corpus |
| Multilingual reasoning | **Medium** — reasoning traces are mostly English/Chinese | LightOn multilingual reasoning SFT (Kai Hakala) |
| General English / math / code | **Near zero** — we would degrade, not improve | **Do not touch** — preserve via delta-learning / anti-forgetting |

---

## Model scaling strategy

Start small, prove the EU delta, then scale within the same family.

```
Qwen3-8B-Instruct     ← start here: fast iteration, ELLIST has tuned DPO HPs already
        ↓
Qwen3.5-9B-Instruct   ← stronger base, same pipeline, minimal re-tuning
        ↓
Qwen3.5-32B-Instruct  ← production artifact if results hold
```

**Why Qwen3-8B-Instruct first:**
- ELLIST (Hannan) has already run BO hyperparameter sweeps for DPO on this model with
  Dolci data — reuse those rather than tuning blind.
- Strong multilingual capability out of the box; EU delta is measurable.
- Fast to iterate on LUMI (fits on 4–8 GPUs for SFT/DPO).

**Alternative base if Qwen3 results disappoint:** Nemotron-3-8B-Instruct (strong reasoning
baseline, open recipe we understand well).

---

## Pipeline

Follows **Nemotron 3 structure** (stronger published results than OLMo3, equally open).
Data backbone is the **Dolci suite** (already translated, multilingual, ELLIST-validated).

```
[instruct base]
      │
      ▼
Stage 1 — EU Multilingual SFT
  • Dolci-Instruct-SFT (70 langs, openeurollm/)
  • Dolci-Think-SFT (reasoning traces, multilingual)
  • LightOn multilingual reasoning SFT
  • Long-ctx SFT: project corpus (birgermoell/oellm-longctx-tokenized-streamed-all-v2)
      │
      ▼
Stage 2 — EU DPO
  • Dolci-Instruct-DPO-translated (11 langs)
  • Dolci-Think-DPO for reasoning preference
  • AI Sweden delta-learning: DPO to prefer original EU text over roundtrip translations
      │
      ▼
Stage 3 — Light RLVR  (secondary — only if time + ROCm stack stable)
  • Verifiable math/IF rewards on EU-language prompts
  • RLVR-GSM-MATH-IF-Mixed-Constraints + Dolci-RL-Zero-Mix
      │
      ▼
Stage 4 — Eval + release
  • RULER (long-ctx must not regress), MMLU, GSM8K, IFEval
  • OneRuler / ArenaHard-EU (multilingual)
  • Release on HF with model card
```

---

## Key design decisions

**Start from instruct, not base.** Avoids redoing reasoning SFT, tool-use SFT, safety —
work that took the base model teams months. We apply the EU delta on top.

**Anti-catastrophic forgetting is critical.** We're improving a good model; degrading English
is a failure mode. Mitigations:
- Include a small English replay mix in SFT (~10–15%)
- AI Sweden's translationese DPO experiments (prefer original EU text over roundtrip)
- Eval English benchmarks after every stage; gate on regression

**Coordinate with ELLIST on DPO HPs.** They've already swept LR × beta on Qwen3-8B.
Reusing their optimal HPs saves ~1 week of compute.

**Long-context data is a differentiator.** No other European post-training effort has the
project's own multilingual long-context PDF corpus. Use it.

---

## T4.6 partner responsibilities

| Partner | Contribution to this pipeline |
|---|---|
| AI Sweden (Birger) | Pipeline orchestration, long-ctx SFT data, delta-learning/translationese DPO |
| ELLIST (Arjun/Hannan) | Post-training framework, DPO HP tuning, ablations, eval |
| Silo AI (Jouni/Elaine) | Translated datasets (Dolci-Instruct-SFT/DPO), long-ctx extension |
| LightOn (Yakine/Kai) | Function calling SFT, multilingual reasoning SFT/RL |

---

## Compute & timeline

**Budget:** ~300,000 GPU-hours · **Deadline:** 2026-08-02

| Stage | Estimated GPU-h (8B) | Notes |
|---|---|---|
| SFT 8B, ~1B tokens | ~100 GPU-h | Cheap |
| DPO 8B | ~50 GPU-h | Cheap |
| RLVR 8B | ~500–1000 GPU-h | Rollouts are expensive |
| Scale to 32B | ~5× above | If 8B results are good |
| **Total realistic** | **~2,000–5,000 GPU-h** | Well within 300k budget |

Compute is **not** the constraint — a working ROCm RLVR stack and good EU data are.

---

## Immediate next actions

1. **Get ELLIST's DPO HPs for Qwen3-8B** — ping Hannan (github issue or Slack)
2. **Stage Qwen3-8B-Instruct on LUMI** (login node, no internet on compute)
3. **Run SFT smoke on 2B** — validates the LUMI/TRL/ROCm stack (in progress)
4. **Run SFT + DPO on 8B** with Dolci data + eval on ArenaHard-EU / RULER
5. **Scale to 9B/32B** once delta is proven on 8B
