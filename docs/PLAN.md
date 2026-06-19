# Post-Training Plan — Best European Model (T4.6)

**Created:** 2026-06-18 · **Revised:** 2026-06-19  
**Owner:** Birger / AI Sweden

---

## Goal

Build the best openly available European language model by applying a targeted EU delta on
top of the strongest viable base model — without regressing its existing capabilities.

**Definition of success:** A model that outperforms all comparable open models on EU language
tasks AND maintains competitive scores on reasoning, coding, and agentic benchmarks. No
cherry-picking evals (per Jenia's point — show SWE-Bench, LCB, TerminalBench even if they
hurt).

---

## Step 0 — Base model selection (do this first, before any training)

**Do not guess the best base model. Measure it.**

Run a 2-3 hour eval job on LUMI comparing candidates on the EU-relevant benchmarks. Pick the
winner. This saves weeks of training on the wrong base.

**Candidates to evaluate:**

| Model | EU coverage | Reasoning | Agentic | Why consider |
|---|---|---|---|---|
| Qwen3-8B-Instruct | Weak (EN/ZH-heavy) | ★★★★★ | ★★★★ | Strongest ceiling |
| Gemma-3-9B-Instruct | Strong (Google multilingual) | ★★★★ | ★★★★ | Best EU+reasoning combo |
| Llama-3.1-8B-Instruct | Medium | ★★★ | ★★★ | Well-understood, safe |
| EuroLLM-9B-Instruct | ★★★★★ (EU-native) | ★★★ | ★★ | Project's own; already EU-optimized |
| Aya-Expanse-8B | ★★★★★ (multilingual native) | ★★ | ★★ | Great EU, low capability ceiling |

**Hypothesis:** Gemma-3-9B-Instruct or Qwen3-8B-Instruct will win — strong reasoning +
acceptable EU coverage that our SFT can close the gap on.

**Eval to run (per-model, ~30 min each on 8 GPUs):**
- ArenaHard-EU (EU multilingual preference)
- RULER-32K (long-context, in English first as sanity check)
- LCB / LiveCodeBench (coding — important to not regress)
- MMLU (knowledge baseline)
- GSM8K / MATH (reasoning baseline)

---

## What can we realistically improve

| Gap | Expected delta | Approach | Risk if done wrong |
|---|---|---|---|
| EU language quality | **Large** | SFT on quality-filtered EU data + DPO on EU preference pairs | Translationese artifacts if data is noisy |
| Long-ctx in EU languages | **Medium** | Long-ctx SFT from project corpus | Context erosion if seq length too short in SFT |
| Multilingual reasoning | **Medium** | LightOn reasoning SFT (multilingual traces) | English reasoning regression |
| EU-domain tasks | **Medium** | SFT on domain data from longctx corpus | Domain overfit |
| English / coding / agentic | **Near zero** — preserve only | Replay mix + eval gates | Catastrophic forgetting — main risk |

---

## What guarantees improvement — the non-negotiables

### 1. Data quality filtering (Propella) — before training, not after
Raw translated data is noisy. ELLIST's Propella tool filters SFT data by quality score.
Apply it to all translated Dolci data before including it in the mix. High-quality subset
only. This is the single highest-leverage pre-processing step.

### 2. Anti-translationese DPO (AI Sweden)
Translated EU text sounds translated. Use DPO to prefer original EU text over roundtrip-
translated text (AI Sweden's existing experiments). Makes the EU output sound native rather
than like a translation.

### 3. Native EU text in SFT, not just translated
The longctx corpus (`oellm-longctx-tokenized-streamed-all-v2`) is real EU documents —
PDFs from governments, universities, scientific publications. Include this as SFT source,
not just as eval. Real EU text > translated English text.

### 4. English replay mix in every SFT stage (~15%)
Prevents catastrophic forgetting. Every SFT batch contains ~15% English data from the
original training mix. Non-negotiable.

### 5. Per-language eval gates after every stage
Don't just measure aggregate multilingual. Measure Swedish, Finnish, French, German, etc.
individually. If Swedish degresses after SFT, catch it before DPO amplifies it.

### 6. On-policy DPO (not just off-policy Dolci pairs)
Generate rejected responses from the model being trained; use those as DPO negatives. Much
stronger signal than static off-policy pairs from a different model. Combine with Dolci-DPO.

---

## Pipeline — Nemotron 3 structure, Dolci data backbone

```
[Step 0: Base model eval → pick winner]
         │
         ▼
Stage 1 — Reasoning SFT
  Data: Dolci-Think-SFT (32B-distilled) + OpenThoughts-114k + LightOn multilingual reasoning
  Goal: establish reasoning in target EU languages before general SFT dilutes it
  Eval gate: GSM8K/MATH must not drop > 2pts vs base
         │
         ▼
Stage 2 — General SFT
  Data: Dolci-Instruct-SFT (70 langs, Propella-filtered) + longctx corpus (native EU docs)
        + ~15% English replay
  Goal: EU instruction following, long-context, domain knowledge
  Eval gate: ArenaHard-EU must improve; LCB must not drop > 3pts
         │
         ▼
Stage 3 — DPO
  Data: Dolci-Instruct-DPO-translated (11 langs) + on-policy negatives + anti-translationese pairs
  Goal: EU preference alignment, remove translationese
  Eval gate: ArenaHard-EU win-rate vs base; per-language checks
         │
         ▼
Stage 4 — RLVR / GRPO  (if ROCm stack stable)
  Data: RLVR-GSM-MATH-IF + EU-language IF constraints
  Goal: verifiable capability boost, especially multilingual reasoning
         │
         ▼
Stage 5 — Eval + release
```

---

## Scaling strategy

```
8B (Gemma-3 or Qwen3)  →  validate EU delta, tune pipeline
        ↓
Qwen3.5-9B-Instruct    →  stronger reasoning base, same pipeline
        ↓
Qwen3.5-32B-Instruct   →  production artifact
```

---

## Eval suite — broad, no cherry-picking

Following Nemotron / Laguna / MAI tech reports. Show everything even when it hurts.

**Multilingual / EU:**
- ArenaHard-EU (EU preference, per language)
- OneRuler / RULER (long-context, in EU languages)
- mMMLU (multilingual knowledge)
- Per-language: sv, fi, fr, de, es, pl, nl, pt eval sets

**Reasoning / Math:**
- GSM8K, MATH-500
- AIME 2024/2025 (hard math)

**Coding / Agentic:**
- HumanEval, MBPP
- **LCB / LiveCodeBench** (harder, more realistic)
- **SWE-Bench Verified** (agentic coding)
- **TerminalBench** (real environment use)

**General:**
- MMLU-Pro
- IFEval
- RULER-128K (long-context English)

**Eval after every stage.** Gate on: EU must improve, LCB/SWE must not drop > 5pts.

---

## T4.6 partner coordination

| Partner | Contribution |
|---|---|
| AI Sweden (Birger) | Pipeline, long-ctx SFT data, anti-translationese DPO, base model eval |
| ELLIST (Arjun/Hannan) | Propella filtering, DPO HP tuning, framework, ablations |
| Silo AI (Jouni/Elaine) | Translated datasets, long-ctx extension |
| LightOn (Yakine/Kai) | Function calling, multilingual reasoning SFT/RL |

**Coordination needed:**
- Arjun: share LUMI container/port fixes once he gets cluster access
- Hannan: request Qwen3-8B DPO hyperparameters from their BO sweep
- Kai: timeline for multilingual reasoning SFT data release

---

## Compute & timeline

**Budget:** ~300,000 GPU-h · **Deadline:** 2026-08-02

| Task | GPU-h | When |
|---|---|---|
| Base model eval (5 models) | ~50 | Week 1 — do first |
| 2B smoke (stack validation) | ~5 | Week 1 — in progress |
| SFT 8B | ~150 | Week 2 |
| DPO 8B | ~75 | Week 2–3 |
| RLVR 8B | ~500–1000 | Week 3–4 (if ROCm stable) |
| Scale to 32B | ~5× | Week 4–5 |
| **Total** | **~2,000–5,000** | Well within 300k budget |

---

## Immediate next actions

1. **Run base model eval** — 5 candidates on ArenaHard-EU + LCB + GSM8K (today)
2. **Finish 2B smoke run** — SFT job 19360004 in progress on LUMI
3. **Get Propella filtering running** on Dolci-Instruct-SFT before SFT stage
4. **Share LUMI fixes with Arjun** once he gets access (container, port, partition)
5. **Check Hannan's DPO HP results** for Qwen3-8B from their BO sweep
