# Distillation Strategy — using LUMI's GPUs to teach the student

**Goal:** use abundant LUMI GPUs to distill a frontier teacher's capability (reasoning,
multilingual fluency, tool use) into our Qwen3.5-9B student — focused on **European
languages**, where teachers are strong but generic-tuned students are weak.

> **Punchline:** The bottleneck in distillation is *teacher generation throughput*, which is
> embarrassingly parallel — exactly what lots of GPUs are for. Stand up a high-throughput
> teacher-serving cluster, generate millions of **verified** traces, filter hard, then SFT
> (and optionally logit-distill) the student.

---

## 1. The teacher decision drives everything

| Teacher | Tokenizer vs student | Distill type | On LUMI? | Verdict |
|---|---|---|---|---|
| **Qwen3.5-32B / 122B-A10B** | **same as 9B student** | **logit/KL (token-level)** + sequence | download easy; vLLM-ROCm | **Best practical** — same tokenizer unlocks the strongest signal |
| GLM-5.2 (753B) | different | sequence-level only | no (~1.5TB, MoE) | best quality, impractical to self-host now |
| GLM-4.5-Air (106B/12B) / GLM-4.7-Flash (30B/3B) | different | sequence-level | downloadable (~60–200 GB) | good quality complement, moderate effort |
| Existing distilled data (DeepSeek-R1) | n/a (already text) | sequence-level SFT | **already staged** | **use now** — zero generation cost |

**Two-speed plan:**
- **Now (no generation needed):** SFT on distilled reasoning data *already on LUMI*
  (`am-deepseek-r1-think`, `Nemotron-v2`, `OpenR1-Math-220k`, Finnish DeepSeek math).
- **Next (use the GPUs):** self-host **Qwen3.5-122B-A10B** as teacher (same tokenizer →
  logit distillation) and/or a downloaded **GLM** for sequence-level reasoning, and generate
  EU-language traces at scale.

The same-tokenizer point is decisive: a larger Qwen3.5 teacher lets us train the 9B on the
teacher's *logits* (KL), which transfers far more than text alone. GLM/DeepSeek give text
only — still useful, but weaker per token.

---

## 2. What to distill (in priority order)

1. **Verified reasoning traces** — math, code, logic, science. Teacher emits `<think>` +
   answer; **keep only traces whose final answer passes a verifier** (exact-match math,
   unit-tested code). This is the single highest-value product (Nemotron/OpenThoughts recipe).
2. **Multilingual reasoning** — the same problems, prompted to **reason in the target EU
   language**. This is the gap no teacher is great at and our differentiator.
3. **EU instruction responses** — teacher answers to native EU prompts → high-quality SFT.
4. **Preference pairs** — teacher's best-of-N as `chosen`, a weaker/student sample as
   `rejected` → DPO data (on-policy if we sample from the student).
5. **Tool-use / agentic trajectories** — function-calling sequences (LightOn-aligned).
6. **Rejection-sampling / STaR** — sample N per prompt, keep the verified-correct ones, SFT
   on them (self-improvement loop once the student is decent).

---

## 3. The GPU-optimal generation pipeline (how to actually use the GPUs)

```
                       ┌─────────────────────────────────────────┐
   prompt bank   ──▶   │  TEACHER SERVING CLUSTER (vLLM-ROCm)     │  ──▶  raw traces
  (EU problems,        │  Qwen3.5-122B-A10B across N nodes        │
   instructions,       │  tensor + expert parallel, high batch    │
   "think in X")       └─────────────────────────────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────────────┐
                       │  VERIFY + FILTER                         │
                       │  math: exact-answer · code: run tests    │
                       │  lang-ID match · dedupe · quality score  │
                       └─────────────────────────────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────────────┐
                       │  DISTILL INTO STUDENT (Qwen3.5-9B)       │
                       │  seq-level SFT  (+ KL logit-distill if   │
                       │  same-tokenizer teacher)                 │
                       └─────────────────────────────────────────┘
```

**Why this scales with GPUs:** generation is independent per prompt. Throughput ≈
(nodes × per-node tokens/s). Doubling nodes doubles trace yield. A persistent vLLM
deployment + a streaming prompt feeder saturates whatever we give it. Split the cluster:
e.g. 6 nodes serving the teacher, 1–2 nodes running verification/dedupe, generation runs
for days unattended producing millions of verified traces.

**Prompt bank** (the fuel — build once, reuse):
- Math/logic: Numina/OpenMath seeds + problems mined from native EU school/exam/science text.
- Code: permissive competitive-programming + unit-test datasets.
- Instructions: native EU prompts from the long-context corpus, per language.
- Each prompt × {answer in EN, answer in target-lang, think in target-lang} variants.

---

## 4. Distillation methods (concrete)

**Sequence-level SFT (any teacher):** train the student on `(prompt → teacher trace+answer)`.
Tokenizer-agnostic. This is what the existing DeepSeek-distilled data already is — works
day one with our SFT pipeline.

**Logit / KL distillation (same-tokenizer teacher only):** minimise
`KL(teacher_logits ‖ student_logits)` on the teacher's sampled sequences. Much higher
information per token. Requires teacher logits — either stored during generation (top-k
logprobs from vLLM) or recomputed. Only valid Qwen-teacher → Qwen-student.

**On-policy distillation (advanced, slime/THUDM):** student generates, teacher scores/
corrects, student trains toward teacher-preferred continuations. Best for long reasoning;
needs both models live. Defer until the off-policy version is paying off.

**Verification is non-negotiable.** Unverified teacher traces include confident-wrong
reasoning that *teaches the student to be wrong*. Always gate math/code on a checker; for
open-ended, use a quality/consistency filter.

---

## 5. Sequencing

| Run | Distillation move | Cost | Data |
|---|---|---|---|
| **#2 (now)** | seq-level SFT on **existing** DeepSeek-distilled reasoning + multilingual DPO | ~0 generation | already on LUMI |
| **#3** | self-host **Qwen3.5-122B** teacher, generate verified EU reasoning at scale | GPU-heavy (the fun part) | we make it |
| **#4** | add **logit/KL** distillation (same-tokenizer) + GLM sequence-level complement | GPU-heavy | we make it |
| **#5** | on-policy distillation / rejection-sampling self-improvement | GPU-heavy | student + teacher |

Each run is gated on the EU holdouts: a distillation move ships only if it moves the
per-language/per-bucket numbers without regressing reasoning_math or English.

---

## 6. Immediate concrete actions
1. **Run #2 reasoning-SFT** on existing distilled data (`am-deepseek-r1-think`,
   `Nemotron-v2`, `OpenR1-Math-220k`, Finnish DeepSeek math) — no generation needed.
2. **Add multilingual DPO** (fin/swe/dan/nor/isl + translated) — fixes the English-only DPO gap.
3. **Prototype the teacher cluster**: serve Qwen3.5-122B-A10B on vLLM-ROCm (the v11 container),
   benchmark tokens/s/node → sizes the generation campaign.
4. **Build the prompt bank** from native EU sources + math/code seeds.

See `docs/EU_DATA_STRATEGY.md` for the broader data plan and `docs/RUNBOOK.md` for how runs execute.
