# Language coverage gap analysis

What our **training data** covers vs the **38 OpenEuroLLM target languages** (the eval-holdout
set). Built from the data actually staged on LUMI (2026-06). Drives what to source/generate next.

**Target (38):** bg bs ca cs cy da de el en es et eu fi fr ga gl hr hu is it lb lt lv mk mt nl
no pl pt ro ru sk sl sq sr sv tr uk

## Coverage by data type

| Data type | Languages covered | Count | Verdict |
|---|---|---|---|
| **General SFT** (EuroBlocks + per-lang dirs) | bg cs da de el en es et fi fr ga hr hu is it lt lv mt nl no pl pt ro sk sl sv (+ sr/bs via hbs) | ~26 | good core, gaps at the edges |
| **Preference / DPO** (native) | **da en fi is no sv** | **6** | ⚠️ **Nordic-only** — biggest native gap |
| **Preference / DPO** (+ exam-MCQ) | 35 (adds most EU langs) | 35 | exam MCQs rescue coverage |
| **Reasoning / CoT** | **en + fi** | **2** | ⚠️ **biggest structural gap** |
| **Exam MCQ** (GRPO/RLVR) | bg ca cs da de el en es et eu fi fr hr hu is it lt lv mk mt nb nl pl pt ro ru sk sl sq sr sv tr uk (+ hy ka non-EU) | 35 | excellent |

## The gaps that matter

### 1. Reasoning is English + Finnish only — the #1 gap
We have zero chain-of-thought/reasoning data in 36 of the 38 target languages. The model can
*answer* in EU languages (SFT) but only *reasons* in English/Finnish. This is exactly what the
**distillation plan** (`docs/DISTILLATION.md`) addresses: generate verified EU-language
reasoning traces from a strong teacher (Qwen-122B / GLM), prompting it to *reason in the target
language*. **Highest-leverage data to create next.**

### 2. Native preference data is Nordic-only
Real (non-MCQ) DPO/preference pairs exist only for da/en/fi/is/no/sv. Every other EU language
gets preference signal **only** through the translated exam MCQs. Native, open-ended preference
data (helpfulness/fluency, not just right-answer) is missing for ~30 languages. The
**anti-translationese DPO** (AI Sweden) and exam-MCQ DPO partly cover this, but open-ended
native preference is a real gap.

### 3. Low-resource "language debt" — thin everywhere
These appear in few/no data types and need targeted sourcing (matches the EU_DATA_STRATEGY
priority list):

| Lang | SFT | Pref | Reason | Exam | Note |
|---|---|---|---|---|---|
| **cy** Welsh | ❌ | ❌ | ❌ | ❌ | absent across the board |
| **lb** Luxembourgish | ❌ | ❌ | ❌ | ❌ | absent (HPLT/native only) |
| **gl** Galician | ❌ | ❌ | ❌ | ❌ | absent |
| **bs** Bosnian | ~hbs | ❌ | ❌ | ❌ | only via Serbo-Croatian |
| **ga** Irish | ✅ gle | ❌ | ❌ | ❌ | SFT only |
| **eu** Basque | ❌ | ❌ | ❌ | ✅ | exam only |
| **sq** Albanian | ❌ | ❌ | ❌ | ✅ | exam only |
| **mk** Macedonian | ❌ | ❌ | ❌ | ✅ | exam only |

### 4. Surprising mid/high-resource SFT gaps
**ca (Catalan), ru (Russian), tr (Turkish), uk (Ukrainian)** have no general SFT data here
(only exam MCQs). These are large languages — easy to source (Aya/translated Dolci already
cover some) and worth filling.

## Recommended actions (priority)

1. **Generate multilingual reasoning** (distillation) for the 36 missing languages — biggest gap, highest leverage.
2. **Fill SFT gaps**: ca, ru, tr, uk (easy — translated Dolci / Aya), then eu, sq, mk, gl, bs.
3. **Native preference data** beyond Nordic — anti-translationese pairs + native helpfulness pairs per language.
4. **Language-debt sourcing**: cy, lb, gl, bs, ga from HPLT/native EU corpora (the longctx
   project already reaches these via HPLT — see `openeuro-longctx-datamix`).

## What's already strong
General SFT core (26 langs) + exam-MCQ verifiable RL (35 langs) is a solid base. The pipeline
(SFT → SimPO → GRPO) works across these. The gaps are **reasoning breadth** and **native
preference breadth** — both addressable via the distillation + native-data plans already
documented.
