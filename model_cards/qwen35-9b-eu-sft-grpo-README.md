---
license: apache-2.0
base_model: birgermoell/Qwen3.5-9B-EU-SFT
library_name: transformers
pipeline_tag: text-generation
tags:
- openeurollm
- european-languages
- multilingual
- grpo
- rlvr
- reinforcement-learning
- qwen3.5
language:
- sv
- fi
- de
- fr
- es
- it
- pt
- nl
- pl
- bg
- hr
- hu
- ro
- sk
- sl
- lt
- lv
- et
- en
---

# Qwen3.5-9B-EU-SFT-GRPO (v2)

The full **SFT → SimPO → GRPO** European post-training of Qwen3.5-9B — the "v2" recipe (full EU
instruction set + reference-free preference optimisation + verifiable-reward RL). The final
**GRPO / RLVR** stage uses **European exam multiple-choice questions** across 13 EU languages. Part
of OpenEuroLLM Task 4.6, trained on **LUMI** (AMD MI250X / ROCm).

This repo ships the **v2** weights — a clear upgrade over the earlier under-baked 9B.

> **Why RLVR:** the reward is a deterministic verifier (did the model pick the correct option
> letter?) — no preference data, no reward model, and **no translationese** in the loop. It
> optimizes European-language *correctness* directly.

## Results

**Training reward** (`mcq_letter_exact`, in-sample) climbed **0.51 → ~0.78** over 500 steps —
the model learned to answer EU exam MCQs correctly ~78% of the time.

**EU eval holdouts** ([`oellm-eu-eval-holdouts-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-eval-holdouts-v1))
— the v2 model scored on the **large `test_public` split (5,472 rows)**, deterministic per-task
scoring:

**Overall: 64.9%**, and — notably — **even across all 38 European languages (56.9 – 71.5%)**, with
no collapsed language. By task bucket:

| Bucket | Score |
|---|---:|
| civic_safety / grounded_qa / reasoning_math / no_answer | **100** |
| summarization | 79.8 |
| instruction_following | 53.9 |
| translationese_preference | 50.0 |
| locale_formatting · tool_calling | 0 · 0 |

The two zero buckets are known artifacts, not capability gaps: **tool_calling** — the 9B has no
tool-use training (that capability lives in the 4B tool model); **locale_formatting** — a
strict exact-match bucket that is chronically ~0. Excluding those two, the model averages **~83%**
on the remaining seven buckets.

*(Full test_public split — this is the robust signal, not a small-bucket sample.)*

## Swedish capability (open-ended rubric)

A vLLM-free Swedish eval (Qwen3.5 isn't yet supported by EuroEval's fast backends): 24 prompts
across 9 categories (sentiment, Swedish knowledge, reasoning, summarization, linguistic
correctness, instruction-following, common sense, creative), scored 1–5 on
**språkkvalitet / korrekthet / instruktion / hjälpsamhet**, plus automatic langdetect Swedish-purity.

| Dimension | 2B-EU | 4B-EU | **9B-EU (this)** |
|---|---|---|---|
| Swedish-purity (langdetect) | 1.00 | 1.00 | **1.00** |
| Språkkvalitet (fluency/grammar) | 4.3 | 4.6 | **4.7** |
| Korrekthet (factual/logical) | 2.7 | 4.4 | **4.6** |
| Instruktion (follows constraints) | 3.4 | 4.0 | **4.5** |
| Hjälpsamhet | 2.9 | 4.3 | **4.5** |

The 9B is the **strongest Swedish model in the lineup** — best on every dimension. It gives the
three largest lakes *with areas*, names **Rayleigh-scattering** for the blue sky, distinguishes
universal vs particular premises in a logic question, and respects instruction constraints
(4-item lists, an < 80-word formal email), all in fluent, fully-Swedish prose (purity 1.00).
Tooling: `scripts/eval_swedish_rubric.py` + `data/swedish_rubric_prompts.jsonl`.

## Training data

This model is the cumulative result of **SFT → GRPO**, so it carries the data from both stages
(all openly-documented OpenEuroLLM Task 4.6 data; no proprietary data):

**1. SFT stage** (inherited from [`Qwen3.5-9B-EU-SFT`](https://huggingface.co/birgermoell/Qwen3.5-9B-EU-SFT)) —
~400k examples (subset of ~1.08M), packed, bf16:
- **General EU instructions** — Dolci `tulu3-euroblocks-85-15`: **EuroBlocks** EU-multilingual
  instruction data (85%) + **Tülu-3** (`allenai/tulu-3`) English replay (15%) — adds EU-language
  instruction-following while preserving English.

**2. GRPO / RLVR stage** (this model):
- [**`birgermoell/oellm-eu-exam-mcq-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1)
  GRPO/RLVR split — European exam multiple-choice questions (EXAMS-QA, 13 EU languages,
  CC-BY-SA-4.0). The broader dataset spans **~35 languages, 28 sources** (national, medical/
  licensing, and academic exams), mixed licenses (filterable per row). Verifiable letter-match
  reward (`mcq_letter_exact`, β=0, no reward model) — optimizes EU-language answer *correctness*
  directly, with no translationese in the loop.

**Evaluation** — held out from training:
[**`birgermoell/oellm-eu-eval-holdouts-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-eval-holdouts-v1)
— EU languages × task buckets, deterministic per-task scoring.

## Training

| | |
|---|---|
| Initialized from | `birgermoell/Qwen3.5-9B-EU-SFT` |
| Method | GRPO / RLVR, **β=0** (no KL/reference model — Dr.GRPO-style, also avoids the 2-model memory cost) |
| Reward | `mcq_letter_exact` — 1.0 if the chosen option letter matches the gold answer |
| Data | [`oellm-eu-exam-mcq-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1) GRPO split (EXAMS-QA, 13 EU languages, CC-BY-SA-4.0) |
| Rollouts | 8 generations/prompt, HF generation backend |
| Steps | 500 |
| Hardware | 1× LUMI-G node (8× MI250X GCD), ROCm 6.4, ~4.5 h |
| Framework | TRL `GRPOTrainer` + 🤗 Transformers |

Code & reproduction: <https://github.com/BirgerMoell/qwen35-posttrain> (`docs/RUNBOOK.md`,
`scripts/grpo_train.py`, `configs/grpo_qwen35_9b_exam.yaml`).

## Intended use & limitations

European-language assistant with improved exam-style/MCQ and reasoning accuracy. Same notes as
the SFT model: it is a **reasoning model** (emits `<think>…</think>`; use `enable_thinking=False`
for direct answers). No tool-use/locale training. RLVR optimizes the verifiable signal (answer
correctness), not open-ended style.

## License

Apache 2.0 (base: Qwen3.5-9B). Exam data: CC-BY-SA-4.0 (EXAMS-QA).

## Citation
```
@misc{oellm-qwen35-9b-eu-sft-grpo,
  title  = {Qwen3.5-9B-EU-SFT-GRPO: RLVR on European exam MCQs},
  author = {Moëll, Birger and OpenEuroLLM Task 4.6},
  year   = {2026},
  url    = {https://github.com/BirgerMoell/qwen35-posttrain}
}
```
