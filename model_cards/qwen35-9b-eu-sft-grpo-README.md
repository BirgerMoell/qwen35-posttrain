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

# Qwen3.5-9B-EU-SFT-GRPO (v1)

[`Qwen3.5-9B-EU-SFT`](https://huggingface.co/birgermoell/Qwen3.5-9B-EU-SFT) further trained with
**GRPO / RLVR** (verifiable-reward reinforcement learning) on **European exam multiple-choice
questions** across 13 EU languages. Part of OpenEuroLLM Task 4.6, trained on **LUMI** (AMD
MI250X / ROCm).

> **Why RLVR:** the reward is a deterministic verifier (did the model pick the correct option
> letter?) — no preference data, no reward model, and **no translationese** in the loop. It
> optimizes European-language *correctness* directly.

## Results

**Training reward** (`mcq_letter_exact`, in-sample) climbed **0.51 → ~0.78** over 500 steps —
the model learned to answer EU exam MCQs correctly ~78% of the time.

**EU eval holdouts** ([`oellm-eu-eval-holdouts-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-eval-holdouts-v1),
out-of-sample, dev), full base → SFT → GRPO trajectory:

| | Overall | reasoning_math |
|---|---:|---:|
| Base Qwen3.5-9B | 48.9 | 0 |
| + EU SFT | 54.4 | 60 |
| **+ GRPO (this model)** | **57.8** | **70** |

GRPO improves over the SFT model both in-sample (reward) and out-of-sample (+3.4 overall,
+10 on reasoning_math). It also makes answers **more direct** (trained on short completions),
reducing the over-generation seen in the SFT model.

*(Quick read, 10 prompts/bucket — trends robust, small-bucket absolutes indicative. Absolute
numbers are conservative: this is a reasoning model whose `<think>` block consumes part of a
short generation budget — see the SFT card.)*

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
