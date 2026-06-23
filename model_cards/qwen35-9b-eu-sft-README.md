---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
pipeline_tag: text-generation
tags:
- openeurollm
- european-languages
- multilingual
- instruction-tuning
- sft
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
- da
- 'no'
- cs
- el
- hu
- ro
- bg
- hr
- sk
- sl
- et
- lv
- lt
- mt
- ga
- en
---

# Qwen3.5-9B-EU-SFT (v1)

European-language instruction-tuned variant of **Qwen3.5-9B**, fine-tuned on the OpenEuroLLM
**Dolci / EuroBlocks** multilingual instruction mix. Part of the OpenEuroLLM Task 4.6
post-training effort, trained on the **LUMI** supercomputer (AMD MI250X / ROCm).

> **What it is:** a first-pass SFT bootstrap that measurably improves European-language
> instruction following and reasoning over the base model, while preserving English ability.
> Not a final artifact — see *Limitations* and *Roadmap*.

## Results — EU evaluation holdouts

Evaluated on [`birgermoell/oellm-eu-eval-holdouts-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-eval-holdouts-v1)
(deterministic per-task scoring, dev split). **Overall accuracy 59.3% → 68.1% (+8.8).**

| Task bucket | Base Qwen3.5-9B | This model (SFT) | Δ |
|---|---:|---:|---:|
| instruction_following | 6.7 | **73.3** | **+66.6** |
| reasoning_math | 73.3 | **93.3** | **+20.0** |
| grounded_qa | 100 | 100 | — |
| civic_safety | 100 | 100 | — |
| summarization | 100 | 100 | — |
| no_answer | 100 | 93.3 | −6.7 |
| translationese_preference | 53.3 | 53.3 | — |
| locale_formatting | 0 | 0 | — |
| tool_calling | 0 | 0 | — |

*(Quick read, 10 prompts/bucket — per-bucket trends are robust; treat absolute values for the
small buckets as indicative. A fuller eval is in progress.)*

The headline: **instruction following 6.7 → 73.3** — the base model rarely produced the
required answer format in European languages; this model reliably does.

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-9B` (instruct, native 262K context) |
| Method | Full-parameter SFT (FSDP ZeRO-3, bf16) |
| Data | Dolci `tulu3-euroblocks-85-15`: EuroBlocks EU-multilingual instructions (85%) + Tülu-3 English replay (15%) |
| Examples | ~400k (subset of 1.08M) |
| Sequence length | 4096, packed |
| Optimizer | AdamW, LR 1e-5, cosine, warmup 0.03 |
| Hardware | 1× LUMI-G node (8× MI250X GCD), ROCm 6.4 |
| Framework | TRL `SFTTrainer` + 🤗 Transformers |

Code & full reproduction: <https://github.com/BirgerMoell/qwen35-posttrain> (`docs/RUNBOOK.md`).

## Intended use

European-language assistant / instruction following across 24+ EU languages, especially
mid/low-resource languages where the base model has the most headroom. Research artifact for
the OpenEuroLLM project.

## Limitations

- **First bootstrap run** — trained on synthetic + translated instruction data; translationese
  is not yet addressed (that's the next, preference-tuning, stage).
- **No tool-use or locale-formatting training** — those buckets score 0 (as does the base).
- Preference optimization (SimPO) and verifiable RL (GRPO on EU exams) are planned follow-ups.
- Inherits the base model's biases and knowledge cutoff.

## Roadmap (OpenEuroLLM T4.6)

SFT (this model) → preference optimization (SimPO, reference-free) → RLVR/GRPO on European exam
MCQs → broad eval. See the repo for the full plan and the distillation strategy.

## License

Apache 2.0 (inherits `Qwen/Qwen3.5-9B`). Training data: Dolci/EuroBlocks (see OpenEuroLLM
dataset cards for component licenses).

## Citation

```
@misc{oellm-qwen35-9b-eu-sft,
  title  = {Qwen3.5-9B-EU-SFT: European-language instruction tuning of Qwen3.5-9B},
  author = {Moëll, Birger and OpenEuroLLM Task 4.6},
  year   = {2026},
  url    = {https://github.com/BirgerMoell/qwen35-posttrain}
}
```
