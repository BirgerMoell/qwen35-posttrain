---
license: other
license_name: mixed-upstream-see-card
language: [en, de, fr, es, it, nl, pl, cs, el, ro, sv, fi, uk, pt, da, "no", is, bg, hr, hu, sk, sl, et, lv, lt, ga, mt, ca, eu, gl, bs, sr, mk, sq, tr, ka]
task_categories: [text-generation]
tags: [sft, reasoning, instruct, multilingual, openeurollm, chain-of-thought, conversational]
size_categories: [1M<n<10M]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
---

# OpenEuroLLM Reasoning-Instruct v1

A multilingual supervised fine-tuning mixture for training a **hybrid reasoning + instruct** model: an opt-in
thinking mode and an unchanged direct-answer default mode. It is the Stage-2 (reasoning SFT) mixture of the
OpenEuroLLM 9B two-stage post-training run, released in messages format.

- **1,164,312 train / 2,461 validation** conversations, ~**2.34B tokens** (OpenEuroLLM tokenizer).
- **~74 % reasoning** (1.74B tokens) — the final assistant turn has a nonempty `reasoning_content` trace.
- **~26 % direct answers** (0.60B tokens) — no trace; keeps the default (non-thinking) mode stable.
- Reasoning traces in the **target language** for 12 European languages (translated Dolci-Think) and 30+ languages
  in a small native-trace pilot; elsewhere traces are English with localized answers.

## Format

| Column | Description |
|---|---|
| `id` | stable row id, prefixed by source |
| `mode` | `reasoning` or `direct` |
| `messages` | list of `{role, content, reasoning_content}`; roles `system`/`user`/`assistant`; text only |
| `source` | source slice (e.g. `dolci_tr:sv`, `dolci32b:WildGuardMix`, `excellent:nemotron/multilingual_de`) |
| `source_dataset` | upstream repo @ pinned revision |
| `license` | upstream license/terms for this row |
| `language` | language code (`mixed` where a source does not label rows) |

Only the **final** assistant turn carries `reasoning_content`; earlier turns carry none (as at inference, where
clients do not send reasoning back). Content never contains `<think>` tags — render the trace with your template.
Our template puts `<think>\n` on the prompt side (generation prompt) and trains on
`trace\n</think>\n\nanswer`, so a model never learns to open a think block on its own and never sees empty ones.

## Composition (train)

| Block | Rows | Tokens | Upstream | License |
|---|---:|---:|---|---|
| Translated reasoning, 12 languages (cs de el es fi fr it nl pl ro sv uk; ~45M tokens each; one language per prompt family; `filter_pass` only) | 69,263 | 541M | [openeurollm/Dolci-Think-SFT-translated](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-translated) | Apache-2.0 (upstream ODC-BY) |
| Precise instruction following (Persona Precise IF, Dolci Think Precise IF) | 67,081 | 137M | [openeurollm/Dolci-Think-SFT-32B-decontaminated](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-32B-decontaminated) | ODC-BY-1.0 |
| Safety reasoning (WildGuardMix, WildJailbreak, CoCoNot) | 34,349 | 27M | same | ODC-BY-1.0 |
| Chat (WildChat, OpenAssistant), Aya, TableGPT | 45,278 | 133M | same | ODC-BY-1.0 |
| Math / code / STEM (capped: OpenThoughts3, Python algorithms, Nemotron code, SYNTHETIC-2 verified) | 15,822 | 150M | same | ODC-BY-1.0 |
| Nemotron multilingual (de, es, fr, it prompts; English traces, localized answers) | 42,752 | 300M | [OpenEuroLLM-Excellent-SFT](https://huggingface.co/datasets/openeurollm/OpenEuroLLM-Excellent-SFT) (Nemotron v2) | CC-BY-4.0 (source exceptions apply) |
| Nemotron chat, OpenThoughts3, SmolTalk2 multilingual8, LongAlign, multi-turn IF, s1K, … | 85,249 | 418M | OpenEuroLLM-Excellent-SFT (SmolTalk2, Nemotron) | Apache-2.0 / CC-BY-4.0 |
| Medical QA (USMLE-style, only verified-correct, completed traces) | 9,514 | 15M | [birgermoell/medqa-reasoning-traces](https://huggingface.co/datasets/birgermoell/medqa-reasoning-traces) | MIT (MedQA, Jin et al. 2021); Kimi-K3 output terms |
| Native-language traces pilot (30+ target languages, accepted rows) | 3,102 | 16M | [openeurollm/reasoning-traces-multilingual](https://huggingface.co/datasets/openeurollm/reasoning-traces-multilingual) | CC-BY-4.0 |
| Swedish reasoning (Kimi-K3) | 382 | 0.3M | [jonasaise/kimi-kr-synth-sv](https://huggingface.co/datasets/jonasaise/kimi-kr-synth-sv) | Apache-2.0; Kimi-K3 output terms |
| **Direct answers:** Excellent-SFT direct rows, translated Dolci-Instruct (12 langs), Aya, OpenEuroLLM identity (v1 + 36-language v2), Swedish Kimi answers | 791,520 | 599M | [birgermoell/openeurollm-instruct-sft-v1](https://huggingface.co/datasets/birgermoell/openeurollm-instruct-sft-v1) sources, jonasaise/kimi-kr-synth-sv | mixed — per row |

Exact per-source counts, budgets, sampling probabilities and rejection counts are in `manifest.json`.

## Construction and filters

- Deterministic selection: per-source rendered-token budgets; rows chosen by a stable hash of their id.
- Every row was rendered with the OpenEuroLLM chat template and tokenizer and audited: reasoning rows have exactly one
  closed trace and a nonempty answer; direct rows contain no think markers; nothing over 32,768 tokens (rejected, never truncated).
- **Identity hygiene**: rows whose assistant text or trace names another assistant or lab (OpenAI, ChatGPT, GPT-n,
  Anthropic, Qwen, DeepSeek, Mistral AI, Meta AI, SmolLM, OLMo/Ai2) while the prompt does not mention it were dropped
  (~550 rows); Dolci's "Olmo Identity Hardcoded" rows were excluded; direct rows inherit the cleaning of
  openeurollm-instruct-sft-v1 (rewritten "As an AI developed by OpenAI" phrasing).
- Prompts referencing missing images, malformed traces (unclosed or empty `<think>`), and Japanese rows were removed.
- Dolci-Think-32B was sampled from 66 of its 156 shards (shards are source-shuffled, so this is proportional).
- Decontamination is inherited from the upstream "decontaminated" releases (MATH-500, AIME, GPQA, LiveCodeBench,
  HumanEval, MBPP, IFEval, AlpacaEval, Arena-Hard); other sources were not separately decontaminated.

## Limitations

Much of the data is machine-translated or model-generated (Gemma, Qwen, DeepSeek, Kimi and others) and has not been
reviewed by native speakers; translation errors and teacher biases are present. Reasoning-language coverage is uneven:
target-language traces exist for 12 languages, the 30+-language pilot is small, and most other traces are English.
Correctness is verified only where the source did so (MedQA, SYNTHETIC-2, OpenR1-style subsets).

## Licenses and attribution

This is a mixed-source dataset: **each row remains under the terms of its upstream source**, given in its `license`
column. Model-generated content is additionally subject to the generating model's output terms where applicable.
Please credit NVIDIA (Nemotron), Ai2 (Dolci/OLMo), Hugging Face (SmolTalk2), Cohere Labs (Aya), the MedQA authors
(Jin et al., 2021), OpenEuroLLM, and jonasaise when using this data.
