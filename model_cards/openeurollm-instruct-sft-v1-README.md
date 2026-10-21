---
license: other
license_name: mixed-upstream
language: [en, de, fr, es, it, pt, nl, pl, cs, sk, sl, hr, bs, sr, mk, bg, ro, hu, fi, et, lv, lt, sv, da, "no", is, ga, mt, el, ca, eu, gl, sq, tr, uk, ka]
task_categories: [text-generation]
tags: [sft, instruct, multilingual, openeurollm, conversational]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
---

# OpenEuroLLM Instruct SFT v1

Direct-answer (non-reasoning) instruction data used for the Stage-1 Instruct SFT of
OpenEuroLLM 9B, cleaned of other assistants' identity claims and extended with a
36-language identity set. **Internal research dataset** — upstream source terms vary and
have not been cleared for redistribution.

- **3,150,701 train / 3,289 validation** conversations in `messages` format
  (`system` / `user` / `assistant`, text only, no reasoning traces, no tool calls).
- Columns: `id`, `source`, `language` (`mixed` for Excellent-SFT rows), `messages`,
  `modification` (empty, `rewrote_openai_self_attribution`, or `new_identity_v2`).

## Sources

| Source | Rows (train) | Notes |
| --- | --- | --- |
| [OpenEuroLLM-Excellent-SFT](https://huggingface.co/datasets/openeurollm/OpenEuroLLM-Excellent-SFT) @ `c4ee558` | ~2.63M | direct-answer rows only; mostly English |
| [Dolci-Instruct-SFT-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated) @ `63202c3` | ~478K | 12 languages, each prompt family in exactly one language |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | ~22K | non-English OpenEuroLLM target languages, ≤8K/language, answers ≥50 chars |
| [OpenEuroLLM Model Identity v1](https://huggingface.co/datasets/birgermoell/openeurollm-model-identity) | 782 | stored once |
| Identity v2 (new) | 510 | see below |

Splits follow the Stage-1 build: Excellent rows keep its prompt-hash split; translated Dolci
and Aya use a source-family/prompt hash; identity v1 keeps its own validation split.

## Cleaning of foreign identity claims

Source data contained assistant turns such as "As an AI developed by OpenAI, I don't have
access to…" (≈750 rows, mostly `orca/follow_up` and English Dolci), which teach a model to
call itself an OpenAI model. Fixes:

- **622 rows rewritten**: English self-attribution removed ("As an AI developed by OpenAI, …" → "As an AI, …").
- **129 rows dropped**: explicit claims ("I am ChatGPT …") that could not be neutrally rewritten.
- **600 rows dropped**: the assistant names another assistant/lab (OpenAI, ChatGPT, GPT-n, Anthropic,
  Qwen, DeepSeek, Mistral AI, Meta AI, SmolLM) although no user/system turn mentions it. This catches
  translated self-claims; it is conservative and also removes some legitimate mentions (~0.02 % of rows).

## Identity v2

Short, direct identity Q&A in all 36 OpenEuroLLM target languages, consistent with the
identity-v1 fact sheet (`identity_v2_core.json` in this repo): "Who are you?", "Who made you?",
"What language model are you?", "Are you ChatGPT?", "Were you made by OpenAI?", "What is your name?".
Each appears plain, under a generic system prompt, and (for English and the 12 Dolci languages)
appended after a real task conversation. Answers: an OpenEuroLLM language model developed through
the European OpenEuroLLM consortium; not ChatGPT / not made by OpenAI; no personal name.
Model-authored translations — **native-speaker review pending**.
