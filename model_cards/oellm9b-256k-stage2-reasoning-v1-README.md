---
base_model: birgermoell/oellm-9b-256k-stage1-instruct-v1
library_name: transformers
pipeline_tag: text-generation
tags: [openeurollm, sft, reasoning, multilingual]
---

# OELLM 9B 256K — Stage-2 Reasoning v1 (internal)

Reasoning SFT on top of [`birgermoell/oellm-9b-256k-stage1-instruct-v1`](https://huggingface.co/birgermoell/oellm-9b-256k-stage1-instruct-v1),
trained on [`birgermoell/openeurollm-reasoning-instruct-v1`](https://huggingface.co/datasets/birgermoell/openeurollm-reasoning-instruct-v1)
(2.34B tokens: ~74 % reasoning, ~26 % direct-answer replay; one epoch, LR 5e-6 cosine, 32K packing, 32 × A100, ~350 GPU-h).
**Internal research checkpoint.**

## Thinking switch

The bundled template has a hard `enable_thinking` switch. Default (`enable_thinking=False`): direct answers, no think
tokens. `enable_thinking=True`: the prompt ends with `<think>\n`, the model writes its reasoning, closes `</think>`, then
answers. The model never opens a think block on its own.

## Generation check (24 multilingual prompts, greedy)

- Default mode: 0/24 replies with think tags; identity correct (OpenEuroLLM) in English, German and Swedish.
- Thinking mode: 23/24 replies close the trace and answer in the prompt language; 1 (Ukrainian) loops without closing
  within 4,096 tokens. Thinking-mode answers are verbose even when brevity is requested.

## Known limitations

Strict "answer only" instructions are still not followed exactly; occasional reasoning loops in thinking mode;
long-context retention not yet re-measured; DPO and RL stages not yet applied.
