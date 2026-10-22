---
base_model: birgermoell/oellm-9b-256k-stage2-reasoning-v1
library_name: transformers
pipeline_tag: text-generation
tags: [openeurollm, sft, tool-use, function-calling, reasoning, multilingual]
---

# OELLM 9B 256K — Stage-3 Tool-Use v1 (internal)

Tool-use SFT on top of [`birgermoell/oellm-9b-256k-stage2-reasoning-v1`](https://huggingface.co/birgermoell/oellm-9b-256k-stage2-reasoning-v1).
**Internal research checkpoint.**

Training data (411M tokens, one epoch, LR 3e-6 cosine, 32K packing, 32 × A100, ~64 GPU-h):

| Part | Rows | Tokens |
|---|---:|---:|
| Tool use — [`birgermoell/openeurollm-tool-use-v1`](https://huggingface.co/datasets/birgermoell/openeurollm-tool-use-v1) (subsampled per source) | 112,877 | 219M |
| Stage-2 replay, reasoning (traces ≤ 8K tokens only) | 30,243 | 85M |
| Stage-2 replay, direct answers | 105,684 | 80M |
| Identity + safety (always kept) | 36,933 | 27M |

## Template (v2.1)

Qwen3/Hermes tool format: tools are listed in `<tools>` in the system turn, the assistant emits `<tool_call>` JSON, and
results come back as `<tool_response>` blocks in a user turn. Pass tools with
`tokenizer.apply_chat_template(messages, tools=tools)`. The hard `enable_thinking` switch from v2 is unchanged
(default off: no think tokens; on: the prompt ends with `<think>\n`). Tool-free conversations render exactly as in v2.
No tool list, tool response or `<think>` opener was ever a training label.

## Generation check (24 multilingual prompts, greedy)

- Default mode: 24/24 replies terminate, 0 with think tags; identity correct (OpenEuroLLM) in English, German and Swedish.
- Thinking mode: 21/24 close the trace and answer in the prompt language; 3 loop without closing (a haiku with
  syllable re-counting, a Danish question that drifts into English, the bat-and-ball puzzle).

## Known limitations

Thinking mode can loop without closing its trace (a termination problem inherited from Stage 2; more tokens do not fix
it). The RL stage targets this. DPO and RL are not yet applied.
