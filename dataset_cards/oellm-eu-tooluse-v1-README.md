---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- function-calling
- tool-use
- agentic
- qwen
- qwen3.5
- openeurollm
size_categories:
- 10K<n<100K
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: train.parquet
- config_name: grpo
  data_files:
  - split: train
    path: grpo/train.jsonl
---

# oellm-eu-tooluse-v1

Function-calling / agentic post-training data, **normalized to Qwen3.5's native tool-call format**
(`<tools>…</tools>` in the system turn, `<tool_call>{json}</tool_call>` from the assistant). Built
for the OpenEuroLLM European post-training of Qwen3.5 (folded into the `Qwen3.5-4B-EU` "v-next"
mobile model as ~10% of the SFT mix, plus a verifiable RL stage).

The value here is **format unification**: three popular tool-use sources each encode calls
differently (Hermes JSON, ToolACE *pythonic* `[fn(a=1)]`, Glaive marker-delimited with
single-quoted argument strings). We parse each faithfully and re-render everything into the one
format Qwen3.5 actually consumes at inference, so training format == inference format.

## Contents

| Config | Split | Rows | Schema |
|---|---|---|---|
| `sft` (default) | train | **55,077** | `{"messages": [{role, content}…], "lang": "en"}` |
| `grpo` | train | **46,366** | `{"prompt": [messages], "answer": "<json [{name, arguments}]>", "language": "en"}` |

- **SFT** rows are ready for chat-template SFT (e.g. TRL `SFTTrainer`) — the tool definitions are
  baked into the system message with Qwen3.5's exact wrapper.
- **GRPO** rows are for verifiable RL (RLVR): single-call examples with clean gold, gradeable by a
  structure+name+arguments reward (no reward model). One example per source conversation.

## Sources & attribution

All upstream sources are Apache-2.0; this derived dataset is released under the same license.

| Source | Rows kept | Notes |
|---|---|---|
| [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 50,000 | JSON-native backbone; includes some "don't call" (relevance) turns |
| [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | 3,184 | multi-turn; pythonic calls parsed via `ast` → JSON |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 1,893 | already Hermes/Qwen-formatted |

Only rows whose calls parsed cleanly into `{name, arguments}` are kept (consistency over volume).

## Language

Currently **English only** — the upstream tool-use corpora are English. EU-language tool-use
(distilling the same schemas/calls with non-English user turns) is planned future work.

## Intended use

Teaching / evaluating function calling in the Qwen3.5 native format. Reproduction script:
`scripts/build_tooluse_sft.py` in the OpenEuroLLM post-training repo. A self-contained,
vLLM-free BFCL-style evaluator (`scripts/eval_tool_calling.py`) grades name/args/format accuracy
plus refusal/relevance on a held-out Glaive slice.

## Limitations

English-only; single-call gold for the GRPO split (parallel/multi-call left to SFT); ToolACE
coverage is a clean subset, not the full corpus. Inherits any artifacts of the upstream sources.
