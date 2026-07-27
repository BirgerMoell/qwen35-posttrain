---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-generation
tags:
- european
- multilingual
- qwen3.5
- post-training
- tool-use
- function-calling
- agentic
language:
- bg
- cs
- da
- de
- el
- en
- es
- et
- fi
- fr
- ga
- hr
- hu
- is
- it
- lt
- lv
- nl
- 'no'
- pl
- pt
- ro
- sk
- sl
- sv
- uk
---

# Qwen3.5-4B-EU-Tool

A **European, tool-use-capable** post-training of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
It folds **function-calling / agentic** data into the EU instruction mix, so the model calls tools
in Qwen3.5's **native `<tool_call>` format** while keeping European multilingual instruction
following. Defaults to fast no-think answers with a `/think` reasoning toggle. Part of OpenEuroLLM
Task 4.6, trained on **LUMI** (AMD MI250X / ROCm).

## Tool-calling evaluation (BFCL-style)

A vLLM-free, self-contained function-calling eval (`scripts/eval_tool_calling.py`): 300 held-out
prompts (150 that require a call + 150 that should **not** be called), scored on argument-exact
match, function-name match, well-formed output, and relevance (correct abstention):

| Metric | base Qwen3.5-4B | **this model (SFT)** |
|---|---:|---:|
| format_valid | 0.993 | 0.98 |
| name_acc | 0.993 | 0.98 |
| **args_exact** | 0.873 | **0.96** |
| abstain / relevance | 0.807 | **0.84** |

The base is already tool-capable (native support); this post-training's win is **argument
precision** (+8.7 pts args_exact) while **improving relevance** (knowing when *not* to call).

> **A note on RL.** A verifiable-reward GRPO stage was also trained, but its `tool_call_exact`
> reward had no abstention component, so it **reward-hacked** — driving relevance to 0.0 (it
> always calls). We therefore ship the **SFT**, not the GRPO. A fixed reward (crediting correct
> abstention, with no-call examples) is the follow-up.

## Training data

Openly-documented European post-training data (no proprietary data). SFT mix (~590k), packed, bf16:
- **General EU instructions** (EuroBlocks EU-multilingual + Tülu-3 English replay), no-think.
- **Reasoning traces** (chain-of-thought, think-format) — keeps the `/think` path sharp.
- **Defect-repair** (native text for weak EU languages), oversampled.
- **Tool-use (~10%)** — [`birgermoell/oellm-eu-tooluse-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1):
  ~55k function-calling examples normalized to Qwen3.5-native format from Glaive-v2 / ToolACE /
  Hermes (all Apache-2.0).

## Usage (with tools)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer  # transformers >= 5.5 (qwen3_5 arch)
import torch

mid = "birgermoell/Qwen3.5-4B-EU-Tool"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object",
        "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "Vad är vädret i Stockholm?"}]
ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                             return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=200)[0, ids.shape[1]:], skip_special_tokens=True))
# -> <tool_call>\n{"name": "get_weather", "arguments": {"city": "Stockholm"}}\n</tool_call>
```

- **No-think (default):** fast, direct answers / tool calls.
- **Thinking:** add `/think` (or `enable_thinking=True`) for step-by-step reasoning.

## License
Apache-2.0 (inherits the Qwen3.5-4B base license). Built within OpenEuroLLM.
