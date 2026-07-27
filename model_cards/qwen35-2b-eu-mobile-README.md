---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
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
library_name: transformers
pipeline_tag: text-generation
tags:
- european
- multilingual
- qwen3.5
- post-training
- on-device
---

# Qwen3.5-2B-EU

A compact **European** post-training of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B),
specialised for the OpenEuroLLM target languages. Small enough for laptops / future on-device use,
and it **defaults to fast no-think answers** while keeping the `/think` reasoning toggle.

Part of a mobile-oriented lineup (2B + 4B). Built with a reproducible SFT → SimPO pipeline on
European multilingual data.

## What we changed and why it's better

Post-training lifts the base model substantially on the **OpenEuroLLM EU eval holdouts**
(38 European languages × 10 task buckets, deterministic scoring, full `dev` split = 1368 rows):

| Model | Overall accuracy |
|---|---|
| Qwen3.5-2B (base) | 32.7% |
| **Qwen3.5-2B-EU** | **45.6%**  (**+12.9**) |

Gains are broad across languages and tasks — e.g. grounded QA 43→95, civic/safety 12→90,
reasoning-math 32→61; per-language examples: uk 31→53, sk 33→58, is 31→56, nl 22→47, eu 25→47.
(Per-cell language/bucket numbers are small-sample; the +12.9 overall is the robust signal.)

## Training data

All stages use openly-documented European post-training data (OpenEuroLLM Task 4.6); no
proprietary data. Two stages: **SFT → SimPO** (this 2B keeps the lighter "v1" recipe — at 2B
scale, longer training + a GRPO stage *overfit* and did not help).

**1. Supervised fine-tuning (SFT)** — ~400k examples, packed, bf16:
- **General EU instructions (~85%, ~340k)** — Dolci `tulu3-euroblocks-85-15`: **EuroBlocks**
  EU-multilingual instruction data (85%) + **Tülu-3** (`allenai/tulu-3`) English replay (15%),
  adding EU-language instruction-following while preserving English. Formatted no-think (direct
  answers) so the model defaults to fast replies.
- **Reasoning traces (~15%, ~60k)** — chain-of-thought SFT (Dolci-Think / OpenThoughts /
  OpenMathInstruct-family), think-format, so the `/think` path stays sharp. *(Currently
  English+Finnish reasoning — the known multilingual-reasoning gap; EU-language reasoning
  distillation is in progress.)*

**2. Preference optimisation (SimPO, reference-free)** — preference accuracy ≈ 0.79:
- [**`birgermoell/oellm-eu-exam-mcq-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1)
  (preference/DPO track) — European exam multiple-choice preference pairs (correct-over-incorrect)
  across **~35 languages, 28 sources** (national, medical/licensing, and academic exams), mixed
  licenses (filterable per row).

**Evaluation** — held out from training:
[**`birgermoell/oellm-eu-eval-holdouts-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-eval-holdouts-v1)
— 38 EU languages × 10 task buckets, deterministic per-task scoring.

## Swedish capability (open-ended rubric)

A vLLM-free Swedish eval (Qwen3.5 isn't yet supported by EuroEval's fast backends): 24 prompts
across 9 categories (sentiment, Swedish knowledge, reasoning, summarization, linguistic
correctness, instruction-following, common sense, creative), scored 1–5 on
**språkkvalitet / korrekthet / instruktion / hjälpsamhet**, plus automatic langdetect Swedish-purity.

| Dimension | **2B-EU** | (4B-EU) | (9B-EU) |
|---|---|---|---|
| Swedish-purity (langdetect) | **1.00** | 1.00 | 1.00 |
| Språkkvalitet (fluency/grammar) | 4.3 | 4.6 | 4.7 |
| Korrekthet (factual/logical) | **2.7** | 4.4 | 4.6 |
| Instruktion (follows constraints) | 3.4 | 4.0 | 4.5 |
| Hjälpsamhet | 2.9 | 4.3 | 4.5 |

The 2B writes **fully in-Swedish, fluent prose** (purity 1.00) and is good at summarization and
grammar, but is **factually unreliable at 2B scale** — it hallucinates Swedish facts (e.g. called
Göta älv a lake; said the sky is blue "because of clouds"). For accuracy, prefer the
[4B-EU](https://huggingface.co/birgermoell/Qwen3.5-4B-EU) /
[9B-EU](https://huggingface.co/birgermoell/Qwen3.5-9B-EU-SFT-GRPO).
Tooling: `scripts/eval_swedish_rubric.py` + `data/swedish_rubric_prompts.jsonl`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer  # transformers >= 5.5 (qwen3_5 arch)
import torch

mid = "birgermoell/Qwen3.5-2B-EU"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")

msgs = [{"role": "user", "content": "Vad är meningen med livet? Svara kort."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=200)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

- **No-think (default):** fast, direct answers — best for interactive/mobile use.
- **Thinking:** add `/think` to the user turn (or `enable_thinking=True` in the chat template) for
  step-by-step reasoning on harder tasks.

## On-device status
- **bf16 (this repo):** runs via `transformers` / vLLM. **Recommended today.**
- **GGUF (llama.cpp / Ollama):** ⚠️ not supported yet — llama.cpp drops Qwen3.5's hybrid
  *linear-attention* (Gated Delta Net) layers during conversion (load fails with a missing
  tensor). Blocked upstream.
- **MLX (Apple):** ⚠️ not supported yet — `mlx-lm` raises `Model type qwen3_5 not supported`.

Qwen3.5's hybrid architecture is very new (Mar 2026), so the on-device quantization runtimes have
not yet added support. True phone deployment will work once llama.cpp / MLX add Gated-Delta-Net
support; until then use the bf16 weights via transformers/vLLM.

## Limitations
2B scale: best for assistant/chat in EU languages, not a frontier reasoner. Some buckets
(tool-calling, locale-formatting, long summarization) remain weak at this size. Inherits the base
model's general knowledge and any biases.

## License
Apache-2.0 (inherits the Qwen3.5-2B base license). Built within OpenEuroLLM.
