---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
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
- ru
- sk
- sl
- sr
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

# Qwen3.5-4B-EU

A **European** post-training of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B),
specialised for the OpenEuroLLM target languages. The strongest model in our compact EU lineup
(2B + 4B). **Defaults to fast no-think answers** and keeps a `/think` reasoning toggle.

Built with a reproducible SFT → SimPO → GRPO pipeline on European multilingual data.

## Evaluation

OpenEuroLLM EU eval holdouts (38 European languages × 10 task buckets, deterministic scoring),
confirmed on the large `test_public` split (6080 rows) — not just the noisier dev set:

| Model | Overall accuracy (test_public) |
|---|---|
| Qwen3.5-4B (base) | ~49.9% |
| Qwen3.5-4B-EU (v1: SFT→SimPO, light) | 52.5% |
| Qwen3.5-4B-EU (v2: SFT→SimPO→GRPO, more data + longer) | 56.1% |
| **Qwen3.5-4B-EU (v3: + defect-repair)** | **58.9%**  (**+2.8 vs v2, ~+9 vs base**) |

This repo ships the **v3** weights — the v2 recipe **plus a defect-repair SFT fold-in** that fixes
the previously-weak European languages. On the holdouts the once-broken languages are now *at parity*
with the rest (defect-language mean **60.3** vs 58.2 for the others; e.g. is 56, mt 70, da 73, bg 68),
with **no collapsed language** across all 38 — and clean task buckets (summarization 93, no_answer 84,
reasoning-math 96). The v2/v3 recipe helps at 4B scale but overfit at 2B, so
[Qwen3.5-2B-EU](https://huggingface.co/birgermoell/Qwen3.5-2B-EU) keeps its lighter recipe.

## Training data

All stages use openly-documented European post-training data (OpenEuroLLM Task 4.6); no
proprietary data. Three stages: **SFT → SimPO → GRPO** (the "v2" recipe — more data + longer
training + a verifiable-RL stage; this gave a clear, robust gain at 4B scale).

**1. Supervised fine-tuning (SFT)** — full mix (~1.27M), packed, bf16:
- **General EU instructions (~85%, full ~1.08M)** — Dolci `tulu3-euroblocks-85-15`: **EuroBlocks**
  EU-multilingual instruction data (85%) + **Tülu-3** (`allenai/tulu-3`) English replay (15%),
  adding EU-language instruction-following while preserving English. No-think formatting (default
  fast answers).
- **Reasoning traces (~15%, ~190k)** — chain-of-thought SFT (Dolci-Think / OpenThoughts /
  OpenMathInstruct-family), think-format, keeping the `/think` path sharp. *(Currently
  English+Finnish reasoning — the known multilingual-reasoning gap; EU-language reasoning
  distillation is in progress.)*

**2. Preference optimisation (SimPO, reference-free)** — preference accuracy ≈ 0.88:
- [**`birgermoell/oellm-eu-exam-mcq-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1)
  (preference/DPO track) — European exam multiple-choice preference pairs (correct-over-incorrect),
  **~35 languages, 28 sources** (national, medical/licensing, academic exams), mixed licenses.

**3. Verifiable RL (GRPO / RLVR)** — reward climbed 0.78 → 0.85:
- [**`birgermoell/oellm-eu-exam-mcq-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1)
  (GRPO/RLVR track) — verifiable letter-match reward (`mcq_letter_exact`, β=0, no reward model),
  optimizing EU-language answer *correctness* directly (no translationese in the loop).

**Evaluation** — held out from training:
[**`birgermoell/oellm-eu-eval-holdouts-v1`**](https://huggingface.co/datasets/birgermoell/oellm-eu-eval-holdouts-v1)
— 38 EU languages × 10 task buckets, deterministic per-task scoring.

## Swedish capability (open-ended rubric)

A vLLM-free Swedish eval (Qwen3.5 isn't yet supported by EuroEval's fast backends): 24 prompts
across 9 categories (sentiment, Swedish knowledge, reasoning, summarization, linguistic
correctness, instruction-following, common sense, creative), scored 1–5 on
**språkkvalitet / korrekthet / instruktion / hjälpsamhet**, plus automatic langdetect Swedish-purity.

| Dimension | 2B-EU | **4B-EU** | (9B-EU) |
|---|---|---|---|
| Swedish-purity (langdetect) | 1.00 | **1.00** | 1.00 |
| Språkkvalitet (fluency/grammar) | 4.3 | **4.6** | 4.7 |
| Korrekthet (factual/logical) | 2.7 | **4.4** | 4.6 |
| Instruktion (follows constraints) | 3.4 | **4.0** | 4.5 |
| Hjälpsamhet | 2.9 | **4.3** | 4.5 |

The 4B is a **big step up from the 2B on factual accuracy** (korrekthet 2.7 → 4.4): it gets the
three largest Swedish lakes right (Mälaren, not Göta älv), explains the 4 % riksdag threshold,
knows Selma Lagerlöf won the Nobel, and handles *de/dem* correctly — all while writing fully
in-Swedish (purity 1.00). One wart: a constrained "list five cities" task degenerated into a
repetition loop. Tooling: `scripts/eval_swedish_rubric.py` + `data/swedish_rubric_prompts.jsonl`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer  # transformers >= 5.5 (qwen3_5 arch)
import torch

mid = "birgermoell/Qwen3.5-4B-EU"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")

msgs = [{"role": "user", "content": "Vad är meningen med livet? Svara kort."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=200)[0, ids.shape[1]:], skip_special_tokens=True))
```

- **No-think (default):** fast, direct answers — best for interactive use.
- **Thinking:** add `/think` to the user turn (or `enable_thinking=True`) for step-by-step reasoning.

## On-device status
- **bf16 (this repo):** runs via `transformers` / vLLM. **Recommended today.**
- **GGUF (llama.cpp / Ollama):** ⚠️ not supported yet — llama.cpp drops Qwen3.5's hybrid
  *linear-attention* (Gated Delta Net) layers during conversion (load fails). Blocked upstream.
- **MLX (Apple):** ⚠️ not supported yet — `mlx-lm` raises `Model type qwen3_5 not supported`.

Qwen3.5's hybrid architecture is very new (Mar 2026); the on-device quantization runtimes haven't
added support yet. Phone deployment works once llama.cpp / MLX add Gated-Delta-Net support.

## License
Apache-2.0 (inherits the Qwen3.5-4B base license). Built within OpenEuroLLM.
