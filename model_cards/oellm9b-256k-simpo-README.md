---
license: apache-2.0
base_model: birgermoell/oellm-9b-256k-sft
library_name: transformers
pipeline_tag: text-generation
tags:
- openeurollm
- qwen3
- long-context
- 256k
- multilingual
- instruction-tuning
- preference-optimization
- simpo
language: [en, sv, de, fr, es, it, nl, pl, pt, cs, fi, da, el, bg, hr, hu, ro, sk, sl, et, lt, lv, ga, mt, eu, gl, is, nb, nn, sr, uk, ca, mk, sq, oc, lb, bs]
---

# OELLM 9B 256K SimPO

Experimental preference-optimized checkpoint derived from
[`birgermoell/oellm-9b-256k-sft`](https://huggingface.co/birgermoell/oellm-9b-256k-sft),
which in turn is based on
[`openeurollm/oellm-9b-256k-theta64m-prelude`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude).
This is the second stage of an OpenEuroLLM post-training experiment on LUMI: supervised fine-tuning
followed by reference-free SimPO preference optimization.

The model is a dense Qwen3 9B checkpoint using the OpenEuroLLM 256K tokenizer. Its architecture
retains `max_position_embeddings=262144` and RoPE theta `64000000` from the base model.

## Intended use and interpretation

This checkpoint is intended for research on multilingual instruction following, preference
optimization, and retention of long-context retrieval after post-training. It is not presented as a
reasoning-specialized or production-ready assistant. The 262,144-token configuration is an
architectural input limit, not by itself evidence of accurate retrieval or reasoning at that length.

The preceding SFT checkpoint retrieved natural-word passphrases in 40/40 single-needle tests across
512 to approximately 262K tokens. Those results belong to the SFT model and must not be assumed to
transfer unchanged through SimPO. Equivalent long-context evaluation of this exact SimPO artifact is
the next required comparison.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "birgermoell/oellm-9b-256k-simpo"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [{"role": "user", "content": "Förklara allemansrätten kort på svenska."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Long-context inference near 256K requires substantial memory for the KV cache and will generally
need multiple accelerators. Keep `max_position_embeddings=262144` and `rope_theta=64000000`
unchanged.

## Training

| Field | Value |
|---|---|
| Starting checkpoint | `birgermoell/oellm-9b-256k-sft` |
| Method | Full-parameter SimPO via TRL `CPOTrainer`, reference-free |
| Preference data | 1,089,256 multilingual EU exam preference pairs |
| Maximum sequence / prompt length | 1,024 / 768 tokens |
| Steps | 1,500 (about 0.176 epoch) |
| Global pair batch | 128 (16 ranks x 1 pair x 8 accumulation steps) |
| SimPO beta / gamma | 2.0 / 0.5 |
| Optimizer schedule | AdamW, peak LR 5e-7, cosine decay, 3% warmup |
| Hardware | 2 LUMI-G nodes, 16 AMD MI250X GCDs |
| Runtime | 5h 23m |
| Aggregate training loss | 0.6549 |
| Last logged loss | 0.6001 |
| Last preference accuracy | 0.7734 |
| Last reward margin | 1.757 |

The published weights are an unquantized BF16 Hugging Face export of the consolidated training
checkpoint. The completed training job used approximately 43.1 LUMI GPU-hours.

## Export and inference validation

The exact published BF16 export passed a Transformers GPU smoke test covering:

- model and tokenizer loading;
- finite forward-pass logits;
- deterministic generation in Swedish, English, and German;
- BF16 weight dtypes and sharded safetensors integrity;
- the included native turn-token chat template;
- EOS token ID 4 (`<end_of_turn>`);
- 262,144-token context metadata and RoPE theta 64M.

This validates that the Hub artifact loads and generates; it is not a capability or safety evaluation.
The export contains four safetensors shards totaling 18.20 GB. All 399 floating tensors were
validated as BF16. Deterministic smoke responses were `Stockholm`, `42`, and a coherent one-sentence
German explanation of photosynthesis. The export and validation ran as LUMI job `21040068`.

## General and Swedish evaluation

The exact BF16 artifact was evaluated with the same protocol used for the preceding SFT model:
24 sampled open Swedish prompts and a deterministic 1,368-example multilingual development holdout,
including 36 Swedish examples. Long-context rows were excluded from this suite and remain a separate
evaluation question.

| Metric | SFT | SimPO | Delta |
|---|---:|---:|---:|
| Multilingual holdout overall | 23.2% | 21.6% | -1.6 pp |
| Swedish holdout | 33.3% | 33.3% | 0.0 pp |
| Instruction following | 63.2% | 59.2% | -4.0 pp |
| Grounded QA | 58.6% | 53.9% | -4.7 pp |
| Reasoning/math | 5.9% | 0.0% | -5.9 pp |
| Mean Swedish response length | 72.3 words | 56.9 words | -15.4 words |
| Swedish language probability | 0.96 | 0.96 | 0.00 |

SimPO made several sampled answers shorter and less repetitive, but did not improve measured general
quality. Manual review found continued failures in basic arithmetic, logical inference, factual recall,
Swedish grammar analysis, and exact list constraints. See the
[detailed Swedish capability report](./eval/SWEDISH_CAPABILITY.md) and raw
[Swedish generations](./eval/swedish-rubric.json) and
[holdout results](./eval/eu-holdouts-dev.json). Evaluation ran as LUMI job `21040549`.

## Long-context evaluation status

SimPO-specific long-context retrieval has not yet been completed. The SFT checkpoint's 40/40
natural-word single-needle retrieval result is only a baseline for the planned same-grid comparison;
it must not be attributed to this checkpoint. Future results should separately report full-value
retrieval and exact-output-format compliance so verbosity is not mistaken for a memory failure.

## Limitations

- Experimental research checkpoint; not a production assistant.
- No completed safety evaluation.
- Long-context retrieval has not yet been re-measured after SimPO.
- The training preference accuracy is an optimization diagnostic, not a held-out benchmark.
- General holdout accuracy regressed by 1.6 percentage points from SFT; reasoning/math scored 0.0%.
- The preference data is exam-oriented, synthetic/translated in part, and may contain translationese.
- The starting SFT model has known weaknesses in factuality, reasoning, grammatical correction,
  structured output, and exact-constraint following; SimPO should not be assumed to eliminate them.
- Inherits limitations, biases, and knowledge gaps from the base model and training data.

## Reproduction

Training, export, and evaluation code:
<https://github.com/BirgerMoell/qwen35-posttrain>, especially
`docs/RUNBOOK_oellm9b_256k_posttrain.md`.

## License

Apache 2.0, inherited from the base model. Consult the component dataset cards for training-data terms.
