---
base_model: openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b
library_name: transformers
pipeline_tag: text-generation
tags: [openeurollm, sft, instruct, multilingual]
---

# OELLM 9B 256K — Stage-1 Instruct v1 (internal)

Direct-answer instruction SFT of `openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`.
Stage 1 of a two-stage plan; a separate reasoning SFT (Stage 2) will start from this checkpoint.
**Internal research checkpoint** — training-data source terms have not been cleared for release.

## Chat template and thinking

ChatML (`<|im_start|>` / `<|im_end|>`). The bundled template has a hard `enable_thinking`
switch; this Stage-1 model is trained only in the default (no-thinking) mode, and no `<think>`
tokens were ever supervised. Use the default `enable_thinking=False`.

## Training

- Data (3.14M rows, 2.36B tokens, 1.71B supervised; one epoch): OpenEuroLLM-Excellent-SFT
  direct-answer rows (2.63M, mostly English); translated Dolci-Instruct in 12 languages
  (cs de el es fi fr it nl pl ro sv uk, ~40K rows each, one language per prompt family);
  Aya human-authored rows in non-English target languages (22K); OpenEuroLLM model-identity
  data (782 rows × 2). Assistant-only loss; rows over 4,096 tokens rejected, not truncated.
- 32 × A100 (Leonardo), FSDP, BFD packing to 16K without cross-document attention,
  LR 8e-6 cosine to 10 %, 3 % warmup, ~0.5M tokens/step, 4,569 steps (~395 GPU-h).
- Final internal eval loss 0.478 (from 0.577 at step 250).

## Known limitations

- Identity is unreliable: in a smoke test it claimed to be made by OpenAI in English and Swedish.
- Strict output-format instructions ("only the number", "nothing else") are often not followed exactly.
- Trained at ≤4K tokens; long-context retention versus the 256K base is not yet measured.
- Language coverage is uneven: strongest in English and the 12 translated-Dolci languages.
