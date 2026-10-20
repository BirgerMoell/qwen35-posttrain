# OpenEuroLLM 9B two-stage post-training: run plan

Status 2026-10-20. Base: `openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`
(staged at `.../sft-migration-preflight/base-256k-anneal300b`, sha256-pinned).
Stage 1 makes a strong **non-reasoning Instruct** model. Stage 2 starts from the
accepted Stage-1 checkpoint and adds an opt-in thinking mode.

## Chat template and think mode (both stages)

`templates/oellm_qwen3_two_stage_v2.jinja` is the single template for training,
evaluation and release (tests: `tests/test_chat_template_v2.py`).

| Mode | Generation prompt | Trained labels |
| --- | --- | --- |
| default / `enable_thinking=False` | `<\|im_start\|>assistant\n` | `answer<\|im_end\|>\n` |
| `enable_thinking=True` | `<\|im_start\|>assistant\n<think>\n` | `trace\n</think>\n\nanswer<\|im_end\|>\n` |

`<think>` is always prompt-side, so the model never learns to open a think block
by itself; no empty think block exists anywhere. Direct rows render byte-identically
to the v1 template used by the earlier audit. Inline `<think>` in message content is
rejected; reasoning must be in `reasoning_content`. Earlier assistant turns carry no
reasoning (Qwen3 convention, matches inference).

## Stage 1 — Instruct SFT

**Data** `data/stage1-mix-v1` (builder `scripts/build_oellm_stage1_mix.py`): pre-tokenized
Parquet (`input_ids`, `assistant_masks`, provenance), so training uses exactly the audited
tokens. Every row: direct-answer, rendered with v2, no supervised think token, final answer
preserved, ≤4096 tokens (rejected, never truncated).

| Source | Selection | Why |
| --- | --- | --- |
| Excellent-SFT direct rows (2.62M) | all, keeps the audited split | curated core; same lineage as the earlier reference SFT |
| Translated Dolci, 12 langs (`cs de el es fi fr it nl pl ro sv uk`) | each source-family id assigned to exactly **one** language by hash (~40K rows/lang) | Excellent's direct pool is overwhelmingly English (all its Nemotron multilingual rows are reasoning); no prompt family repeated across languages |
| Aya (human) non-English target languages | ≤8K/lang, answer ≥50 chars | adds `da eu ga hu lt pt sq sr tr …`; capped because answers are short |
| Model identity v1 | 782 train rows × 2 exposures | middle of Codex's 1–3 exposure range; identity validation kept out |

Not included (deliberately): EU-Instruct-Synthetic (quality concerns in samples),
Aya Collection, tool-use, long-context data. Coverage is **not** 36 languages.

**Training** `configs/sft_oellm_stage1_instruct_v1_leonardo.yaml`,
`slurm/sft_oellm_stage1_instruct_v1_leonardo.sbatch`: 8 nodes × 4 A100, FSDP full shard,
BFD packing into 16K padding-free bins (flash-attn varlen, no cross-document attention),
liger fused CE (263K vocab), ~0.5M tokens/step, 1 epoch, LR 8e-6 cosine→10%, 3% warmup.
Checkpoints every 500 steps with optimizer; resubmitting the sbatch resumes.
Budget: ~2.4B tokens; expected ≈400–550 GPU-h (cap 1000).

Safety: per-batch abort if labels contain `<think>`/`</think>` or if no labels are masked.

**After training**: convert FP32 export → BF16, then evaluate before Stage 2:
general/EU holdouts, identity accuracy, per-language fluency spot checks, rate of
think tags in default-mode generations (must be 0), and NIAH/RULER at 32K–128K vs the
base (SFT is short-context only; add long-context replay if retrieval regresses).

## Stage 2 — Reasoning SFT (prepared, not launched)

**Pool** `data/stage2-reasoning-pool-v1` (builder `scripts/build_oellm_stage2_reasoning_pool.py`):
same pre-tokenized format plus `reasoning_tokens` and normalized `messages_json`.
Inline traces normalized into `reasoning_content`; rows must have one prompt-side `<think>`,
one supervised `</think>`, nonempty trace and answer; ≤32K tokens; `<image>`-placeholder
prompts dropped; Japanese excluded. Deterministic per-source sampling (pool, not final mix):

| Source | Keep | Answer language |
| --- | --- | --- |
| Nemotron multilingual `de es fr it` | 10% each (~60K/lang) | localized; **traces are English** |
| OpenThoughts3 think | 30% | en |
| Nemotron chat | 60% | en |
| SmolTalk2 multilingual8 think, multi-turn IF think, s1k, Aya think, table, everyday | 100% | mixed |

Built (job 61558735): **432,715 train / 396 validation rows, 3.17B rendered tokens**
(2.73B of them reasoning). By source (rows, tokens): Nemotron de/es/fr/it ≈60K and ≈420M
each; OpenThoughts3 84K / 1.11B; Nemotron chat 43K / 156M; SmolTalk multilingual8 59K / 150M;
LongAlign think 2.9K / 49M (902 over 32K rejected); small sets (IF multi-turn, s1k, Aya,
table) ≈4.5K rows. Rejections are `<image>` prompts and traces without a final answer.
The pool is larger than one Stage-2 epoch should be; select by token budget per source.

Observed in the source: every trace is English, including those under German/Spanish
prompts ("English trace, localized answer"). Translated Dolci-Think is the candidate for
native-language traces; compare both before committing.

**Planned Stage-2 mixture** (decide after Stage-1 evals): start from the Stage-1
checkpoint; ~70% reasoning-pool tokens, ~30% Stage-1 direct replay (same files, sampled)
so default mode does not degrade; pack to 32K; lower LR than Stage 1 (≈5e-6).
Trainer changes needed: a Stage-2 batch guard (no supervised `<think>`, `</think>` only
in reasoning rows) replacing `SFT_FORBID_THINK_LABELS`, and a token-weighted mixer for
the two globs. Evaluate both modes in every language, the mode switch itself, and
trace/answer language consistency. Decide then between one hybrid model and separate
Instruct + Thinking releases (if default-mode quality drops vs Stage 1).
