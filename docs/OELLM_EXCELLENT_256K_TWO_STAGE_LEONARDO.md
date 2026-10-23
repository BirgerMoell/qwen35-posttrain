# Excellent-SFT data on the 256K anneal300b base: Leonardo migration

The source-by-source and 36-language Instruct/Reasoning selection plan is in
`OELLM_36_LANGUAGE_INSTRUCT_REASONING_DATA_PLAN.md`; the datasets listed there
are candidates until their quality, overlap, token, and release gates pass.

Status (2026-10-20): tokenizer, template, eight real source rows, and CPU-only
TRL collator preflights passed. Leonardo job `61531798` staged and verified
the 18.2 GB base weights. Four-A100 job `61534358` completed **one BF16-loaded
optimizer step** on four real direct-answer rows (loss 1.167, 4,579 tokens)
and saved a checkpoint. Its FSDP full-state export is FP32/34 GB despite the
BF16 load, so this is only a pipeline smoke; it is not a releasable or
quality-tested instruct model. Job `61535956` staged and verified all 62
Excellent-SFT Parquet shards (6,187,716 rows, 27,609,764,587 bytes). CPU job
`61536087` completed a structural survey of all 6,187,716 rows. **No
production SFT has begun.**

## Exact lineage

- Starting **weights**: `openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`,
  revision `783ab41b06d1937b10b5b8fde232051f7ffd2998`.
  `max_position_embeddings=262144`, `rope_theta=64000000`, BF16,
  model embedding vocabulary padded to 263168 rows. The base has not been
  instruction tuned.
- Starting **data**: `openeurollm/OpenEuroLLM-Excellent-SFT`, revision
  `c4ee55804f256c5f8cc4a6ef3fdecbe4a4ea95e0` (6,187,716 rows,
  one train split, 62 Parquet shards). The dataset card says the mixture was
  decontaminated, quality filtered, normalized, limited to 32,768 tokens and
  conversation-deduplicated. The `messages` field is `{role, content}` turns,
  not a pre-rendered prompt. The `tools` field is non-null for 11,873 rows;
  leave those out of the first two text-only stages.
- Reference **format/tokenizer**, not weights:
  `openeurollm/OpenEuroLLM-Prelude-SFT-Excellent-Instructions`, revision
  `6551e54cb2fb5f5b9c77fe222b6584624b7acb04`. That model was trained
  from a different, 128K Prelude anneal300b base, so its weights are not our
  starting point. Its `tokenizer.json` remaps eight reserved IDs from the
  base; all other token IDs remain the same. In particular 3/4 become
  `<|im_start|>`/`<|im_end|>` and 17/18 become `</think>`/`<think>`.
  Package and validate that exact tokenizer with
  `scripts/prepare_oellm_two_stage_tokenizer.py`, setting chat EOS to ID 4.
  Do not pair the base's raw tokenizer with the ChatML template.
- Published **evaluation baseline, not starting weights**:
  [`openeurollm/oellm-9b-256k-sft`](https://huggingface.co/openeurollm/oellm-9b-256k-sft)
  used the earlier 256K Prelude base, a Tulu/EuroBlocks/Nemotron mixture,
  4K packed sequences and `assistant_only_loss=false`. Its model card reports
  that the Nemotron reasoning slice was 18.77% of rows but 64.41% of tokens,
  no held-out validation split, and a pathological 3M-token duplicated row.
  These are concrete failure modes to prevent, not a recipe to replicate.

Reference cards: [Excellent-SFT dataset](https://huggingface.co/datasets/openeurollm/OpenEuroLLM-Excellent-SFT),
[reference SFT model](https://huggingface.co/openeurollm/OpenEuroLLM-Prelude-SFT-Excellent-Instructions),
[256K base](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b),
[taskboard SFT report](https://github.com/OpenEuroLLM/Taskboard/issues/416).

## Target and proposed sequence

The target is a useful, independently evaluated, multilingual **direct-answer
instruct model**, not a tag-format demonstration. Keep the later reasoning
curriculum possible by reserving the same tokenizer IDs and template, never
teaching empty think spans, and holding back valid reasoning traces. Do not
equate dataset size or a falling train loss with instruction-following quality.

1. Materialize an auditable **Instruct SFT** subset of *direct-only* text
   conversations. Reject any assistant turn with an inline think block, any
   nontext/tool turn, malformed tags, empty answer, or overlength rendering.
   Train with `assistant_only_loss=true`, `packing=false` for the first smoke,
   and `templates/oellm_qwen3_no_empty_assistant_mask.jinja`. This template
   never inserts an empty think span into a direct answer or generation prompt.
2. Starting from the Stage 1 checkpoint, materialize **Reasoning SFT** rows
   whose assistant content starts with a nonempty, closed `<think>` block and
   ends with a nonempty final answer. Mix a measured sample of Stage 1 direct
   rows as replay. Do not infer that every row from a "reasoning" source has
   a valid trace; classify and audit the rendered labels. Prioritize verified
   math/code and multilingual reasoning for the first pilot, while measuring
   source/language coverage and answer quality.
   Decide and evaluate a clear inference policy: default direct answers, with
   nonempty visible `<think>...</think>` only when a documented reasoning mode
   is requested. Construct explicit mode-conditioned prompts plus direct
   replay; otherwise mixing traces and direct answers on indistinguishable
   prompts may teach an unreliable toggle.
3. The 1-step GPU smoke on 4 A100s passed loading, assistant masking, forward,
   backward, optimizer update and save. It used a 4K cap, no packing, and the
   256K *weights*. `dtype: bfloat16` is the effective TRL YAML key;
   `torch_dtype: bfloat16` was ignored and caused an earlier FP32 diagnostic
   job (`61533906`). The saved FSDP full state is still FP32 and needs an
   export/reload fix. Do not use the LUMI ROCm Slurm script.
4. Choose production mix, sequence length and budget only after counting
   direct/valid-reasoning/rejected rows by source and language, supervised
   tokens, lengths, source terms, and held-out prompt overlap. The reference
   SFT report gives ~29.15B Prelude-tokenized tokens per epoch of the full
   mixture, so two full epochs are not a tiny migration test. The issue's
   update also reports no Elo improvement signal, despite healthy training
   loss/accuracy; that is a reason to compare alternative mixes and stop
   on held-out behavior rather than repeat two epochs blindly.
5. Run a **small curated pilot**, then an explicitly budgeted production run
   with fixed validation and checkpoints. A pilot and an early production
   checkpoint must beat the base and useful published reference baselines on
   held-out instruction following, EU-language quality, factuality, code/math
   final answers, safety, and 4K/32K/128K/256K retrieval. Track hallucination,
   repetitive output, unwanted think tags, and catastrophic forgetting. Avoid
   calling the result 2026 SOTA until external benchmark comparisons support it.

Freeze the Stage-1 evaluation set *before* materializing training rows: use
the repository's existing `eval/README.md` and EU holdout pack where valid,
instruction-constraint tests such
as IFEval, ArenaHard-EU, native-language human review, and independent
code/math answer checkers. Keep an untouched final test slice apart from
pilot model selection. Compare the 256K anneal base, the published earlier
256K SFT, the 128K Excellent-SFT reference, and each new checkpoint under
the same decoding and prompt protocol. Report per-language results and
long-context retrieval by length/depth, not just one average score.

## Stage-1 data policy

The full pinned-corpus structural survey found 2,649,634 direct rows,
3,526,040 reasoning rows, 11,873 tool-declared rows, and 169 malformed or
unsupported rows. The direct pool is only **42.8%** of the reference corpus by
row count, before rendered-length, quality, licensing, and overlap filters.
This survey is not a final training manifest. Its source report is at
`$OELLM_WORK/sft-migration-preflight/reports/excellent-sft-structural-profile.json`.
All five large Nemotron multilingual slices (`de`, `es`, `fr`, `it`, `ja`)
classify as reasoning, roughly 3.10M rows total, so they cannot supply the
direct multilingual Stage-1 core. The direct SmolTalk2 multilingual-8 slice
has only 48,737 rows. **A language-balanced supplemental direct-answer
corpus is therefore necessary** for the intended EU instruct checkpoint,
especially Swedish. Do not mistake source names for a validated language
distribution; measure language on the final selected rows.

- **Core candidate:** direct-only, text-only, license-cleared rows from the
  pinned Excellent-SFT corpus. Its 6.19M rows are not all Stage-1 eligible;
  `scripts/profile_excellent_sft.py` provides a structural source survey, and
  the final builder must additionally render/tokenize every selected row.
- **Possible EU-language additions:**
  [EU-Instruct-Synthetic](https://huggingface.co/datasets/openeurollm/EU-Instruct-Synthetic)
  is Apache-2.0 and explicitly multilingual. Add only verified subsets after
  human native-language spot checks, instruction-compliance checks, and
  prompt/answer deduplication; some previewed Czech rows appear unnatural.
  [Dolci-Instruct-SFT-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated)
  includes Swedish, but verify original Dolci terms, translation quality,
  and overlap with the English Dolci portion already inside Excellent-SFT.
- **Small targeted repair candidate:**
  [oellm-eu-defect-repair-sft-v1](https://huggingface.co/datasets/birgermoell/oellm-eu-defect-repair-sft-v1)
  can be tested as a low-resource-language repair slice, with its CC-BY-SA
  provenance/attribution and quality reviewed. Do not let 57K narrow repair
  examples dominate general instruction behavior.
- **Not Stage-1 core:** math/code RLVR problem sets, evaluation holdouts,
  medical drafts without source-level review, tool-execution transcripts,
  and any assistant reasoning trace. If the Stage-1 product must support
  function calling, add a separately rendered **non-reasoning tool-use SFT
  substage** and BFCL-style evaluation before release. The present text-only
  template deliberately rejects tools; simply mixing tool JSON into it would
  not teach a reliable tool protocol. Genuine reasoning remains a later
  curriculum with its own data and evaluations.
- Set per-source and per-language caps by **supervised tokens**, not row counts.
  Limit near-duplicate synthetic patterns and the dominant Nemotron portion;
  preserve high-quality multi-turn, instruction-constraint, EU-language,
  long-input, and direct code/math answer coverage. Make train/validation/test
  splits by canonical prompt hash *before* sampling; decontaminate against
  evaluation prompts. Keep a manifest of source revision, terms, filters,
  sampling weights, row counts, token counts, and rejection reasons.
- The [Nemotron-v2 source card](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)
  specifically warns that publicly distributed models trained on its
  synthetic outputs may be subject to Qwen/DeepSeek redistribution terms.
  Because Nemotron is the largest Excellent-SFT component, do not interpret
  the aggregate dataset's `other` license as release clearance. Obtain a
  source-level decision before a public checkpoint.

Before choosing a full budget: inspect stratified native-language samples,
measure rendered sequence/token distributions at 4K/8K/16K and longer
curriculum lengths, and price the resulting token budget in measured
Leonardo tokens/s. The base supports 256K positions, but that does not mean
every SFT step should be 256K; long-context ability needs separate retention
checks and a selected long-example curriculum.

## Non-negotiable gates

- `scripts/audit_two_stage_sft.py` on a real sample with the **exact** saved
  tokenizer and template: no empty/malformed think spans, no prompt loss,
  full final answer preserved, no silent truncation.
- CPU-only `scripts/preflight_trl_two_stage.py` through the **container's**
  real `SFTTrainer` and collator, including direct and reasoning source rows.
- Save tokenizer, chat template, model config, data revision, source counts,
  code revision, and preflight hashes with every checkpoint.
- Fix and independently reload a correctly typed export before promotion;
  a BF16 training load does not imply a BF16 saved full-state model.
- Evaluate before and after each stage on fixed held-out instruction prompts,
  verifier-checked reasoning tasks, empty-think rate, multilingual tasks, and
  long-context retrieval at several lengths. A model that merely emits
  nonempty think tags without improving valid answers is not a success.

The dataset card labels the mixture's license as mixed/other; review each
source's terms before publishing derivative weights or redistributing data.

## Completed preflight evidence

- The packaged tokenizer preserved every non-reserved token ID from the base,
  kept exactly 262145 tokenizer IDs within the model's 263168-row padded
  embedding table, and set chat EOS to ID 4. The published reference
  `tokenizer.json` SHA-256 is
  `5123f74b254340ffadc9218925030916e47058178967cdd7110b39b848db23fd`;
  the base `tokenizer.model` SHA-256 matches its model card:
  `ac214401227404105817141c21c6693e3b17c878465a42765d2c345310e003ce`.
- Four direct and four nonempty-reasoning public dataset rows at the pinned
  revision passed the render/assistant-mask audit. Maximum rendered lengths
  were 1785 and 2956 tokens respectively. This is a deliberately small
  smoke sample, not a distributional or quality assessment.
- Leonardo's shared `post-training-25.10-transformers-olmo3-patch.sif` container
  passed real TRL SFTTrainer preprocessing and collator checks on one direct
  and one inline-reasoning source row, in addition to synthetic sentinels.
  It did not load the 9B weights or perform a forward/backward pass.
- `configs/sft_oellm_excellent_256k_leonardo_one_step.yaml` and
  `slurm/sft_oellm_excellent_256k_leonardo_one_step.sbatch` were used for the
  one-step FSDP/CUDA smoke only. The config's four data rows are not a
  proposed production mix.
