# Experiment: repair OELLM-9B long-context instruction following

## Decision and hypothesis

Warm-start `oellm-9b-256k-sft` and run a low-learning-rate mixed SFT stage at
8K-64K. This is the cheapest direct test of the leading diagnosis:

- the prelude learned long-position access;
- the original 4K-only SFT did not align instructions at long positions;
- multi-key binding, state tracking, multi-hop lookup, and aggregation need
  explicit supervised examples.

This first experiment intentionally stops at 64K. RULER already collapses from
16K through 64K, while the current FSDP data-parallel stack does not shard
128K/256K activations across sequence dimensions. Promote to 128K only after a
64K checkpoint demonstrates a real gain.

## What is new

- `scripts/build_long_context_sft.py` builds deterministic, auditable chat rows
  at 8K/16K/32K/64K.
- Tasks target the observed failures: multi-key, multi-value, multi-query,
  variable/state tracking, multi-hop resolution, aggregation, and no-answer.
- Evidence is distributed across early, middle, and late context positions.
- 30% of long rows contain compact evidence-grounded reasoning traces; the rest
  require concise answers.
- RULER templates, lexicon, prompts, and values are not used.
- A generation-aware Gemma template enables assistant-only loss while rendering
  the same `<start_of_turn>` / `<end_of_turn>` text as the published checkpoint.
- The trainer can opt into reproducible dataset interleaving rather than
  concatenating data sources by row.

## Data build on a LUMI login node

The default build creates 3,800 examples and approximately 80-100M unique SFT
tokens. It uses English for the targeted long tasks because the immediate gate
is English RULER; multilingual capability is protected with the existing general
EU replay dataset.

```bash
cd /scratch/project_465002530/users/bmoell/qwen35-posttrain

C=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
MODEL=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft
OUT=/scratch/project_465002530/users/bmoell/posttrain-data/oellm9b-longctx-sft-v1-64k

singularity exec -B /pfs,/scratch,/appl,/opt/cray "$C" bash -lc "
  export PYTHONPATH=/scratch/project_465002530/users/bmoell/pylibs-overlay:\${PYTHONPATH:-}
  cd /scratch/project_465002530/users/bmoell/qwen35-posttrain
  python scripts/build_long_context_sft.py \
    --output-dir '$OUT' \
    --tokenizer '$MODEL' \
    --chat-template templates/oellm_gemma_assistant_mask.jinja
  python scripts/validate_long_context_sft.py \
    --parquet '$OUT/train.parquet' \
    --tokenizer '$MODEL' \
    --chat-template templates/oellm_gemma_assistant_mask.jinja
"
```

Optional natural documents can replace 80% of the synthetic background by
passing one or more `--source-glob '/scratch/.../*.jsonl'`. The JSONL rows must
contain `text` or `content`. Do not point this at evaluation documents or RULER
data. Keep the first diagnostic reproducible; record source paths and licenses.

For a quick builder smoke before producing the full artifact:

```bash
python scripts/build_long_context_sft.py \
  --output-dir /tmp/oellm-longctx-sft-smoke \
  --lengths 1024:4,2048:4 \
  --overwrite
python scripts/validate_long_context_sft.py \
  --parquet /tmp/oellm-longctx-sft-smoke/train.parquet
```

## Why the row mixture is 0.90:0.10

The two configured sources are:

1. ordinary multilingual instruction replay, usually around 1K-2K tokens/row;
2. targeted long tasks, averaging around 20K-30K tokens/row.

Sampling them 90%/10% **by row** yields approximately 30-40% replay tokens and
60-70% long-task tokens after packing. A 35%/65% row mixture would make short
replay only a few percent of training tokens. Verify this assumption from the
generated manifest and a token-length sample of the replay data before the full
job; adjust the row probabilities if its mean length differs materially.

## Smoke and production launch

The launch helper sets the assistant-mask template, row mixture, deterministic
mixture seed, and rank-0 model loading. It prints the Slurm job ID.

```bash
cd /scratch/project_465002530/users/bmoell/qwen35-posttrain
bash scripts/submit_oellm9b_longctx_sft.sh smoke
```

Require all of the following before production:

- assistant mask validation reports a nonzero, non-full target mask;
- all three losses are finite;
- no tokenizer/chat-template assertion fails;
- the saved smoke checkpoint retains `max_position_embeddings=262144` and
  `rope_theta=64000000`.

Then submit:

```bash
bash scripts/submit_oellm9b_longctx_sft.sh production
```

The production config uses:

- 4 nodes / 32 GPUs;
- maximum physical sequence length 65,536;
- 250 optimizer steps, at most about 524M packed tokens;
- global batch 32 packed sequences;
- learning rate `1.5e-6`, cosine decay, 5% warmup;
- checkpoints every 50 steps;
- assistant-only loss.

Do not select the final checkpoint by SFT loss alone.

## Evaluation and promotion gate

Evaluate the starting checkpoint and checkpoints 50/100/150/200/250 with the
same pinned RULER environment. Run all 13 tasks, including CWE, at 4K, 8K, 16K,
32K, 64K, 128K, and 256K. Missing generations count as failures.

Also rerun EU holdouts and the Swedish generation rubric. Record response length
and document-continuation failures because targeted long SFT can otherwise trade
retrieval gains for verbosity.

Promote the best checkpoint only if:

1. the complete 13-task 4K mean falls by no more than 2 points;
2. 32K and 64K means each improve by at least 10 points;
3. 32K variable tracking improves by at least 20 points;
4. the gain appears in multiple categories, not only single-needle retrieval;
5. short multilingual instruction quality does not materially regress.

If the 64K experiment makes these gains, prepare a second 64K-128K stage with
lower LR and a smaller number of steps. If instruction behavior improves but
multi-key/state-tracking scores remain near the current ceiling, stop SFT work
and return to long-context continuation pretraining: the prelude is then the
likely bottleneck.
