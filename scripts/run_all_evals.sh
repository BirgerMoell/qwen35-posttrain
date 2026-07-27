#!/bin/bash
# Run all three post-training evals with transformers (device_map=auto, no vLLM). Each eval is
# wrapped so one failure doesn't abort the rest; every eval writes its own JSON. Invoked inside the
# LUMI container by slurm/eval_batch_lumi.sbatch.
set -uo pipefail
O=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output
M=/scratch/project_465002530/users/bmoell/models
OUT=eval/results/batch_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
echo "== eval batch -> $OUT =="

# 1) Tool-use (BFCL-style): base vs v-next SFT vs +GRPO — the agentic payoff
for pair in "base:$M/Qwen3.5-4B" "vnext-sft:$O/qwen35-4b-eu-sft-vnext" "tooluse-grpo:$O/qwen35-4b-eu-tooluse-grpo"; do
  tag=${pair%%:*}; path=${pair#*:}
  echo "== tool-use eval: $tag =="
  python scripts/eval_tool_calling.py --model "$path" --n 300 --out "$OUT/tool_$tag.json" \
    || echo "!! tool_$tag FAILED"
done

# 2) Swedish rubric GENERATIONS (langdetect purity + raw answers; no LLM-judge — scored offline)
for pair in "2b-final:$O/qwen35-2b-eu-sft-v3-final" "4b-final:$O/qwen35-4b-eu-sft-v3-final" "9b-v2:$O/qwen35-9b-eu-grpo-v2full"; do
  tag=${pair%%:*}; path=${pair#*:}
  echo "== swedish rubric: $tag =="
  python scripts/eval_swedish_rubric.py --model "$path" --prompts data/swedish_rubric_prompts.jsonl \
    --out "$OUT/svrubric_$tag.json" --tag "$tag" || echo "!! svrubric_$tag FAILED"
done

# 3) EU eval holdouts on the 9B-v2 flagship (deterministic per-task scoring), test_public split
echo "== EU holdouts: 9b-v2 (test_public) =="
python scripts/eval_eu_holdouts.py --model "$O/qwen35-9b-eu-grpo-v2full" \
  --data data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/test_public.jsonl \
  --out "$OUT/holdouts_9b_v2.json" --batch-size 32 || echo "!! holdouts_9b FAILED"

echo "== eval batch DONE -> $OUT =="
ls -la "$OUT"
