#!/bin/bash
# Tool-use (BFCL-style) eval only: base-4B vs v-next SFT vs +GRPO. Split out so it can be re-run
# independently of the heavier rubric/holdouts batch.
set -uo pipefail
O=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output
M=/scratch/project_465002530/users/bmoell/models
OUT=eval/results/tooluse_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
echo "== tool-use eval batch -> $OUT =="
for pair in "base:$M/Qwen3.5-4B" "vnext-sft:$O/qwen35-4b-eu-sft-vnext" "tooluse-grpo:$O/qwen35-4b-eu-tooluse-grpo"; do
  tag=${pair%%:*}; path=${pair#*:}
  echo "== tool-use eval: $tag =="
  python scripts/eval_tool_calling.py --model "$path" --n 300 --out "$OUT/tool_$tag.json" \
    || echo "!! tool_$tag FAILED"
done
echo "== tool-use eval DONE -> $OUT =="; ls -la "$OUT"
