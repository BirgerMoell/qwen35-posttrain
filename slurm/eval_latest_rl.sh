#!/bin/bash
# Evaluate the newest step_* checkpoint of an RL run inside the current allocation (1 node x 4 GPUs).
#   eval_latest_rl.sh <outputs dir> <name> quick|tool|code
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
out=$1; name=$2; kind=$3
model=$(ls -d "$out"/*_checkpoints/step_* 2>/dev/null | sort -t_ -k2 -V | tail -1)
[[ -n "$model" ]] || { echo "no checkpoint under $out"; exit 1; }
step=$(basename "$model"); echo "evaluating $model as $name-$step"
cd "$OELLM_WORK/sft-migration-preflight"
case "$kind" in
  quick) MODEL=$model NAME=$name-$step bash eval_reasoning_quick_leonardo.sbatch ;;
  tool)  MODEL=$model NAME=$name-$step bash eval_tool_calling_heldout_leonardo.sbatch ;;
  code)  MODELS="$name-$step=$model" bash eval_code_evalplus_leonardo.sbatch ;;
esac
