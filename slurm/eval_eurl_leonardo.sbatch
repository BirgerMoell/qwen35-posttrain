#!/bin/bash
#SBATCH --job-name=oellm-eurl
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:50:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# L1 baselines for european-rl typed environments (scripts/eval_eurl.py), default and thinking mode, one GPU.
#   MODEL=<dir> NAME=<name> DATA=<exported rows.jsonl> [MODES="default thinking"] sbatch slurm/eval_eurl_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
: "${MODEL:?}" "${NAME:?}" "${DATA:?}"
EURL=${OELLM_EUROPEAN_RL:-$OELLM_SCRATCH/oellm-rl/european-rl}
out=$root/eval/eurl/$NAME
mkdir -p "$out"
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 OELLM_EUROPEAN_RL="$EURL" \
  "$VLLM_PY" "$root/eval_eurl.py" gen --model "$MODEL" --data "$DATA" --out "$out" --modes ${MODES:-default thinking}
singularity exec -B /leonardo_work,/leonardo_scratch "$C" env OELLM_EUROPEAN_RL="$EURL" \
  "$VLLM_PY" "$root/eval_eurl.py" score --out "$out"
