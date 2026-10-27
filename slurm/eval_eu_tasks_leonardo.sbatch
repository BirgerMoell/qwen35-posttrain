#!/bin/bash
#SBATCH --job-name=oellm-eu-tasks
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:50:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# EU task suites (eu_qa calibrated factual QA + eu_doc document tasks), default and thinking mode, one GPU, then score.
# One GPU and < 2 h so the job backfills on a full partition.
#   MODEL=<dir> NAME=<name> [MODES="default thinking"] sbatch slurm/eval_eu_tasks_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
: "${MODEL:?}" "${NAME:?}"
DATA=${DATA:-$root/data/eu-tasks/eval.jsonl}
out=$root/eval/eu-tasks/$NAME
mkdir -p "$out"
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 "$VLLM_PY" "$root/eval_eu_tasks.py" gen \
  --model "$MODEL" --data "$DATA" --out "$out" --modes ${MODES:-default thinking}
singularity exec -B /leonardo_work,/leonardo_scratch "$C" "$VLLM_PY" "$root/eval_eu_tasks.py" score --out "$out"
