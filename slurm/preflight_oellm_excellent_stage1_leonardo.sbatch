#!/bin/bash
#SBATCH --job-name=oellm-stage1-gate
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
data="$root/data/excellent-stage1-full-v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
test -s "$data/manifest.json"

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/verify_oellm_stage1_manifest.py" \
  --manifest "$data/manifest.json" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --tokenizer "$root/tokenizer" \
  --verify-file-hashes

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/preflight_trl_two_stage.py" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --instruct-data "$data/warmup-00000.jsonl" \
  --reasoning-data "$root/data/two-stage-pilot-v1/reasoning_candidates.jsonl"

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/preflight_oellm_stage1_loader.py" \
  --train-glob "$data/warmup-*.jsonl" \
  --eval-glob "$data/validation-*.jsonl" \
  --eval-max-rows 128
