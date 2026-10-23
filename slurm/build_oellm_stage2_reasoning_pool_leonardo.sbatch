#!/bin/bash
#SBATCH --job-name=oellm-stage2-pool
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Pre-tokenized Stage-2 reasoning candidate pool (v2 template, <=32K). Resumable.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

srun --unbuffered singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/build_oellm_stage2_reasoning_pool.py" \
  --excellent-dir "$root/excellent-sft-dataset" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_two_stage_v2.jinja" \
  --output-dir "$root/data/stage2-reasoning-pool-v1" \
  --max-length 32768 --workers 8
