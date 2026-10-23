#!/bin/bash
#SBATCH --job-name=oellm-sft-more-data
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
mkdir -p "$root/hf-cache" "$root/source-datasets"
export HF_HOME="$root/hf-cache"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export OMP_NUM_THREADS=4

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/stage_oellm_instruct_expansion.py" \
  --output-root "$root/source-datasets"
