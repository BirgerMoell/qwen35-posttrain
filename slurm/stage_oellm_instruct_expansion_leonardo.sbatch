#!/bin/bash
#SBATCH --job-name=oellm-sft-more-data
#SBATCH --account=aifac_f02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err

set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
mkdir -p "$root/hf-cache" "$root/source-datasets"
export HF_HOME="$root/hf-cache"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export OMP_NUM_THREADS=4

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/stage_oellm_instruct_expansion.py" \
  --output-root "$root/source-datasets"
