#!/bin/bash
#SBATCH --job-name=oellm-sft-identity
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
export HF_HOME="$root/hf-cache" HF_HUB_DOWNLOAD_TIMEOUT=120 OMP_NUM_THREADS=4
test -s "$root/tokenizer/tokenizer.json"
test -s "$root/oellm_qwen3_no_empty_assistant_mask.jinja"
srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/stage_oellm_model_identity.py" \
  --source-dir "$root/source-datasets/model-identity-v1" \
  --output-dir "$root/data/model-identity-v1" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --max-length 4096
