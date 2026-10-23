#!/bin/bash
#SBATCH --job-name=oellm-sft-pilot-data
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
test -s "$root/aya-dataset/data/train-00000-of-00001.parquet"
test -s "$root/tokenizer/tokenizer.json"
test -s "$root/oellm_qwen3_no_empty_assistant_mask.jinja"
srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/build_oellm_two_stage_pilot.py" \
  --excellent-dir "$root/excellent-sft-dataset" \
  --aya-parquet "$root/aya-dataset/data/train-00000-of-00001.parquet" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --output-dir "$root/data/two-stage-pilot-v1" \
  --max-length 4096
