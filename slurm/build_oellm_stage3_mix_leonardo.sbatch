#!/bin/bash
#SBATCH --job-name=oellm-stage3-mix
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Stage-3 tool-use mix: tool pool + Stage-2 replay, pre-tokenized (template v2.1).
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
rm -rf "$root/data/stage3-mix-v1"
srun singularity exec -B /leonardo_work /leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif \
  python "$root/build_oellm_stage3_mix.py" --tool-pool "$root/data/stage3-tool-pool-v1" --stage2-mix "$root/data/stage2-mix-v2" \
  --template "$root/oellm_qwen3_two_stage_v2_1.jinja" --tokenizer "$root/tokenizer" --output-dir "$root/data/stage3-mix-v1"
