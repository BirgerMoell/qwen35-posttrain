#!/bin/bash
#SBATCH --job-name=oellm-instruct-ds
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# OpenEuroLLM Instruct SFT dataset (messages format): Stage-1 selection, identity-claim cleaning, identity v2.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
rm -rf "$root/data/openeurollm-instruct-sft-v1"
srun singularity exec -B /leonardo_work:/leonardo_work /leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif \
  python "$root/build_openeurollm_instruct_sft.py" --root "$root" --identity-v2 "$root/resources/identity_v2_core.json" \
  --output-dir "$root/data/openeurollm-instruct-sft-v1" --workers 8
