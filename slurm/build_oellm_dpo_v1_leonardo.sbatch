#!/bin/bash
#SBATCH --job-name=oellm-dpo-v1-build
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Stage-4 DPO data: stage the threeway mix from the pinned, prefetched HF cache (offline), then pre-tokenize
# with our tokenizer + template v2.1. Run `stage_oellm_dpo_v1.py --prefetch-only` on a login node first.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
D=$OELLM_SCRATCH/oellm-dpo
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
run() { singularity exec -B /leonardo_work,/leonardo_scratch "$C" env HF_HOME=$D/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python "$@"; }
rm -rf $D/staged-v1 && run $root/stage_oellm_dpo_v1.py --out $D/staged-v1
run $root/build_oellm_dpo_v1_pretok.py --staged $D/staged-v1 --tokenizer $root/output/stage3-tooluse-v1.1-bf16 \
  --template $root/oellm_qwen3_two_stage_v2_1.jinja --out $D/pretok-v1 --workers 4
