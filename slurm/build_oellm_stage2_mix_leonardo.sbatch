#!/bin/bash
#SBATCH --job-name=oellm-stage2-mix
#SBATCH --account=aifac_f02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err
# Stage-2 mixture: budgeted reasoning pool + cleaned direct-answer replay, pre-tokenized (v2 template).
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
rm -rf "$root/data/stage2-mix-v1"
srun --unbuffered singularity exec -B /leonardo_work:/leonardo_work /leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif \
  python "$root/build_oellm_stage2_mix.py" --pool-dir "$root/data/stage2-reasoning-pool-v1" \
  --instruct-dir "$root/data/openeurollm-instruct-sft-v1" --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_two_stage_v2.jinja" --output-dir "$root/data/stage2-mix-v1" \
  --replay-fraction 0.19 --identity-exposures 2 --max-length 32768 --workers 8
