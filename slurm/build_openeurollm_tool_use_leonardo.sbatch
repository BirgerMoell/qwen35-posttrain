#!/bin/bash
#SBATCH --job-name=oellm-tooluse-ds
#SBATCH --account=aifac_f02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err
# OpenEuroLLM Tool-Use v1: canonical tool-use dataset + pre-tokenized pool (template v2.1).
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
scratch=/leonardo_scratch/fast/AIFAC_F02_352/pmoell00
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
rm -rf "$root/data/openeurollm-tool-use-v1" "$root/data/stage3-tool-pool-v1" "$scratch/tooluse-work"
srun --unbuffered singularity exec -B /leonardo_work,/leonardo_scratch /leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif \
  python "$root/build_openeurollm_tool_use.py" --staging "$scratch/tooluse-staging" --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_two_stage_v2_1.jinja" --dataset-dir "$root/data/openeurollm-tool-use-v1" \
  --pool-dir "$root/data/stage3-tool-pool-v1" --work-dir "$scratch/tooluse-work" --workers 8
