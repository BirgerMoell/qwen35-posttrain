#!/bin/bash
#SBATCH --job-name=oellm-excellent-stage1
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

mode=${BUILD_MODE:-full}
if [[ "$mode" == smoke ]]; then
  output="$root/data/excellent-stage1-smoke-v1"
  args=(--max-shards 1 --max-rows-per-shard 10000 --workers 1)
elif [[ "$mode" == full ]]; then
  output="$root/data/excellent-stage1-full-v1"
  args=(--workers 4)
else
  echo "BUILD_MODE must be smoke or full" >&2
  exit 2
fi

test -s "$root/tokenizer/tokenizer.json"
test -s "$root/oellm_qwen3_no_empty_assistant_mask.jinja"
test -s "$root/excellent-sft-dataset/data/train-00000-of-00062.parquet"
mkdir -p "$output"

srun --unbuffered singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/build_oellm_excellent_stage1.py" \
  --excellent-dir "$root/excellent-sft-dataset" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --output-dir "$output" \
  --max-length 4096 \
  "${args[@]}"
