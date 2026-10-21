#!/bin/bash
#SBATCH --job-name=oellm9b-sft-eval
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Evaluate the exact BF16 artifact uploaded as birgermoell/oellm-9b-256k-sft.
# The Swedish generations run first so they survive even if the larger holdout
# suite reaches the dev-g wall-time. Override MODEL or TAG at submission time.

if [ -z ${SLURM_JOB_ID+x} ]; then
  mkdir -p logs
  sbatch "$0" "$@"
  exit
fi

set -euo pipefail
module purge

: "${MODEL:=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft-bf16}"
: "${TAG:=oellm9b-256k-sft}"
: "${BATCH:=16}"
: "${MAXNEW:=192}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
DATA=$SLURM_SUBMIT_DIR/data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/dev.jsonl
SV_PROMPTS=$SLURM_SUBMIT_DIR/data/swedish_rubric_prompts.jsonl
SV_OUT=$SLURM_SUBMIT_DIR/eval/results/swedish-rubric/${TAG}.json
EU_OUT=$SLURM_SUBMIT_DIR/eval/results/eu-holdouts/${TAG}.json

mkdir -p "$(dirname "$SV_OUT")" "$(dirname "$EU_OUT")"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "OELLM GENERAL EVAL model=$MODEL tag=$TAG started=$(date --iso-8601=seconds)"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  cd $SLURM_SUBMIT_DIR &&
  python scripts/eval_swedish_rubric.py \
    --model '$MODEL' \
    --prompts '$SV_PROMPTS' \
    --out '$SV_OUT' \
    --tag 'birgermoell/oellm-9b-256k-sft' \
    --max-new-tokens 256 &&
  python scripts/eval_eu_holdouts.py \
    --model '$MODEL' \
    --data '$DATA' \
    --out '$EU_OUT' \
    --exclude-buckets long_context_retrieval \
    --batch-size $BATCH \
    --max-new-tokens $MAXNEW
"
echo "OELLM GENERAL EVAL DONE $(date --iso-8601=seconds) sv=$SV_OUT eu=$EU_OUT"
