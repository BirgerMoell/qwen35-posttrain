#!/bin/bash
#SBATCH --job-name=oellm9b-sft-niah
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Resumable 8K -> 256K needle-in-a-haystack grid for the exact uploaded model.
# `balanced` distributes transformer layers (and therefore KV cache) over all
# eight GCDs. The evaluator persists the report after every grid cell.

if [ -z ${SLURM_JOB_ID+x} ]; then
  mkdir -p logs
  sbatch "$0" "$@"
  exit
fi

set -euo pipefail
module purge

: "${MODEL:=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft-bf16}"
: "${TAG:=oellm9b-256k-sft}"
: "${LENGTHS:=8192,32768,131072,262144}"
: "${DEPTHS:=0,25,50,75,100}"
: "${MAXNEW:=64}"
: "${SECRET_STYLE:=hex}"
: "${ALLOW_NONSTANDARD_THETA:=0}"
: "${PROMPT_FORMAT:=chat}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
: "${REPO:=/scratch/project_465002530/users/bmoell/qwen35-posttrain}"
OUT=$REPO/eval/results/niah/${TAG}.json
THETA_FLAG=""
if [ "$ALLOW_NONSTANDARD_THETA" = "1" ]; then
  THETA_FLAG="--allow-nonstandard-theta"
fi

mkdir -p "$(dirname "$OUT")"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "OELLM NIAH model=$MODEL lengths=$LENGTHS depths=$DEPTHS max_new_tokens=$MAXNEW secret_style=$SECRET_STYLE prompt_format=$PROMPT_FORMAT started=$(date --iso-8601=seconds)"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  cd $REPO &&
  python scripts/eval_niah.py \
    --model '$MODEL' \
    --out '$OUT' \
    --lengths '$LENGTHS' \
    --depths '$DEPTHS' \
    --max-new-tokens $MAXNEW \
    --secret-style '$SECRET_STYLE' \
    --prompt-format '$PROMPT_FORMAT' \
    --attn-implementation flash_attention_2 \
    --device-map balanced \
    $THETA_FLAG \
    --resume
"
echo "OELLM NIAH DONE $(date --iso-8601=seconds) out=$OUT"
