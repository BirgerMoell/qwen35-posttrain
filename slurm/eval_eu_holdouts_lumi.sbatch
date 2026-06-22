#!/bin/bash
#SBATCH --job-name=pt-eu-holdouts
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-04:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Evaluate a checkpoint on the EU eval holdouts (38 langs x 10 buckets, deterministic scoring).
#   MODEL : checkpoint to evaluate
#   TAG   : results label (eval/results/eu-holdouts/$TAG.json)
#   SPLIT : dev (1520) or test_public (6080); default dev
#   LIMIT : optional --limit-per-bucket for a quick read
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${MODEL:?}"; : "${TAG:=ckpt}"; : "${SPLIT:=dev}"; : "${LIMIT:=}"
: "${EXCLUDE:=long_context_retrieval}"   # skip long-ctx by default (slow + irrelevant for FT)
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
DATA=$SLURM_SUBMIT_DIR/data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/${SPLIT}.jsonl
OUT=$SLURM_SUBMIT_DIR/eval/results/eu-holdouts/${TAG}.json
mkdir -p "$(dirname "$OUT")"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIM=""; [ -n "$LIMIT" ] && LIM="--limit-per-bucket $LIMIT"

echo "EU-HOLDOUTS  model=$MODEL  tag=$TAG  split=$SPLIT  limit=${LIMIT:-all}"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  cd $SLURM_SUBMIT_DIR &&
  python scripts/eval_eu_holdouts.py --model '$MODEL' --data '$DATA' --out '$OUT' $LIM --exclude-buckets '$EXCLUDE'
"
echo "EU-HOLDOUTS $TAG DONE $(date) -> $OUT"
