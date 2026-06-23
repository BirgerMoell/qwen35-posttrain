#!/bin/bash
#SBATCH --job-name=euroeval-sv
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=0-06:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# EuroEval (formerly ScandEval) — Swedish capability benchmark for a checkpoint.
# Tasks: sentiment, NER (SUC 3.0), QA (multi-wiki-qa-sv), linguistic acceptability (DaLaJ),
# summarization, knowledge, common-sense. Transformers backend (vLLM lacks qwen3_5 here).
#   MODEL : checkpoint to evaluate
#   TAG   : label (results land in euroeval_cache/euroeval_benchmark_results.jsonl)
# NOTE: datasets must be pre-cached (compute nodes have no internet) — see euroeval_precache.sh.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${MODEL:?}"; : "${TAG:=ckpt}"; : "${LANGUAGES:=sv}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
RESULTS=$SLURM_SUBMIT_DIR/eval/results/euroeval/$TAG
mkdir -p "$RESULTS"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "EUROEVAL  model=$MODEL  tag=$TAG  langs=$LANGUAGES"
: "${OFFLINE:=0}"   # 1 = force offline (datasets must be pre-cached). 0 = allow HF download.
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  export HF_HUB_OFFLINE=$OFFLINE HF_DATASETS_OFFLINE=$OFFLINE &&
  cd '$RESULTS' &&
  python3 -m euroeval --model '$MODEL' --language $LANGUAGES \
    --model-impl transformers --device cuda --verbose --force
"
echo "EUROEVAL $TAG DONE $(date) -> $RESULTS"
