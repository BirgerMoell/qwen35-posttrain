#!/bin/bash
#SBATCH --job-name=pt-lmeval
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-06:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Eval a checkpoint with lm-evaluation-harness. Reads MODEL / TAG / TASKS from --export.
#   MODEL  : path to the checkpoint (or base model) to evaluate
#   TAG    : label for the results dir (eval/results/$TAG)
#   TASKS  : comma-separated lm-eval tasks (default: core regression + EU multilingual)
#
# Backend: HuggingFace (transformers 5.x overlay loads qwen3_5 / gemma4). vLLM 0.15.1 in
# the container may not support the 2026 architectures, so HF is the safe default.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${MODEL:?set MODEL=/path/to/checkpoint}"; : "${TAG:=ckpt}"
: "${TASKS:=gsm8k,ifeval,arc_challenge,hellaswag,mmlu}"
: "${LIMIT:=}"   # optional: cap examples per task for a quick smoke (e.g. LIMIT=100)

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
PYLIBS_OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
RESULTS="$SLURM_SUBMIT_DIR/eval/results/$TAG"
mkdir -p "$RESULTS"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "EVAL  model=$MODEL  tag=$TAG  tasks=$TASKS  limit=${LIMIT:-all}"
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# data-parallel across 8 GCDs via accelerate; HF backend with the transformers 5.x overlay
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$PYLIBS_OVERLAY:\${PYTHONPATH:-} &&
  export HF_DATASETS_CACHE=/scratch/project_465002530/users/bmoell/hf_cache &&
  cd $SLURM_SUBMIT_DIR &&
  accelerate launch --num_processes 8 --num_machines 1 \
    --main_process_ip '$MASTER_ADDR' --main_process_port '$MASTER_PORT' \
    -m lm_eval --model hf \
    --model_args pretrained='$MODEL',dtype=bfloat16,trust_remote_code=True \
    --tasks '$TASKS' \
    --batch_size auto \
    $LIMIT_ARG \
    --output_path '$RESULTS' \
    --log_samples
"
echo "EVAL $TAG DONE $(date) -> $RESULTS"
