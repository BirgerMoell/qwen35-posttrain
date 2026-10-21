#!/bin/bash
#SBATCH --job-name=pt-simpo
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-01:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# SFT stage (TRL). Reads STAGE_LOAD / STAGE_SAVE / CFG from the orchestrator's --export.
# SKELETON — fill the TRL entrypoint. See docs/lumi_rocm_runbook.md.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
ln -sf "${SLURM_JOB_NAME}-${SLURM_JOBID}.out" logs/latest.out 2>/dev/null || true
: "${STAGE_LOAD:?set by orchestrator}"; : "${STAGE_SAVE:?}"; : "${CFG:=configs/dpo_2b_smoke.yaml}"
: "${RANK0_META_LOAD:=0}"   # 1 = only global rank 0 reads weights; other ranks initialize on meta
: "${WRAP_PREFERENCE_STRINGS:=0}"  # 1 = turn bare prompt/chosen/rejected strings into chat messages
: "${SIMPO_PREPROCESS_ONLY:=0}"  # 1 = validate distributed preprocessing, then exit before model load
: "${SIMPO_SKIP_FINAL_SAVE:=0}"  # 1 = run training canary without writing a full model artifact
: "${SIMPO_MAX_STEPS:=}"  # optional numeric CLI override for a bounded training canary
if [ -n "$SIMPO_MAX_STEPS" ] && [[ ! "$SIMPO_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SIMPO_MAX_STEPS must be a positive integer" >&2
  exit 2
fi
SIMPO_MAX_STEPS_ARG=""
if [ -n "$SIMPO_MAX_STEPS" ]; then
  SIMPO_MAX_STEPS_ARG="--max_steps $SIMPO_MAX_STEPS"
fi
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB OMP_NUM_THREADS=7
echo "SimPO  load=$STAGE_LOAD  save=$STAGE_SAVE  cfg=$CFG  port=$MASTER_PORT"
# ntasks-per-node=1: accelerate launches once and spawns 8 GPU workers internally
# PYTHONPATH overlay: transformers 5.x for Gemma 4 / Qwen 3.5 (container ships 4.57.3)
PYLIBS_OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
# HF caches MUST live on scratch — default ~/.cache fills the small LUMI home quota.
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
# Isolate each run from incomplete Arrow shards left by an interrupted or failed dataset map.
: "${SIMPO_DATASET_CACHE:=$HF_CACHE/simpo-datasets/$SLURM_JOB_ID}"
mkdir -p "$SIMPO_DATASET_CACHE"
# Multi-node aware (see sft_lumi.sbatch): same file runs on 1 node or `sbatch --nodes=N`.
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
NUM_PROC=$(( GPUS_PER_NODE * SLURM_NNODES ))
echo "Launching SimPO on $SLURM_NNODES node(s), $NUM_PROC GPU workers total"
echo "Dataset cache=$SIMPO_DATASET_CACHE  preprocess_only=$SIMPO_PREPROCESS_ONLY  max_steps=${SIMPO_MAX_STEPS:-config}"
echo "skip_final_save=$SIMPO_SKIP_FINAL_SAVE"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$PYLIBS_OVERLAY:\${PYTHONPATH:-} &&
  export HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$SIMPO_DATASET_CACHE &&
  export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1 RANK0_META_LOAD=$RANK0_META_LOAD &&
  export WRAP_PREFERENCE_STRINGS=$WRAP_PREFERENCE_STRINGS SIMPO_PREPROCESS_ONLY=$SIMPO_PREPROCESS_ONLY &&
  export SIMPO_SKIP_FINAL_SAVE=$SIMPO_SKIP_FINAL_SAVE &&
  cd $SLURM_SUBMIT_DIR &&
  accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_PROC \
    --machine_rank \$SLURM_PROCID \
    scripts/simpo_train.py \
    --config $CFG --model_name_or_path '$STAGE_LOAD' --output_dir '$STAGE_SAVE' --save_only_model true $SIMPO_MAX_STEPS_ARG
"
echo "SimPO DONE $(date)"
