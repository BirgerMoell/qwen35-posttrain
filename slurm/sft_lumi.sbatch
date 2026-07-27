#!/bin/bash
#SBATCH --job-name=pt-sft
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
: "${STAGE_LOAD:?set by orchestrator}"; : "${STAGE_SAVE:?}"; : "${CFG:=configs/sft_2b_smoke.yaml}"
: "${RANK0_META_LOAD:=0}"   # 1 = only global rank 0 loads weights (others on meta) — for big models
: "${RESUME_FROM_CHECKPOINT:=0}"  # 1 = resume from latest checkpoint-* in STAGE_SAVE (or a path).
# NOTE: needs a FULL FSDP checkpoint (pytorch_model_fsdp.bin + optimizer). Our runs use
# --save_only_model true (consolidated model.safetensors only), which CANNOT optimizer-resume —
# FileNotFoundError on pytorch_model_fsdp.bin. To continue such a run, WARM-START instead: point
# STAGE_LOAD at the checkpoint dir with a low-constant-LR finish config (see *_v3_finish.yaml).
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB OMP_NUM_THREADS=7
# Reduce GPU memory fragmentation (ROCm honours both names). Recovers reserved-but-unallocated VRAM.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "SFT  load=$STAGE_LOAD  save=$STAGE_SAVE  cfg=$CFG  port=$MASTER_PORT"
# ntasks-per-node=1: accelerate launch is run ONCE per node and spawns 8 GPU workers internally
# PYTHONPATH overlay: transformers 5.x for Gemma 4 / Qwen 3.5 (container ships 4.57.3)
PYLIBS_OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
# HF caches MUST live on scratch — the default ~/.cache fills the small LUMI home quota
# ("OSError: Not enough disk space" during dataset tokenize/pack).
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
# Multi-node aware: derive world size from the allocation. Same file works on 1 node (default)
# or N nodes via `sbatch --nodes=N`. srun runs one task/node (ntasks-per-node=1) so each node's
# task carries SLURM_PROCID = its machine_rank; accelerate spawns 8 GPU workers per node.
# Auto-detect GPUs/node from the allocation (was hardcoded 8). Lets us run FEWER ranks/node
# for big models: transformers 5.x dropped the env-based FSDP cpu_ram_efficient (rank-0-only)
# loading, so EVERY rank materializes a full CPU copy. A 27B is 54 GB/rank → 8/node OOMs the
# node RAM; submit `--gpus-per-node=4 --mem=480G` so only 4×54=216 GB loads per node.
# Honor an explicit GPUS_PER_NODE override first: SLURM_GPUS_ON_NODE reports the node's PHYSICAL
# GCD count (8) even when --gpus-per-node is smaller, which would over-spawn ranks and OOM CPU RAM
# at 27B. So for big models pass `--export=...,GPUS_PER_NODE=4 --gpus-per-node=4`.
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
NUM_PROC=$(( GPUS_PER_NODE * SLURM_NNODES ))
echo "Launching on $SLURM_NNODES node(s), $NUM_PROC GPU workers total"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$PYLIBS_OVERLAY:\${PYTHONPATH:-} &&
  export HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1 RANK0_META_LOAD=$RANK0_META_LOAD &&
  export RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT &&
  cd $SLURM_SUBMIT_DIR &&
  accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_PROC \
    --machine_rank \$SLURM_PROCID \
    scripts/sft_train.py \
    --config $CFG --model_name_or_path '$STAGE_LOAD' --output_dir '$STAGE_SAVE' --save_only_model true
"
echo "SFT DONE $(date)"
