#!/bin/bash
#SBATCH --job-name=glm-spmd
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-02:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# GLM-5.2 SPMD offline generation via vLLM external_launcher (NO Ray) — reuses the SAME srun +
# torch-distributed / RCCL setup as our working multi-node training. One task per GCD.
#   GLM_MODEL  model dir   (default GLM-5.2-FP8)
#   PROMPTS    jsonl of {prompt}   (empty = smoke test)
#   OUT        output jsonl
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
SCRATCH=/scratch/project_465002530/users/bmoell
: "${GLM_MODEL:=$SCRATCH/models/GLM-5.2-FP8}"; : "${PROMPTS:=}"; : "${OUT:=}"; : "${MAXLEN:=8192}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=$SCRATCH/pylibs-overlay
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=3442
# LUMI Slingshot RCCL env (same as training) — the networking that already works multi-node
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
echo "GLM SPMD: model=$GLM_MODEL nodes=$SLURM_NNODES tasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT"

srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc '
  # ROCm: use HIP_VISIBLE_DEVICES; Slurm binds 1 GCD/task so each task sees its own device 0
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES; unset ROCR_VISIBLE_DEVICES; fi
  export PYTHONPATH='"$SCRATCH"'/glm-numpy-fix:'"$OVERLAY"':${PYTHONPATH:-} HF_HOME='"$SCRATCH"'/hf_cache HF_HUB_OFFLINE=1
  export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
  export MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  export GLM_MODEL='"$GLM_MODEL"' PROMPTS="'"$PROMPTS"'" OUT="'"$OUT"'" MAXLEN='"$MAXLEN"'
  cd '"$SLURM_SUBMIT_DIR"' &&
  python scripts/glm_generate_spmd.py
'
echo "GLM SPMD done $(date)"
