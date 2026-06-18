#!/bin/bash
#SBATCH --job-name=sft-2b-smoke
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g           # dev-g (30min) for smoke; standard-g for real runs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-00:30:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# SFT smoke on LUMI (AMD/ROCm) via TRL. SKELETON — see docs/lumi_rocm_runbook.md.
# Validate-tiny-first: prove TRL SFT runs on MI250X before scaling to 9B / real data.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail
module purge

# TODO: point at the post-training ROCm container (extend laif-rocm with trl/vllm, or a built image)
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=2

CFG=configs/sft_2b_smoke.yaml

# TODO: replace with the actual TRL entrypoint once the repo's stages/sft launcher is wired.
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  cd $SLURM_SUBMIT_DIR &&
  accelerate launch --config_file configs/accelerate_fsdp.yaml \
    -m trl.scripts.sft --config $CFG
"
echo "DONE $(date)"
