#!/bin/bash
#SBATCH --job-name=pt-grpo
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-00:30:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# SFT stage (TRL). Reads STAGE_LOAD / STAGE_SAVE / CFG from the orchestrator's --export.
# SKELETON — fill the TRL entrypoint. See docs/lumi_rocm_runbook.md.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
ln -sf "${SLURM_JOB_NAME}-${SLURM_JOBID}.out" logs/latest.out 2>/dev/null || true
: "${STAGE_LOAD:?set by orchestrator}"; : "${STAGE_SAVE:?}"; : "${CFG:=configs/grpo_2b_smoke.yaml}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=29503
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB OMP_NUM_THREADS=2
echo "GRPO (needs vLLM rollouts on ROCm; veRL for scale)  load=$STAGE_LOAD  save=$STAGE_SAVE  cfg=$CFG"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  cd $SLURM_SUBMIT_DIR &&
  accelerate launch --config_file configs/accelerate_fsdp.yaml -m trl.scripts.grpo \
    --config $CFG --model_name_or_path '$STAGE_LOAD' --output_dir '$STAGE_SAVE' --save_only_model true
"   # TODO: confirm trl entrypoint/flags against the project's BirgerMoell/trl fork
echo "GRPO (needs vLLM rollouts on ROCm; veRL for scale) DONE $(date)"
