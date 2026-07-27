#!/bin/bash
#SBATCH --job-name=pt-evalbatch
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=0-05:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# Runs scripts/run_all_evals.sh inside the container: tool-use (BFCL-style) + Swedish rubric
# generations + EU holdouts. transformers device_map=auto (no vLLM). Offline (uses HF cache).
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${DRIVER:=scripts/run_all_evals.sh}"   # override to scripts/run_tooluse_evals.sh for tool-only
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
PYLIBS_OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$PYLIBS_OVERLAY:\${PYTHONPATH:-} &&
  export HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 &&
  cd $SLURM_SUBMIT_DIR &&
  bash $DRIVER
"
echo "EVAL BATCH DONE $(date)"
