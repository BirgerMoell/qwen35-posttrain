#!/bin/bash
#SBATCH --job-name=glm-pp
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-01:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# Pure-transformers pipeline-parallel GLM-5.2 — one pipeline STAGE per node (1 task/node, 8 GCDs).
#   SCRIPT   python to run (default the load test)
#   GLM_MODEL model dir (default GLM-5.2-FP8)
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
SCRATCH=/scratch/project_465002530/users/bmoell
: "${GLM_MODEL:=$SCRATCH/models/GLM-5.2-FP8}"; : "${SCRIPT:=scripts/glm_pp_load_test.py}"; : "${MAX_NEW:=8}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=3455
echo "GLM PP: model=$GLM_MODEL stages=$SLURM_NNODES script=$SCRIPT"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc '
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES; unset ROCR_VISIBLE_DEVICES; fi
  export PYTHONPATH='"$SCRATCH"'/glm-numpy-fix:'"$SCRATCH"'/pylibs-overlay:${PYTHONPATH:-}
  export HF_HOME='"$SCRATCH"'/hf_cache HF_HUB_OFFLINE=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  export MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'
  export RANK=$SLURM_NODEID WORLD_SIZE=$SLURM_NNODES GLM_MODEL='"$GLM_MODEL"' MAX_NEW='"$MAX_NEW"'
  cd '"$SLURM_SUBMIT_DIR"' && python '"$SCRIPT"'
'
echo "GLM PP done $(date)"
