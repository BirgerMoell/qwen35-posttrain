#!/bin/bash
#SBATCH --job-name=glm-pp-tiny
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=0-00:20:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# Fast dev of the pipeline-generation mechanics on a TINY random glm_moe_dsa: 2 ranks (stages) on
# 1 node, 1 GCD each. Instant load -> cheap iteration before the ~26-min real-model runs.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
SCRATCH=/scratch/project_465002530/users/bmoell
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=3466
srun --label --gpu-bind=none singularity exec -B "$BIND" "$CONTAINER" bash -lc '
  # both tasks see both GCDs; each picks its own by LOCAL_RANK (set_device(LOCAL) in the script)
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
  export PYTHONPATH='"$SCRATCH"'/glm-numpy-fix:'"$SCRATCH"'/pylibs-overlay:${PYTHONPATH:-}
  export HF_HOME='"$SCRATCH"'/hf_cache HF_HUB_OFFLINE=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  export MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS GLM_TINY=1
  export GLM_MODEL='"$SCRATCH"'/models/GLM-5.2-FP8 MAX_NEW=16
  cd '"$SLURM_SUBMIT_DIR"' && python scripts/glm_pp_generate.py
'
echo "tiny PP done $(date)"
