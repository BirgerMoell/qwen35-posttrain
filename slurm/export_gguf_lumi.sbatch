#!/bin/bash
#SBATCH --job-name=export-gguf
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-01:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.out
#
# CPU-side GGUF export+quantize for a finished checkpoint. No GPU math needed (1 GPU just to land
# on a GPU partition quickly); conversion/quantization are CPU-bound.
#   MODEL  : HF checkpoint dir to export
#   PREFIX : output prefix (writes <PREFIX>-<QUANT>.gguf)
#   QUANTS : optional, default "Q4_K_M Q5_K_M Q8_0"
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${MODEL:?}"; : "${PREFIX:?}"; : "${QUANTS:=Q4_K_M Q5_K_M Q8_0}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
B=/scratch/project_465002530/users/bmoell
echo "EXPORT-GGUF model=$MODEL prefix=$PREFIX quants=$QUANTS"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export LLAMACPP=$B/llama.cpp PYLIBS=$B/pylibs-overlay:$B/pylibs-llamacpp &&
  cd $SLURM_SUBMIT_DIR && bash scripts/export_gguf.sh '$MODEL' '$PREFIX' $QUANTS
"
echo "EXPORT-GGUF DONE $(date)"
