#!/bin/bash
#SBATCH --job-name=oellm9b-sft-hf
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

REPO=/scratch/project_465002530/users/bmoell/qwen35-posttrain
: "${SRC:=$REPO/output/oellm9b-256k-sft}"
: "${DST:=$REPO/output/oellm9b-256k-sft-bf16}"
: "${CARD:=$REPO/model_cards/oellm9b-256k-sft-README.md}"
: "${REPORT:=$REPO/eval/results/oellm9b_256k_sft_inference_smoke.json}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

cd "$REPO"
echo "Preparing BF16 HF export: $SRC -> $DST"
srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  python3 scripts/prepare_oellm_hf_export.py --src '$SRC' --dst '$DST' --card '$CARD'
"

echo "Running GPU inference validation"
srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  python3 scripts/validate_hf_inference.py --model '$DST' --out '$REPORT'
"
echo "EXPORT + INFERENCE VALIDATION DONE $(date)"
