#!/bin/bash
#SBATCH --job-name=pt-arena-eu
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-06:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# ArenaHard-EU win-rate: generate answers for CANDIDATE and BASELINE on the EU prompts,
# then LLM-judge head-to-head -> candidate win-rate (overall + per language).
#   CANDIDATE : checkpoint to evaluate (e.g. output/qwen35-9b-sft)
#   CAND_TAG  : label (e.g. sft)
#   BASELINE  : reference model (e.g. models/Qwen3.5-9B)
#   JUDGE     : judge model (default: BASELINE — self-judge start; swap a bigger model later)
#   LIMIT     : optional cap on #prompts (quick smoke)
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${CANDIDATE:?}"; : "${CAND_TAG:=cand}"; : "${BASELINE:?}"
: "${JUDGE:=$BASELINE}"; : "${LIMIT:=}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
PROMPTS=/scratch/project_462000963/datasets/posttraining_data/ArenaHard/eng/train.jsonl,/scratch/project_462000963/datasets/posttraining_data/ArenaHard/fin/train.jsonl
ARENA=$SLURM_SUBMIT_DIR/eval/results/arena
mkdir -p "$ARENA"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"

echo "ARENA-EU  candidate=$CANDIDATE ($CAND_TAG)  baseline=$BASELINE  judge=$JUDGE"

srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} &&
  export HF_DATASETS_CACHE=/scratch/project_465002530/users/bmoell/hf_cache &&
  cd $SLURM_SUBMIT_DIR &&
  echo '== generate candidate ==' &&
  python scripts/arena_generate.py --model '$CANDIDATE' --prompts '$PROMPTS' --out '$ARENA/$CAND_TAG.jsonl' $LIM &&
  echo '== generate baseline ==' &&
  python scripts/arena_generate.py --model '$BASELINE' --prompts '$PROMPTS' --out '$ARENA/base.jsonl' $LIM &&
  echo '== judge ==' &&
  python scripts/arena_judge.py --candidate '$ARENA/$CAND_TAG.jsonl' --baseline '$ARENA/base.jsonl' --judge '$JUDGE' --out '$ARENA/${CAND_TAG}_vs_base.json' $LIM
"
echo "ARENA-EU DONE $(date) -> $ARENA/${CAND_TAG}_vs_base.json"
