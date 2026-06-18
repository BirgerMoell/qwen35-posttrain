#!/bin/bash
# Orchestrate the post-training pipeline on LUMI as a Slurm dependency chain with EVAL GATES:
#
#   (extend*) -> SFT -> [eval] -> DPO -> [eval] -> GRPO -> [eval]
#
# Each stage loads the previous stage's checkpoint and saves a new one. Each eval job runs
# AFTER its stage and, if a gate metric regresses (e.g. long-context RULER, or core score),
# exits non-zero — which (via afterok) BLOCKS the next stage. So a bad stage halts the chain.
#
# *extend: skipped for Qwen3.5 (native 262K). For a custom base, add an extend stage first.
#
# Usage:  bash pipeline/run_pipeline.sh 2b      # validate the whole chain on 2B first
#         bash pipeline/run_pipeline.sh 9b      # the real artifact
set -euo pipefail

SIZE="${1:-2b}"
# TODO: stage the base model to LUMI scratch first (no internet on compute nodes).
BASE="${BASE_MODEL:-/scratch/project_465002530/users/bmoell/models/Qwen3.5-${SIZE}-Base}"
OUT="$PWD/output/$SIZE"
mkdir -p "$OUT" logs

sub () {  # sub <dep-jobid|-> <sbatch> KEY=VAL ...   -> echoes new jobid
  local dep="$1"; shift; local script="$1"; shift
  local exp="ALL"; for kv in "$@"; do exp="$exp,$kv"; done
  if [ "$dep" = "-" ]; then sbatch --parsable --export="$exp" "$script"
  else sbatch --parsable --dependency=afterok:"$dep" --export="$exp" "$script"; fi
}

echo "Base: $BASE   Out: $OUT"

# Baseline eval of the base (gate reference)
ev0=$(sub - slurm/eval_lumi.sbatch CKPT="$BASE" TAG=base)
# SFT
sft=$(sub "$ev0" slurm/sft_lumi.sbatch  STAGE_LOAD="$BASE"     STAGE_SAVE="$OUT/sft"  CFG="configs/sft_${SIZE}_smoke.yaml")
ev1=$(sub "$sft" slurm/eval_lumi.sbatch CKPT="$OUT/sft"  TAG=sft  BASELINE="$BASE")
# DPO  (gated on SFT eval)
dpo=$(sub "$ev1" slurm/dpo_lumi.sbatch  STAGE_LOAD="$OUT/sft"  STAGE_SAVE="$OUT/dpo"  CFG="configs/dpo_${SIZE}_smoke.yaml")
ev2=$(sub "$dpo" slurm/eval_lumi.sbatch CKPT="$OUT/dpo"  TAG=dpo  BASELINE="$OUT/sft")
# GRPO (gated on DPO eval)
grpo=$(sub "$ev2" slurm/grpo_lumi.sbatch STAGE_LOAD="$OUT/dpo" STAGE_SAVE="$OUT/grpo" CFG="configs/grpo_${SIZE}_smoke.yaml")
ev3=$(sub "$grpo" slurm/eval_lumi.sbatch CKPT="$OUT/grpo" TAG=grpo BASELINE="$OUT/dpo")

echo "Submitted chain (size=$SIZE):"
echo "  base-eval=$ev0  sft=$sft  eval=$ev1  dpo=$dpo  eval=$ev2  grpo=$grpo  eval=$ev3"
echo "Watch:  squeue -u \$USER   |   tail -f logs/latest.out"
