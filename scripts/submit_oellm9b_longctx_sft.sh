#!/bin/bash
# Submit the gated OELLM-9B 64K long-context SFT repair on LUMI.
#
# Usage:
#   bash scripts/submit_oellm9b_longctx_sft.sh smoke
#   bash scripts/submit_oellm9b_longctx_sft.sh production
set -euo pipefail

MODE="${1:-smoke}"
REPO="${REPO:-/scratch/project_465002530/users/bmoell/qwen35-posttrain}"
CURRENT_SFT="${CURRENT_SFT:-$REPO/output/oellm9b-256k-sft}"
TEMPLATE="${CHAT_TEMPLATE_FILE:-$REPO/templates/oellm_gemma_assistant_mask.jinja}"
ROW_MIX="${DATASET_MIX_PROBABILITIES:-0.90:0.10}"

case "$MODE" in
  smoke)
    CFG="${CFG:-configs/sft_oellm9b_256k_longctx_repair_smoke.yaml}"
    STAGE_SAVE="${STAGE_SAVE:-$REPO/output/oellm9b-256k-longctx-sft-smoke}"
    NODES="${NODES:-1}"
    WALLTIME="${WALLTIME:-0-02:00:00}"
    ;;
  production)
    CFG="${CFG:-configs/sft_oellm9b_256k_longctx_repair_64k.yaml}"
    STAGE_SAVE="${STAGE_SAVE:-$REPO/output/oellm9b-256k-longctx-sft-64k}"
    NODES="${NODES:-4}"
    WALLTIME="${WALLTIME:-0-20:00:00}"
    ;;
  *)
    echo "usage: $0 [smoke|production]" >&2
    exit 2
    ;;
esac

test -d "$CURRENT_SFT" || { echo "missing CURRENT_SFT: $CURRENT_SFT" >&2; exit 1; }
test -f "$TEMPLATE" || { echo "missing assistant-mask template: $TEMPLATE" >&2; exit 1; }
test -f "$REPO/$CFG" || { echo "missing config: $REPO/$CFG" >&2; exit 1; }

cd "$REPO"
sbatch --parsable \
  --nodes="$NODES" \
  --gpus-per-node=8 \
  --time="$WALLTIME" \
  --export="ALL,RANK0_META_LOAD=1,STAGE_LOAD=$CURRENT_SFT,STAGE_SAVE=$STAGE_SAVE,CFG=$CFG,CHAT_TEMPLATE_FILE=$TEMPLATE,DATASET_MIX_PROBABILITIES=$ROW_MIX,DATASET_MIX_SEED=20260817,DATASET_MIX_STOPPING_STRATEGY=all_exhausted" \
  slurm/sft_lumi.sbatch
