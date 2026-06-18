#!/bin/bash
# Stage post-training datasets to LUMI scratch (run on a LOGIN node — compute nodes have no internet).
# Pulls the prioritized sets from data/SOURCES.md. Idempotent / resumable (hf download caches).
#
# Usage:  bash scripts/stage_data_lumi.sh [P1|P2|all]      (default: P1)
set -euo pipefail

TIER="${1:-P1}"
DEST="${POSTTRAIN_DATA_DIR:-/scratch/project_465002530/users/bmoell/posttrain-data}"
export HF_HOME="${HF_HOME:-/scratch/project_465002530/users/bmoell/hf_cache}"
mkdir -p "$DEST" "$HF_HOME"
command -v hf >/dev/null || { echo "hf CLI not found — activate an env with huggingface_hub"; exit 1; }

# repo_id                                              -> tier
P1=(
  openeurollm/dolci-instruct-sft-tokenized
  openeurollm/dolci-think-sft-tokenized
  openeurollm/Dolci-Instruct-DPO-translated
  allenai/RLVR-GSM-MATH-IF-Mixed-Constraints
  allenai/Dolci-RL-Zero-Mix-7B
)
P2=(
  allenai/Dolci-Instruct-SFT-Tool-Use
  allenai/Dolci-Think-SFT-Python
  allenai/Dolci-Think-DPO-7B
  allenai/rlvr-code-data-python-r1-format-filtered
  nvidia/ChatQA2-Long-SFT-data
)

dl () {
  local repo="$1"
  echo "==> $repo"
  hf download "$repo" --repo-type dataset --local-dir "$DEST/${repo//\//__}" || {
    echo "   WARN: failed $repo (gated? check access on huggingface.co/datasets/$repo)"; }
}

case "$TIER" in
  P1)  for r in "${P1[@]}"; do dl "$r"; done ;;
  P2)  for r in "${P2[@]}"; do dl "$r"; done ;;
  all) for r in "${P1[@]}" "${P2[@]}"; do dl "$r"; done ;;
  *)   echo "usage: $0 [P1|P2|all]"; exit 1 ;;
esac

echo "Done. Data under: $DEST"
echo "Note: project-native sources (grpo-data-bootstrap, openeurollm-language-dataset-candidates,"
echo "oellm-longctx-tokenized-streamed-all-v2) — see data/SOURCES.md; clone/pull those too."
