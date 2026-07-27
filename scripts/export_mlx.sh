#!/bin/bash
# Convert an HF model to MLX 4-bit for Apple silicon (iPhone/iPad/Mac).
# RUN THIS ON A MAC (Apple silicon) — MLX needs Metal; it does NOT run on LUMI.
#
# Setup (once):  pip install -U mlx-lm huggingface_hub
# Usage:         export_mlx.sh <hf_model_id_or_dir> <out_dir> [bits] [group_size]
#   hf_model_id_or_dir : HF repo id (e.g. birgermoell/Qwen3.5-4B-EU) or a local dir.
#                        If the checkpoint is on LUMI, first: `hf download <repo>` or rsync it down.
#   out_dir            : e.g. ./mlx-out/Qwen3.5-4B-EU-mlx-4bit
#   bits               : default 4
#   group_size         : default 64
#
# The native Qwen3.5 chat template (thinking toggle) is preserved; no-think is the default.
set -euo pipefail
SRC=${1:?hf_model_id_or_dir}; OUT=${2:?out_dir}; BITS=${3:-4}; GS=${4:-64}
echo "[mlx] convert $SRC -> $OUT  (${BITS}-bit, group_size=$GS)"
python3 -m mlx_lm convert --hf-path "$SRC" --mlx-path "$OUT" -q --q-bits "$BITS" --q-group-size "$GS"
echo "[mlx] done:"; du -sh "$OUT"; ls "$OUT"
echo "[mlx] smoke test:"
python3 -m mlx_lm generate --model "$OUT" --prompt "Vad är meningen med livet?" --max-tokens 64
