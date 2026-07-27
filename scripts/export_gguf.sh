#!/bin/bash
# Convert an HF model dir to GGUF and quantize for on-device (llama.cpp / Ollama / LM Studio).
# Qwen3.5 arch is supported by llama.cpp's converter AND runtime (verified on Qwen3.5-0.8B).
# The native chat template (with the thinking toggle) is embedded in the GGUF — no-think is the
# built-in default; clients enable reasoning via the /think path.
#
# Usage: export_gguf.sh <hf_model_dir> <out_prefix> [quants...]
#   hf_model_dir : local HF checkpoint (config.json + safetensors + tokenizer)
#   out_prefix   : e.g. /scratch/.../gguf-out/qwen35-4b-eu  -> writes <prefix>-<QUANT>.gguf
#   quants       : default "Q4_K_M Q5_K_M Q8_0"
#
# Env:
#   LLAMACPP : llama.cpp checkout with a built ./build (default /scratch/.../llama.cpp)
#   PYLIBS   : ':'-joined overlays giving transformers+safetensors+numpy and gguf
set -euo pipefail
MODEL=${1:?hf_model_dir}; PREFIX=${2:?out_prefix}; shift 2 || true
QUANTS=${*:-Q4_K_M Q5_K_M Q8_0}
B=/scratch/project_465002530/users/bmoell
LLAMACPP=${LLAMACPP:-$B/llama.cpp}
PYLIBS=${PYLIBS:-$B/pylibs-overlay:$B/pylibs-llamacpp}

mkdir -p "$(dirname "$PREFIX")"
F16="$PREFIX-f16.gguf"
echo "[gguf] convert $MODEL -> $F16"
PYTHONPATH="$PYLIBS:${PYTHONPATH:-}" python3 "$LLAMACPP/convert_hf_to_gguf.py" \
  "$MODEL" --outfile "$F16" --outtype f16

# Prefer the in-container build (binaries linked against container libs, run anywhere the
# container does); fall back to a host build.
QBIN="$LLAMACPP/build-container/bin/llama-quantize"
[ -x "$QBIN" ] || QBIN="$LLAMACPP/build/bin/llama-quantize"
for Q in $QUANTS; do
  OUT="$PREFIX-${Q}.gguf"
  echo "[gguf] quantize $Q -> $OUT"
  "$QBIN" "$F16" "$OUT" "$Q"
done
echo "[gguf] done:"; ls -la "$PREFIX"-*.gguf
