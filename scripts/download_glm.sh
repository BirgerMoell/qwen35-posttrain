#!/bin/bash
# Download GLM-5.2 (or any GLM-MoE) weights to scratch. RUN ON A LUMI LOGIN NODE — compute nodes
# have no internet. Parameterized so you set the exact HF repo (the GLM-5.2 repo id / a quantized
# mirror) and target dir.
#
#   GLM_REPO   HF repo id           (default: zai-org/GLM-5.2 — adjust to the real repo)
#   GLM_DIR    local target dir     (default: $SCRATCH/models/GLM-5.2)
#   HF_TOKEN   set if the repo is gated
#
# Usage:  GLM_REPO=zai-org/GLM-5.2 bash scripts/download_glm.sh
set -euo pipefail
SCRATCH=/scratch/project_465002530/users/bmoell
: "${GLM_REPO:=zai-org/GLM-5.2}"
: "${GLM_DIR:=$SCRATCH/models/$(basename "$GLM_REPO")}"
C=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif

echo "Downloading $GLM_REPO -> $GLM_DIR"
mkdir -p "$GLM_DIR"
singularity exec -B /pfs,/scratch,/appl,/opt/cray "$C" bash -lc "
  export HF_HOME=$SCRATCH/hf_cache ${HF_TOKEN:+HF_TOKEN=$HF_TOKEN} &&
  python - <<PY
from huggingface_hub import snapshot_download
# skip nothing — GLM MoE needs config, tokenizer, and all safetensors shards
p = snapshot_download('$GLM_REPO', local_dir='$GLM_DIR',
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py'])
print('downloaded ->', p)
PY
"
echo "Done. Model at $GLM_DIR"
du -sh "$GLM_DIR"
