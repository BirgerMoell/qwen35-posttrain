#!/bin/bash
#SBATCH --job-name=recast-bf16
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-01:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Recast an fp32 checkpoint to bf16 on a compute node (login nodes are memory-limited and
# can't load the 36 GB model). Streams the safetensors tensor-by-tensor (low peak memory),
# then copies config/tokenizer/chat-template into a sibling *-bf16 dir ready to upload.
#   SRC : checkpoint dir (default output/qwen35-9b-sft)
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${SRC:=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-sft}"
DST="${SRC}-bf16"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
echo "Recast $SRC -> $DST"
srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} &&
  python3 - '$SRC' '$DST' <<'PYEOF'
import sys, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
new = {}
with safe_open(os.path.join(src, 'model.safetensors'), framework='pt', device='cpu') as f:
    for k in f.keys():
        new[k] = f.get_tensor(k).to(torch.bfloat16)
save_file(new, os.path.join(dst, 'model.safetensors'), metadata={'format': 'pt'})
for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
             'special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt',
             'preprocessor_config.json'):
    p = os.path.join(src, name)
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(dst, name))
print('bf16 size GB:', round(os.path.getsize(os.path.join(dst,'model.safetensors'))/1e9, 1))
print('DONE ->', dst)
PYEOF
"
echo "RECAST DONE $(date) -> $DST"
