#!/bin/bash
#SBATCH --job-name=oellm-stage4-post
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Export one Stage-4 DPO checkpoint (FP32 model.safetensors written by the trainer) to BF16 with the v2.1 template and
# the Stage-3 v1.1 tokenizer, then the generation check in both modes. Queue the held-out tool eval on the export as
# well: DPO's pairs are tool-free and default-mode, so tool-call closers (tag tokens 11-18) and thinking are the risks.
#   SRC=<checkpoint dir> DST=<export dir> sbatch slurm/post_oellm_stage4_dpo_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
src=${SRC:?checkpoint dir}
dst=${DST:?export dir}
parent="$root/output/stage3-tooluse-v1.1-bf16"
template="$root/oellm_qwen3_two_stage_v2_1.jinja"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run() { srun singularity exec --nv -B /leonardo_work,/leonardo_scratch "$container" "$@"; }

if [[ ! -s "$dst/model.safetensors.index.json" ]]; then
  run python - "$src" "$dst" "$parent" "$template" <<'PY'
import sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
src, dst, parent, template = sys.argv[1:]
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
assert next(model.parameters()).dtype == torch.bfloat16
model.save_pretrained(dst, max_shard_size="5GB")
tok = AutoTokenizer.from_pretrained(parent)  # the checkpoint carries no tokenizer; DPO never changes it
tok.chat_template = Path(template).read_text(encoding="utf-8")
tok.save_pretrained(dst)
print("bf16 export ->", dst)
PY
fi
status=0
run python "$root/eval_stage1_generation_check.py" --model "$dst" --out "$dst-generation-check-default.json" || status=1
run python "$root/eval_stage1_generation_check.py" --model "$dst" --enable-thinking --max-new-tokens 4096 \
  --out "$dst-generation-check-thinking.json" || status=1
exit $status
