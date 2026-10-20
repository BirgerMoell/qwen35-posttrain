#!/bin/bash
#SBATCH --job-name=oellm-stage1-post
#SBATCH --account=aifac_f02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err

# After Stage-1 training: FP32 export -> BF16 (with the v2 template), then generation check.
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
src="$root/output/stage1-instruct-v1"
dst="$root/output/stage1-instruct-v1-bf16"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

srun singularity exec --nv -B /leonardo_work:/leonardo_work "$container" python - "$src" "$dst" "$root/oellm_qwen3_two_stage_v2.jinja" <<'PY'
import sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
src, dst, template = sys.argv[1:]
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
assert next(model.parameters()).dtype == torch.bfloat16
model.save_pretrained(dst, max_shard_size="5GB")
tok = AutoTokenizer.from_pretrained(src)
tok.chat_template = Path(template).read_text(encoding="utf-8")
tok.save_pretrained(dst)
print("bf16 export ->", dst)
PY
srun singularity exec --nv -B /leonardo_work:/leonardo_work "$container" \
  python "$root/eval_stage1_generation_check.py" --model "$dst" --out "$root/output/stage1-instruct-v1-generation-check.json"
