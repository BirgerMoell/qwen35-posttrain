#!/bin/bash
#SBATCH --job-name=oellm-format-pairs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Format-compliance pairs, generate phase: Stage-3 v1.1 greedy default-mode replies to the selected strict-format prompts
# (scripts/build_oellm_format_pairs.py select must have written prompts.jsonl), then pre-tokenize the kept pairs with
# the same builder and guards as the main DPO data.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
out=$OELLM_SCRATCH/oellm-dpo/format-v1
model=$root/output/stage3-tooluse-v1.1-bf16
template=$root/oellm_qwen3_two_stage_v2_1.jinja
test -s "$out/prompts.jsonl"
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 "$VLLM_PY" \
  "$root/build_oellm_format_pairs.py" generate --out "$out" --model "$model" --template "$template"
mkdir -p "$out/staged"
mv "$out"/format-train.jsonl.gz "$out"/format-validation.jsonl.gz "$out/staged/"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python "$root/build_oellm_dpo_v1_pretok.py" \
  --staged "$out/staged" --tokenizer "$model" --template "$template" --out "$out/pretok" --workers 8
