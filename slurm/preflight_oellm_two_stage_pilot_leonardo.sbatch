#!/bin/bash
#SBATCH --job-name=oellm-sft-pilot-audit
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
data="$root/data/two-stage-pilot-v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
test -s "$data/manifest.json"

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/audit_two_stage_sft.py" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --stage instruct --data "$data/instruct_train.jsonl" \
  --max-rows 128 --max-length 4096
srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/audit_two_stage_sft.py" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --stage reasoning --data "$data/reasoning_candidates.jsonl" \
  --max-rows 29 --max-length 4096
srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/preflight_trl_two_stage.py" \
  --tokenizer "$root/tokenizer" \
  --template "$root/oellm_qwen3_no_empty_assistant_mask.jinja" \
  --instruct-data "$data/instruct_train.jsonl" \
  --reasoning-data "$data/reasoning_candidates.jsonl"
