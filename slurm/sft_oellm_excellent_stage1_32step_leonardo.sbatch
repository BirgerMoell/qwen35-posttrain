#!/bin/bash
#SBATCH --job-name=oellm-stage1-32step
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
config="$root/sft_oellm_excellent_stage1_32step_leonardo.yaml"
# Site paths in the YAML are ${OELLM_*} references; render them for this job.
mkdir -p "$root/tmp"; rendered="$root/tmp/${SLURM_JOB_ID:-local}-$(basename "$config")"
python3 "$root/render_site_config.py" "$config" > "$rendered"; config="$rendered"
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
data="$root/data/excellent-stage1-full-v1"
output="$root/output/excellent-stage1-32step-v1"
test -s "$data/manifest.json"
test -s "$root/base-256k-anneal300b/pytorch_model.bin"
echo "5dcf789668d1233b48c8e99bf3733f0e80abbde4a74de1974b0605108030378c  $root/base-256k-anneal300b/pytorch_model.bin" | sha256sum --check
if [[ -e "$output" ]]; then
  echo "Output already exists; this weight-only slice requires a new output path" >&2
  exit 2
fi

export HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZER_NAME_OR_PATH="$root/tokenizer"
export CHAT_TEMPLATE_FILE="$root/oellm_qwen3_no_empty_assistant_mask.jinja"
export REQUIRED_MODEL_DTYPE=bfloat16
export SFT_FORBID_THINK_LABELS=1
export SFT_TRAIN_DATA_GLOB="$data/warmup-*.jsonl"
export SFT_EVAL_DATA_GLOB="$data/validation-*.jsonl"
export SFT_EVAL_MAX_ROWS=128
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun singularity exec -B /leonardo_work:/leonardo_work "$container" \
  python "$root/verify_oellm_stage1_manifest.py" \
  --manifest "$data/manifest.json" \
  --template "$CHAT_TEMPLATE_FILE" \
  --tokenizer "$TOKENIZER_NAME_OR_PATH"

port=$((20000 + SLURM_JOB_ID % 20000))
srun --unbuffered singularity exec --nv -B /leonardo_work:/leonardo_work \
  "$container" accelerate launch \
  --config_file "$root/accelerate_fsdp.yaml" \
  --num_machines 1 --num_processes 4 --main_process_port "$port" \
  "$root/sft_train.py" --config "$config"
