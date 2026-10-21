#!/bin/bash
#SBATCH --job-name=oellm-stage2-reasoning
#SBATCH --account=aifac_f02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err

# Stage-2 Reasoning SFT, 8 nodes x 4 A100. Resumes from the latest checkpoint in
# output_dir, so a timed-out job is continued by resubmitting this script unchanged.
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
# Overrides exist only for the engineering smoke (STAGE1_SMOKE=1, tiny data, few steps).
data="${STAGE1_DATA:-$root/data/stage2-mix-v2}"
config="$root/sft_oellm_stage2_reasoning_v1_leonardo.yaml"
output="${STAGE1_OUTPUT:-$root/output/stage2-reasoning-v1}"
extra="${STAGE1_EXTRA_ARGS:-}"
template="$root/oellm_qwen3_two_stage_v2.jinja"

if [[ -s "$output/model.safetensors" || -s "$output/model.safetensors.index.json" ]]; then
  echo "Final model already saved in $output; nothing to do"; exit 0
fi
test -s "$data/manifest.json"
test -s "$root/output/stage1-instruct-v1-bf16/model.safetensors.index.json"
# The pre-tokenized data must come from exactly this template and tokenizer.
python3 - "$data/manifest.json" "$template" "$root/tokenizer/tokenizer.json" "${STAGE1_SMOKE:-0}" <<'PY'
import hashlib, json, sys
manifest, template, tokenizer, smoke = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(manifest))
allowed = {"stage2_reasoning_mix_pretokenized"} | ({"engineering_smoke_not_production"} if smoke == "1" else set())
assert m["kind"] in allowed, m["kind"]
assert m["config"]["template_sha256"] == sha(template), "template differs from build"
assert m["config"]["tokenizer_json_sha256"] == sha(tokenizer), "tokenizer differs from build"
print("manifest ok:", json.dumps(m["totals"]))
PY

export HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZER_NAME_OR_PATH="$root/tokenizer"
export CHAT_TEMPLATE_FILE="$template"
export REQUIRED_MODEL_DTYPE=bfloat16
export SFT_FORBID_LABEL_IDS=18   # <think> is always prompt-side; </think> is trained
export SFT_TRAIN_DATA_GLOB="$data/train-*.parquet"
export SFT_EVAL_DATA_GLOB="$data/validation-*.parquet"
export SFT_EVAL_MAX_ROWS=1024
export RESUME_FROM_CHECKPOINT=$([[ -n $(ls -d "$output"/checkpoint-* 2>/dev/null) ]] && echo 1 || echo 0)
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "resume=$RESUME_FROM_CHECKPOINT master=$MASTER_ADDR nodes=$SLURM_NNODES"

srun --unbuffered --kill-on-bad-exit=1 singularity exec --nv -B /leonardo_work:/leonardo_work "$container" \
  bash -c 'accelerate launch --config_file '"$root"'/accelerate_fsdp.yaml \
    --num_machines "$SLURM_NNODES" --machine_rank "$SLURM_NODEID" \
    --num_processes $((SLURM_NNODES * 4)) \
    --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
    '"$root"'/sft_train.py --config '"$config"' --output_dir '"$output"' '"$extra"
