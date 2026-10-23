#!/bin/bash
#SBATCH --job-name=oellm-stage4-dpo
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

# Stage-4 DPO, 8 nodes x 4 A100 by default (submit with --nodes=4 when the queue is full; continuations keep the
# node count and grad-accum is derived from it). Resumes from the latest checkpoint in output_dir. Each production job, once it
# is running, queues its own continuation (afterany on itself) while DPO_CHAIN_REMAINING > 0 (default 1), so a
# continuation can never start because a predecessor was cancelled before it ran.
# Engineering smoke (DPO_SMOKE=1, submit with --nodes=2 --time=01:00:00; one node cannot hold the fp32 optimizer
# state): 1,536 pairs = 12 steps at the production global batch; phase A stops with a checkpoint after step 6,
# phase B resumes and finishes.
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
data=/leonardo_scratch/large/userexternal/pmoell00/oellm-dpo/pretok-v1
config="$root/dpo_oellm_stage4_v1_leonardo.yaml"
template="$root/oellm_qwen3_two_stage_v2_1.jinja"
model="$root/output/stage3-tooluse-v1.1-bf16"
output=/leonardo_scratch/large/userexternal/pmoell00/oellm-dpo/output/stage4-dpo-v1
smoke="${DPO_SMOKE:-0}"
[[ "$smoke" == 1 ]] && output="${output}-smoke"

if [[ -s "$output/model.safetensors.index.json" ]]; then
  echo "Final model already saved in $output; nothing to do"; exit 0
fi
test -s "$data/manifest.json" && test -s "$model/model.safetensors.index.json"
cmp "$model/chat_template.jinja" "$template"
gpus=$((SLURM_NNODES * 4))
per_device=$(awk '/^per_device_train_batch_size:/ {print $2}' "$config")
(( 128 % (gpus * per_device) == 0 )) || { echo "global batch 128 is not divisible by $gpus GPUs x $per_device"; exit 1; }
accum=$((128 / (gpus * per_device)))

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/pmoell00/oellm-dpo/datasets-cache
export TOKENIZER_NAME_OR_PATH="$model" CHAT_TEMPLATE_FILE="$template" REQUIRED_MODEL_DTYPE=bfloat16
export DPO_DATA_MANIFEST="$data/manifest.json"
export DPO_TRAIN_DATA_GLOB="$data/train-*.parquet" DPO_EVAL_DATA_GLOB="$data/validation-*.parquet"
export DPO_FORBID_TOKEN_IDS=11,12,13,14,15,16,17,18
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

launch() {  # $1 = extra trainer args
  srun --unbuffered --kill-on-bad-exit=1 singularity exec --nv -B /leonardo_work:/leonardo_work \
    -B /leonardo_scratch:/leonardo_scratch "$container" \
    bash -c 'accelerate launch --config_file '"$root"'/accelerate_fsdp.yaml \
      --num_machines "$SLURM_NNODES" --machine_rank "$SLURM_NODEID" \
      --num_processes $((SLURM_NNODES * 4)) \
      --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
      '"$root"'/dpo_train_pretok.py --config '"$config"' --output_dir '"$output"' \
      --gradient_accumulation_steps '"$accum"' '"$1"
}

if [[ "$smoke" == 1 ]]; then
  export DPO_TRAIN_MAX_ROWS=1536 DPO_EVAL_MAX_ROWS=64
  smoke_args="--logging_steps 1 --eval_steps 6 --save_steps 6"
  echo "SMOKE phase A: steps 1-6, checkpoint, stop (accum=$accum on $gpus GPUs)"
  RESUME_FROM_CHECKPOINT=0 DPO_STOP_AFTER_STEP=6 launch "$smoke_args"
  test -d "$output/checkpoint-6"
  echo "SMOKE phase B: resume from checkpoint-6, finish step 12"
  RESUME_FROM_CHECKPOINT=1 launch "$smoke_args"
  exit 0
fi
export DPO_EVAL_MAX_ROWS=1024
chain="${DPO_CHAIN_REMAINING:-1}"
if (( chain > 0 )); then
  next=$(sbatch --parsable --nodes="$SLURM_NNODES" --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,DPO_CHAIN_REMAINING=$((chain - 1)) \
    "$root/dpo_oellm_stage4_v1_leonardo.sbatch")
  echo "queued continuation $next (afterany:$SLURM_JOB_ID, $((chain - 1)) more after it)"
fi
export RESUME_FROM_CHECKPOINT=$([[ -n $(ls -d "$output"/checkpoint-* 2>/dev/null) ]] && echo 1 || echo 0)
echo "resume=$RESUME_FROM_CHECKPOINT master=$MASTER_ADDR nodes=$SLURM_NNODES accum=$accum"
launch ""
