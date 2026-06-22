#!/bin/bash
#SBATCH --job-name=distill-teacher
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-12:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Multi-node teacher serving for distillation: start a Ray cluster across the allocation,
# serve a large teacher with vLLM (tensor_parallel = total GPUs), generate verified traces.
# Scale the campaign with `sbatch --nodes=N` (a 122B teacher needs several nodes).
#   TEACHER : teacher model path (default Qwen3.5-32B for prototype; 122B needs more nodes)
#   PROMPTS : prompt-bank jsonl ({prompt|problem|question}[, answer])
#   OUT     : output traces jsonl
#   N, TEMP, MAXTOK, VERIFY, MODEL_IMPL, LIMIT : generation knobs (see distill_generate.py)
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${TEACHER:=/scratch/project_465002530/users/bmoell/models/Qwen3.5-32B}"
: "${PROMPTS:?set PROMPTS=/path/to/prompt_bank.jsonl}"
: "${OUT:?set OUT=/path/to/traces.jsonl}"
: "${N:=4}"; : "${TEMP:=0.8}"; : "${MAXTOK:=4096}"; : "${VERIFY:=none}"
: "${MODEL_IMPL:=auto}"; : "${LIMIT:=}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
RUN="export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets VLLM_ROCM_USE_AITER=1"

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD=$(echo "$NODES" | head -n1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')
PORT=6379
TP=$(( 8 * SLURM_NNODES ))
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"
echo "Teacher=$TEACHER  nodes=$SLURM_NNODES  TP=$TP  head=$HEAD_IP"

# ── Ray head on node 0 ──
srun --nodes=1 --ntasks=1 -w "$HEAD" singularity exec -B "$BIND" "$CONTAINER" bash -lc \
  "$RUN; ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=8 --disable-usage-stats --block" &
sleep 20

# ── Ray workers on the rest ──
for node in $(echo "$NODES" | tail -n +2); do
  srun --nodes=1 --ntasks=1 -w "$node" singularity exec -B "$BIND" "$CONTAINER" bash -lc \
    "$RUN; ray start --address=$HEAD_IP:$PORT --num-gpus=8 --disable-usage-stats --block" &
done
sleep 20

# ── Driver on the head: vLLM uses the whole Ray cluster ──
srun --nodes=1 --ntasks=1 -w "$HEAD" singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  $RUN &&
  export RAY_ADDRESS=$HEAD_IP:$PORT &&
  cd $SLURM_SUBMIT_DIR &&
  python scripts/distill_generate.py \
    --teacher '$TEACHER' --prompts '$PROMPTS' --out '$OUT' \
    --tp $TP --n $N --temperature $TEMP --max-tokens $MAXTOK \
    --verify $VERIFY --model-impl $MODEL_IMPL $LIM
"
echo "DISTILL-TEACHER DONE $(date) -> $OUT"
