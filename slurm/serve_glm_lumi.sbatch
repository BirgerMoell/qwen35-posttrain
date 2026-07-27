#!/bin/bash
#SBATCH --job-name=glm-serve
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-08:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Serve GLM-5.2 (GLM-MoE, Glm4MoeForCausalLM — supported by vLLM 0.15.1 on ROCm) as an
# OpenAI-compatible endpoint. Single-node (--nodes=1, TP=8) or multi-node via a Ray cluster
# (--nodes=N, TP=8 x PP=N). Publishes the endpoint URL to logs/glm_endpoint.txt so clients on the
# LOGIN node (or another job) can reach it — compute nodes have no external internet but the
# internal Slingshot network + login->node routing work.
#
#   MODEL     model dir            (default $SCRATCH/models/GLM-5.2)
#   PORT      server port          (default 8000)
#   MAXLEN    max model len        (default 32768)
#   TP        tensor-parallel/node (default 8 = all GCDs on a node)
#   QUANT     e.g. fp8             (default empty = bf16; MI250X has no native FP8, prefer bf16)
#   EXTRA     extra vllm args
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
SCRATCH=/scratch/project_465002530/users/bmoell
: "${MODEL:=$SCRATCH/models/GLM-5.2}"; : "${PORT:=8000}"; : "${MAXLEN:=32768}"
: "${TP:=8}"; : "${QUANT:=}"; : "${EXTRA:=}"; : "${MODEL_IMPL:=}"  # MODEL_IMPL=transformers for glm_moe_dsa
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
HF_CACHE=$SCRATCH/hf_cache
PP=$SLURM_NNODES                                  # one pipeline stage per node
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}' | head -n1)
RAY_PORT=6379
# ROCm/RCCL networking (same as our training jobs) + let vLLM see all 8 GCDs
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
echo "GLM serve: model=$MODEL nodes=$SLURM_NNODES TP=$TP PP=$PP head=$HEAD:$RAY_PORT port=$PORT"

OVERLAY=$SCRATCH/pylibs-overlay        # transformers 5.12 has GlmMoeDsa (glm_moe_dsa)
run() { singularity exec -B "$BIND" "$CONTAINER" bash -lc "$1"; }
# ROCm needs HIP_VISIBLE_DEVICES (not the ROCR_VISIBLE_DEVICES Slurm sets); overlay on PYTHONPATH so
# vLLM's transformers backend can import glm_moe_dsa.
COMMON="if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]; then export HIP_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES; unset ROCR_VISIBLE_DEVICES; fi; \
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_HUB_OFFLINE=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1"

if [ "$SLURM_NNODES" -gt 1 ]; then
  # ---- multi-node: start a Ray cluster (head on node0, workers on the rest), then vllm on head ----
  echo "Starting Ray head on $HEAD ..."
  srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" singularity exec -B "$BIND" "$CONTAINER" bash -lc \
    "$COMMON && ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --num-gpus=8 --block" &
  sleep 20
  echo "Starting Ray workers ..."
  srun --overlap --nodes=$((SLURM_NNODES-1)) --ntasks=$((SLURM_NNODES-1)) --exclude="$HEAD" \
    singularity exec -B "$BIND" "$CONTAINER" bash -lc \
    "$COMMON && ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=8 --block" &
  sleep 25
  DIST="--distributed-executor-backend ray --pipeline-parallel-size $PP"
else
  DIST=""
fi

echo "http://$HEAD_IP:$PORT/v1" > "$SLURM_SUBMIT_DIR/logs/glm_endpoint.txt"
echo "ENDPOINT -> http://$HEAD_IP:$PORT/v1  (also in logs/glm_endpoint.txt)"

# ---- launch the OpenAI-compatible server on the head node ----
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  $COMMON &&
  vllm serve '$MODEL' \
    --served-model-name glm-5.2 \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size $TP $DIST \
    ${MODEL_IMPL:+--model-impl $MODEL_IMPL} \
    --max-model-len $MAXLEN \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    ${QUANT:+--quantization $QUANT} $EXTRA
"
echo "GLM serve exited $(date)"
