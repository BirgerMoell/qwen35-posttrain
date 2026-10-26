#!/bin/bash
#SBATCH --job-name=europa-stage1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# EUROPA Eval + Alignment for a HF checkpoint, matching the Neonkraft SFT baseline protocol
# (temperature 0, seed 42, bf16, judge gemma4:31b @ sha256:6316f062... via Ollama 0.32.8).
# Judge on GPU 1, target on GPU 0. Offline; model pre-cached in $E/hf-cache. Resumable.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
E=$OELLM_WORK/europa
T=$OELLM_WORK/tools
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
MODEL=${MODEL:-birgermoell/oellm-9b-256k-stage1-instruct-v1}
REV=${REV:-58e42ab4e06751275393441b3f2e6c2f08755159}
NAME=${NAME:-birgermoell-oellm-9b-256k-stage1-instruct-v1-huggingface}
JUDGE=gemma4:31b
JREV=sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7
PORT=$((30000 + SLURM_JOB_ID % 20000))

export OLLAMA_MODELS=$T/ollama-models-baseline OLLAMA_HOST=127.0.0.1:$PORT
CUDA_VISIBLE_DEVICES=1 $T/ollama-0.32.8/bin/ollama serve > $E/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
until curl -sf http://$OLLAMA_HOST/api/version > /dev/null; do sleep 2; done
got=$(curl -s http://$OLLAMA_HOST/api/tags | python3 -c "import json,sys; print([m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='$JUDGE'][0])")
[[ "sha256:$got" == "$JREV" ]] || { echo "judge digest $got != $JREV" >&2; exit 2; }
echo "judge ok: $JUDGE $got"

run() {  # $1 = repo dir, $2 = cli name
  (cd "$E/$1" && singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env \
    PYTHONPATH="$E/pylibs:$E/$1/src" HF_HOME="$E/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m "$2" run --backend huggingface --model "$MODEL" --revision "$REV" --device cuda:0 --dtype bfloat16 \
      --seed 42 --judge-backend ollama --judge-model "$JUDGE" --judge-revision "$JREV" \
      --judge-base-url "http://$OLLAMA_HOST" --output "results/runs/$NAME.json")
}
run europa-eval europa_eval
echo "europa-eval done $(date)"
run europa-alignment europa_alignment
echo "europa-alignment done $(date)"
