#!/bin/bash
#SBATCH --job-name=svrl-baseline
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# L1 baseline of a checkpoint on every svenska-rl environment (github.com/BirgerMoell/svenska-rl): vLLM OpenAI server
# with Hermes tool parsing on one GPU, `svrl.cli rollout` (standard library) inside the container, greedy, eval split.
#   POLICY=<model dir> NAME=<run name> [COUNT=64] [SVRL=<svenska-rl checkout>] sbatch slurm/svenska_rl_baseline_leonardo.sbatch
# The checkout must be prepared on a login node (compute nodes are offline): git clone/pull there first.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
: "${POLICY:?policy model dir}" "${NAME:?run name}"
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
SVRL=${SVRL:-$OELLM_SCRATCH/svenska-rl}
COUNT=${COUNT:-64}
OUT=$SVRL/runs/$NAME
mkdir -p "$OUT"
cd "$OUT"
echo "svenska-rl $(git -C "$SVRL" rev-parse --short HEAD), policy $POLICY"

# OpenCV's loader prepends its dir to sys.path; a spawned engine process then resolves `typing` to cv2/typing.
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" \
  env PYTHONPATH=$OELLM_WORK/europa/pylibs VLLM_WORKER_MULTIPROC_METHOD=fork HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$POLICY" --served-model-name policy --host 127.0.0.1 \
  --port 8000 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.9 --enable-auto-tool-choice \
  --tool-call-parser hermes > "$OUT/vllm.log" 2>&1 &
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo "vLLM did not come up" >&2; grep -m5 -i error "$OUT/vllm.log" >&2; exit 3; }
echo "server up $(date)"

for env in $(singularity exec -B /leonardo_scratch "$C" bash -c "cd $SVRL && python3 -c 'from svrl.core import manifests; print(\" \".join(m[\"id\"] for m in manifests()))'"); do
  singularity exec -B /leonardo_work,/leonardo_scratch "$C" bash -c "cd $SVRL && python3 -m svrl.cli rollout $env \
    --base-url http://127.0.0.1:8000/v1 --model policy --split eval --count $COUNT --workers 16 --max-tokens 4096 \
    --out $OUT/$env.jsonl" > "$OUT/$env.summary.json" 2> "$OUT/$env.err" || echo "rollout failed: $env"
  echo "== $env $(date +%T)"; cat "$OUT/$env.summary.json"
done
