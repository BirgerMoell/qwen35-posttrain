#!/bin/bash
#SBATCH --job-name=search-qa
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Search-augmented factual QA (scripts/eval_search_qa.py): the offline Wikipedia search server (oellm-rlvr
# scripts/wiki_search.py, its own venv, host process on this node) + vLLM policy with the wiki_search tool.
#   MODEL=<dir> NAME=<name> [DATA=hallu-v1/eval.jsonl] [MODES="default thinking"] sbatch slurm/eval_search_qa_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
SEARCH=$OELLM_SCRATCH/oellm-search
CODE=$OELLM_SCRATCH/oellm-rl/oellm-rlvr
: "${MODEL:?}" "${NAME:?}"
out=$root/eval/search-qa/$NAME
mkdir -p "$out"
"$SEARCH/.venv/bin/python" "$CODE/scripts/wiki_search.py" serve --root "$SEARCH" --port 18090 \
  --preload en de fr es it nl sv pl pt fi > "$out/search_server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf -o /dev/null http://127.0.0.1:18090/health && break; sleep 5; done
curl -sf http://127.0.0.1:18090/health || { echo "search server did not start" >&2; tail -20 "$out/search_server.log" >&2; exit 2; }
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  PYTHONPATH=$CODE/src "$PY" "$root/eval_search_qa.py" --model "$MODEL" --data "${DATA:-$root/data/hallu-v1/eval.jsonl}" \
  --out "$out" --modes ${MODES:-default thinking}
