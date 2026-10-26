#!/bin/bash
#SBATCH --job-name=oellm-safety-dpo-data
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Judge-labelled on-policy DPO data (scripts/judge_onpolicy_dpo.py): translate (Ollama gemma4:31b, once) -> sample
# the POLICY in default mode (4 vLLM processes, one per GPU) -> judge (Ollama gemma4:31b, the EUROPA judge digest) ->
# pairs -> pre-tokenize with the main DPO guards. Resumable phase by phase.
#   POLICY=<model dir> [WORK=<dir with prompts.jsonl>] [PHASES="translate sample judge pairs pretok"] [SMOKE=1] sbatch ...
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
tools=$OELLM_WORK/tools
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
: "${POLICY:?policy model dir}"
WORK=${WORK:-$OELLM_SCRATCH/oellm-dpo/safety-v1}
PHASES=${PHASES:-translate sample judge pairs pretok}
JREV=sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7
if [[ "${SMOKE:-0}" == 1 ]]; then  # 12 prompts per behaviour/category, separate work dir
  src=$WORK; WORK=$WORK-smoke; mkdir -p $WORK
  python3 -c "
import json,collections,sys
c=collections.Counter(); out=open('$WORK/prompts.jsonl','w')
for l in open('$src/prompts.jsonl'):
    r=json.loads(l); k=(r['behaviour'],r['category'] if r['behaviour']=='grounded' else '')
    if c[k]<12: out.write(l); c[k]+=1
"
fi
test -s "$WORK/prompts.jsonl"
py() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH=$OELLM_WORK/europa/pylibs HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$@"; }
endpoints=""
start_ollama() {
  export OLLAMA_MODELS=$tools/ollama-models-baseline OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=-1
  endpoints=""
  for g in 0 1 2 3; do
    port=$((32000 + SLURM_JOB_ID % 10000 + g))
    CUDA_VISIBLE_DEVICES=$g OLLAMA_HOST=127.0.0.1:$port $tools/ollama-0.32.8/bin/ollama serve > $root/logs/ollama-safety-$SLURM_JOB_ID-$g.log 2>&1 &
    endpoints="$endpoints${endpoints:+,}http://127.0.0.1:$port"
  done
  for e in ${endpoints//,/ }; do until curl -sf $e/api/version > /dev/null; do sleep 2; done; done
  got=$(curl -s ${endpoints%%,*}/api/tags | python3 -c "import json,sys; print([m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='gemma4:31b'][0])")
  [[ "sha256:$got" == "$JREV" ]] || { echo "judge digest $got != $JREV" >&2; exit 2; }
}
stop_ollama() { pkill -u "$USER" -f "ollama serve" || true; sleep 5; }
J="$root/judge_onpolicy_dpo.py"
for phase in $PHASES; do
  echo "== $phase $(date)"
  case $phase in
    translate|judge) start_ollama; py python "$J" $phase --work "$WORK" --endpoints "$endpoints" --judge-revision "$JREV" --threads 32; stop_ollama ;;
    sample) for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g py "$VLLM_PY" "$J" sample --work "$WORK" --policy "$POLICY" --shard $g --shards 4 > "$WORK/sample-$g.log" 2>&1 & done; wait ;;
    pairs) py python "$J" pairs --work "$WORK" --policy "$POLICY" --judge-revision "$JREV" ;;
    pretok) py python "$root/build_oellm_dpo_v1_pretok.py" --staged "$WORK/staged" --tokenizer "$POLICY" \
              --template "$root/oellm_qwen3_two_stage_v2_1.jinja" --out "$WORK/pretok" --workers 8 ;;
  esac
done
