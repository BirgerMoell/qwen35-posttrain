#!/bin/bash
#SBATCH --job-name=oellm-tool-translate
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Multilingual slice for openeurollm-tool-use-v1: one Ollama gemma4:31b server per GPU, then finalize.
# Pass extra arguments via TRANSLATE_ARGS (e.g. "--limit 32" for a smoke run). Resumable.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
tools=$OELLM_WORK/tools
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
export OLLAMA_MODELS=$tools/ollama-models-baseline OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=-1
ngpu=$(nvidia-smi -L | wc -l); endpoints=""
for g in $(seq 0 $((ngpu - 1))); do
  port=$((31000 + SLURM_JOB_ID % 10000 + g))
  CUDA_VISIBLE_DEVICES=$g OLLAMA_HOST=127.0.0.1:$port $tools/ollama-0.32.8/bin/ollama serve > $root/logs/ollama-translate-$SLURM_JOB_ID-$g.log 2>&1 &
  endpoints="$endpoints${endpoints:+,}http://127.0.0.1:$port"
done
for e in ${endpoints//,/ }; do until curl -sf $e/api/version > /dev/null; do sleep 2; done; done
digest=$(curl -s ${endpoints%%,*}/api/tags | python3 -c "import json,sys; print([m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='gemma4:31b'][0])")
echo "endpoints=$endpoints gemma4:31b digest=$digest"
srun singularity exec --nv -B /leonardo_work "$C" env PYTHONPATH=$OELLM_WORK/europa/pylibs HF_HUB_OFFLINE=1 \
  python "$root/translate_tool_use.py" --dataset-dir "$root/data/openeurollm-tool-use-v1" --pool-dir "$root/data/stage3-tool-pool-v1" \
  --work-dir "$root/data/tool-use-translation-work" --tokenizer "$root/tokenizer" --template "$root/oellm_qwen3_two_stage_v2_1.jinja" \
  --endpoints "$endpoints" --model gemma4:31b --model-revision "sha256:$digest" --threads $((ngpu * 8)) ${TRANSLATE_ARGS:-}
