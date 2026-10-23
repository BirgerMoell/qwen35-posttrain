#!/bin/bash
#SBATCH --job-name=oellm-code-evalplus
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# HumanEval+ / MBPP+ baseline for up to 4 (model, mode) pairs, one GPU each, then isolated scoring.
# MODELS="name=path name=path" (each model gets a default-mode and a thinking-mode GPU; 2 models fill the node).
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
E=$OELLM_SCRATCH/code-eval
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
UV_PY=$OELLM_SCRATCH/oellm-rl/uv-python
SITE=$E/evalplus-venv/lib/python3.12/site-packages
script=$root/eval_code_evalplus.py
run_dir=$E/runs/$SLURM_JOB_ID
mkdir -p "$run_dir"

gpu=0
for pair in $MODELS; do
  name=${pair%%=*}; model=${pair#*=}
  for mode in default thinking; do
    mkdir -p "$run_dir/$name"
    CUDA_VISIBLE_DEVICES=$gpu singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" \
      env HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$E/cache PYTHONPATH=$SITE "$VLLM_PY" "$script" gen \
      --model "$model" --out "$run_dir/$name/$mode" --modes $mode > "$run_dir/$name-$mode.log" 2>&1 &
    gpu=$((gpu + 1))
  done
done
wait

# Scoring executes model-written code: no home, no project directories; only this run's samples are writable.
for d in "$run_dir"/*/*/; do
  cp -r "$E/cache" "$d/cache"
  cp "$script" "$d/score.py"
  singularity exec --contain --cleanenv --no-home \
    -B "$E/evalplus-venv:$E/evalplus-venv:ro" -B "$UV_PY:$UV_PY:ro" -B "$d:$d" "$C" \
    env XDG_CACHE_HOME="$d/cache" "$E/evalplus-venv/bin/python" "$d/score.py" score --out "$d" --parallel 16 \
    > "$d/score.log" 2>&1 || echo "scoring failed in $d"
done
for f in "$run_dir"/*/*/report.json; do echo "== $f"; cat "$f"; done
