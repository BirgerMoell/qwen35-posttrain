#!/bin/bash
#SBATCH --job-name=oellm-reasoning-quick
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Quick default-vs-thinking reasoning eval of one checkpoint: 4 GPUs data-parallel, then score.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
MODEL=${MODEL:-$root/output/stage2-reasoning-v1-bf16}
NAME=${NAME:-stage2-reasoning-v1}
DATA=${DATA:-$root/data/eval-quick/reasoning_quick.jsonl}
MODES=${MODES:-default thinking}
MAXNEW=${MAXNEW:-0}
out=$root/eval/reasoning-quick/$NAME
mkdir -p $out/results
run() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 python "$root/eval_reasoning_quick.py" "$@"; }
for mode in $MODES; do
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g run run --model "$MODEL" --data "$DATA" --max-new-tokens $MAXNEW \
      --out "$out/results/$mode-$g.jsonl" --mode $mode --rank $g --world 4 > $out/$mode-$g.log 2>&1 &
  done
  wait
done
run score --results "$out/results" --out "$out/report.json"
