#!/bin/bash
#SBATCH --job-name=oellm-reasoning-quick
#SBATCH --account=aifac_f02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err
# Quick default-vs-thinking reasoning eval of one checkpoint: 4 GPUs data-parallel, then score.
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
MODEL=${MODEL:-$root/output/stage2-reasoning-v1-bf16}
NAME=${NAME:-stage2-reasoning-v1}
out=$root/eval/reasoning-quick/$NAME
mkdir -p $out/results
run() { singularity exec --nv -B /leonardo_work "$C" env HF_HUB_OFFLINE=1 python "$root/eval_reasoning_quick.py" "$@"; }
for mode in default thinking; do
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g run run --model "$MODEL" --data "$root/data/eval-quick/reasoning_quick.jsonl" \
      --out "$out/results/$mode-$g.jsonl" --mode $mode --rank $g --world 4 > $out/$mode-$g.log 2>&1 &
  done
  wait
done
run score --results "$out/results" --out "$out/report.json"
