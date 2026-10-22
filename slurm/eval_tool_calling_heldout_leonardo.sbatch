#!/bin/bash
#SBATCH --job-name=oellm-tool-heldout
#SBATCH --account=aifac_f02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err
# Held-out tool-calling check of one checkpoint (default mode): 4 GPUs data-parallel, then score.
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
MODEL=${MODEL:-$root/output/stage3-tooluse-v1-bf16}
NAME=${NAME:-stage3-tooluse-v1}
DATA=$root/data/eval-quick/tool_heldout.jsonl
out=$root/eval/tool-heldout/$NAME
mkdir -p $out/results
run() { singularity exec --nv -B /leonardo_work "$C" env HF_HUB_OFFLINE=1 python "$root/eval_tool_calling_heldout.py" "$@"; }
[[ -s $DATA ]] || run prepare --dataset $root/data/openeurollm-tool-use-v1 --out $DATA
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g run run --model "$MODEL" --data "$DATA" --out "$out/results/$g.jsonl" --rank $g --world 4 \
    > $out/$g.log 2>&1 &
done
wait
run score --results "$out/results" --out "$out/report.json"
