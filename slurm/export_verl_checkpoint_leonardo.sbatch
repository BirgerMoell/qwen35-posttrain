#!/bin/bash
#SBATCH --job-name=export-verl
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Export a verl FSDP checkpoint (global_step_N/actor) to a Hugging Face model directory with our pinned tokenizer and
# chat template v2.1 (verl.model_merger), then check tokenizer and template hashes against the parent.
#   RUN=<verl output dir> PARENT=<HF dir of the student> DST=<dir> [STEP=latest] sbatch slurm/export_verl_checkpoint_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
V=$OELLM_SCRATCH/oellm-rl/verl/.venv/bin/python
: "${RUN:?}" "${PARENT:?}" "${DST:?}"
step=${STEP:-$(ls -d "$RUN"/global_step_* | sort -t_ -k3 -V | tail -1)}
[[ -d $step ]] || step=$RUN/global_step_$STEP
echo "exporting $step -> $DST"
mkdir -p "$DST"
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH="$OELLM_SCRATCH/oellm-rl/verl" "$V" -m verl.model_merger merge \
  --backend fsdp --local_dir "$step/actor" --target_dir "$DST"
cp "$PARENT/tokenizer.json" "$PARENT/tokenizer_config.json" "$DST/"
cp "$root/oellm_qwen3_two_stage_v2_1.jinja" "$DST/chat_template.jinja"
[[ -f $PARENT/generation_config.json ]] && cp "$PARENT/generation_config.json" "$DST/"
echo "$step" > "$DST/SOURCE"
sha256sum "$DST/tokenizer.json" "$PARENT/tokenizer.json" "$DST/chat_template.jinja" | cut -c1-16
ls -la "$DST"
