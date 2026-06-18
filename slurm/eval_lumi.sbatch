#!/bin/bash
#SBATCH --job-name=pt-eval
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-00:30:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# EVAL GATE. Evaluates $CKPT (lm-eval-harness + RULER via vLLM-ROCm). If a gate metric
# regresses vs $BASELINE (esp. long-context RULER, or a hard core-score floor), exit non-zero
# so the orchestrator's `afterok` dependency BLOCKS the next stage.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${CKPT:?}"; : "${TAG:=ckpt}"; : "${BASELINE:=}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
RESULTS="$SLURM_SUBMIT_DIR/eval/results/$TAG"
mkdir -p "$RESULTS"
echo "EVAL $TAG  ckpt=$CKPT  baseline=${BASELINE:-none}"

srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  cd $SLURM_SUBMIT_DIR &&
  # TODO: run the suite (eval/README.md): MMLU, GSM8K/MATH, HumanEval, IFEval, RULER (long-ctx),
  # multilingual (OneRuler). vLLM (ROCm) backend. Write metrics to '$RESULTS/metrics.json'.
  python eval/run_eval.py --model '$CKPT' --out '$RESULTS/metrics.json' --suite core,ruler
"
# GATE: compare to baseline; non-zero exit halts the chain.
# TODO: implement eval/check_gate.py (e.g. RULER must not drop >2 pts; core must not drop >1 pt).
python eval/check_gate.py --metrics "$RESULTS/metrics.json" ${BASELINE:+--baseline-tag "$(basename "$BASELINE")"} \
  || { echo "GATE FAILED for $TAG — blocking downstream stages"; exit 1; }
echo "EVAL $TAG PASSED $(date)"
