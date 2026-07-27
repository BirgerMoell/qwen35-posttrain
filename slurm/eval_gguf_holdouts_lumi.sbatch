#!/bin/bash
#SBATCH --job-name=eval-gguf
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-03:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.out
#
# Quantized quality-retention eval: serve a GGUF with llama-server (in-container build) and run
# the SAME EU-holdouts scorers against it (scripts/eval_gguf_holdouts.py). CPU inference — the GPU
# is just for the partition. Reports overall + per-language acc; comparable to the full-precision
# eval_eu_holdouts.py number.
#   GGUF  : path to the .gguf to evaluate (e.g. .../gguf-out/Qwen3.5-2B-EU-Q4_K_M.gguf)
#   TAG   : results label (eval/results/eu-holdouts/$TAG.json)
#   SPLIT : dev|test_public (default dev);  LIMIT : optional --limit-per-bucket
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
: "${GGUF:?}"; : "${TAG:=gguf}"; : "${SPLIT:=dev}"; : "${LIMIT:=}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
B=/scratch/project_465002530/users/bmoell
DATA=$SLURM_SUBMIT_DIR/data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/$SPLIT.jsonl
OUT=$SLURM_SUBMIT_DIR/eval/results/eu-holdouts/$TAG.json
PORT=8311
LIM=${LIMIT:+--limit-per-bucket $LIMIT}
echo "EVAL-GGUF gguf=$GGUF tag=$TAG split=$SPLIT"
SRVLOG=$SLURM_SUBMIT_DIR/logs/srv_${TAG}_${SLURM_JOB_ID}.log   # shared path (node /tmp is invisible)
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  cd $B/llama.cpp && export OMP_NUM_THREADS=32
  ./build-container/bin/llama-server -m '$GGUF' -c 4096 --host 127.0.0.1 --port $PORT -t 32 --no-webui >'$SRVLOG' 2>&1 &
  SRV=\$!
  UP=0
  for i in \$(seq 1 240); do
    if curl -s 127.0.0.1:$PORT/health 2>/dev/null | grep -q '\"status\":\"ok\"'; then UP=1; break; fi
    kill -0 \$SRV 2>/dev/null || { echo 'SERVER DIED during startup; log:'; tail -20 '$SRVLOG'; exit 1; }
    sleep 1
  done
  [ \$UP -eq 1 ] || { echo 'SERVER NOT READY after 240s; log:'; tail -20 '$SRVLOG'; kill \$SRV 2>/dev/null; exit 1; }
  echo \"server READY after \${i}s; sample generation:\"
  curl -s 127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Vad är meningen med livet? Svara kort på svenska.\"}],\"max_tokens\":120}'
  echo; echo '--- running holdouts eval ---'
  cd $SLURM_SUBMIT_DIR
  python3 scripts/eval_gguf_holdouts.py --base-url http://127.0.0.1:$PORT --data '$DATA' --out '$OUT' --tag '$TAG' $LIM
  kill \$SRV 2>/dev/null
"
echo "EVAL-GGUF $TAG DONE $(date) -> eval/results/eu-holdouts/$TAG.json"
