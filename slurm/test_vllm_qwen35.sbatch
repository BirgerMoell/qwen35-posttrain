#!/bin/bash
#SBATCH --job-name=test-vllm-qwen35
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=0-00:30:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Can vLLM serve Qwen3.5 for GRPO rollouts? vLLM 0.15.1 has no NATIVE qwen3_5 model, but it
# has a TRANSFORMERS BACKEND that runs any transformers-supported arch (our overlay ships
# transformers 5.x which DOES support qwen3_5). This tests that path end to end.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
MODEL=/scratch/project_465002530/users/bmoell/models/Qwen3.5-2B   # small, fast probe

srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-} HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  export VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 &&
  python3 - <<'PYEOF'
from vllm import LLM, SamplingParams
import traceback
for impl in ('transformers', 'auto'):
    try:
        print(f'=== trying model_impl={impl} ===', flush=True)
        llm = LLM(model='/scratch/project_465002530/users/bmoell/models/Qwen3.5-2B',
                  model_impl=impl, dtype='bfloat16', tensor_parallel_size=2,
                  gpu_memory_utilization=0.85, max_model_len=4096, trust_remote_code=True,
                  enforce_eager=True)
        out = llm.generate(['Hej, vad är 2+2? Svara kort.'],
                           SamplingParams(temperature=0, max_tokens=32))
        print(f'SUCCESS impl={impl}:', repr(out[0].outputs[0].text), flush=True)
        break
    except Exception as e:
        print(f'FAILED impl={impl}: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
PYEOF
"
echo "vLLM qwen3.5 test DONE $(date)"
