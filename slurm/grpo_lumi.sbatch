#!/bin/bash
#SBATCH --job-name=pt-grpo
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-12:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# GRPO / RLVR stage (TRL GRPOTrainer + verifiable reward in scripts/grpo_train.py).
# Reads STAGE_LOAD (policy ckpt) / STAGE_SAVE / CFG. Multi-node: `sbatch --nodes=N`.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
ln -sf "${SLURM_JOB_NAME}-${SLURM_JOBID}.out" logs/latest.out 2>/dev/null || true
: "${STAGE_LOAD:?set by orchestrator}"; : "${STAGE_SAVE:?}"; : "${CFG:=configs/grpo_qwen35_9b_exam.yaml}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
PYLIBS_OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
HF_CACHE=/scratch/project_465002530/users/bmoell/hf_cache
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS_PER_NODE=8; NUM_PROC=$(( GPUS_PER_NODE * SLURM_NNODES ))
echo "GRPO  load=$STAGE_LOAD  save=$STAGE_SAVE  cfg=$CFG  nodes=$SLURM_NNODES  procs=$NUM_PROC"
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$PYLIBS_OVERLAY:\${PYTHONPATH:-} &&
  export HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets &&
  export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1 &&
  cd $SLURM_SUBMIT_DIR &&
  accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_PROC \
    --machine_rank \$SLURM_PROCID \
    scripts/grpo_train.py \
    --config $CFG --model_name_or_path '$STAGE_LOAD' --output_dir '$STAGE_SAVE'
"
echo "GRPO DONE $(date)"
