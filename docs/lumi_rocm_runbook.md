# LUMI / ROCm Runbook (post-training)

LUMI = AMD **MI250X** (2 GCDs × 64 GB per node = 8 "GPUs"/node), **ROCm 6.4-era**, Slurm,
Lustre. Compute nodes have **no internet** → stage all models/data/pip on a login node first.

## Accounts / storage
- Project: `project_465002530` (compute) / `project_462000963` (shared data, containers).
- Persistent data: `/scratch/project_4650025*/...` and `/flash/...`. Home is tiny.
- Containers: `/scratch/project_462000963/containers/` (e.g. the OELLM `laif-rocm-*.sif`).

## Container
Post-training needs **TRL + transformers + vLLM (ROCm) + datasets**, and **veRL** for RL.
Options (TODO pick/build):
1. Extend the existing `laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif`
   with `pip install trl peft vllm` (ROCm wheels) inside a writable overlay.
2. Build a dedicated ROCm image (TRL/veRL/vLLM) via the AMD ROCm base images.
- **vLLM on MI250X is the key risk** (AMD's RL guides target MI300X+ROCm7.0). Validate a
  tiny vLLM generate on MI250X before committing to GRPO.

## Slurm pattern (mirror the long-ctx scripts)
- `--account=project_465002530 --partition=dev-g` (30 min, validation) or `standard-g` (real).
- `--gpus-per-node=8 --ntasks-per-node=8 --cpus-per-task=7`.
- `srun ... singularity exec -B /scratch,/flash,... $CONTAINER launch.sh <python ...>`.
- NCCL/RCCL: `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`, `NCCL_NET_GDR_LEVEL=PHB`.
- Parallelism: SFT/DPO use FSDP or DeepSpeed ZeRO-3; RL uses veRL's engine + vLLM rollouts.

## Validate-tiny-first checklist
1. `vllm` generates on 1 MI250X GCD with Qwen3.5-2B-Base (ROCm sanity).
2. TRL `SFTTrainer` 10 steps on 2B (loss decreases, checkpoint saves).
3. TRL `DPOTrainer` 10 steps on the SFT checkpoint.
4. GRPO 5 steps on 2B with a trivial verifiable reward (vLLM rollout loop works).
5. Then scale to 9B and real data budgets.

## Gotchas (carried from the long-ctx work)
- dev-g 30-min cap: keep validation runs short; for full runs use standard-g.
- Checkpoint saves to Lustre are slow — save weights-only between stages where the next
  stage doesn't need optimizer state.
- LUMI login nodes occasionally throw Lustre "transport endpoint" errors — retry.
- Direct LUMI↔other-cluster SSH is blocked (portal-managed keys); move big files via the
  shared project dirs or HF (mind HF's 128-commits/hour limit on many-file uploads).
