# Self-hosted GLM-5.2 inference on LUMI

Run an open-weight **GLM-MoE** model as an OpenAI-compatible endpoint on LUMI, for use as a
**distillation teacher**. See `glm52_architecture_report.md` for the GLM-5.2 teardown.

> **GLM-5.2 serving reality (verified on LUMI, 2026-07):**
> - **GLM-5.2** is `GlmMoeDsaForCausalLM` (`glm_moe_dsa`) — **NOT in vLLM 0.15.1's registry** (same
>   gap as qwen3_5), but **IS in our transformers 5.12 overlay and RUNS on ROCm** (validated with a
>   tiny random-init model: forward + generate OK).
> - **CRITICAL FLAG:** load with **`experts_implementation="eager"`** and
>   **`attn_implementation="eager"`**. The default MoE uses DeepGEMM grouped-gemm (CUDA-only) →
>   `grouped gemm is not supported on ROCM`. Eager forces the pure-torch loop MoE (`GlmMoeDsaNaiveMoe`),
>   which works on MI250X (slower, but correct). Set it via kwarg OR bake
>   `"_experts_implementation": "eager"` into the downloaded `config.json`.
> - **Footprint:** GLM-5.2 = 744B / 40B active, ~1.5 TB bf16 (`zai-org/GLM-5.2`) → 3–4 nodes; FP8
>   (`zai-org/GLM-5.2-FP8`) ~744 GB → 2 nodes (FP8-on-ROCm dequant is an extra unknown). No vLLM
>   kernels → serve via **transformers + accelerate/device_map** (eager MoE = low throughput, fine
>   for distillation data-gen, not for latency).
> - **GLM-4.6** (`Glm4MoeForCausalLM`) remains the *fast* option (native vLLM, 2 nodes) if throughput
>   matters more than being exactly 5.2.

## Architecture

```
  login node (has internet)                compute nodes (NO external internet)
  ┌───────────────────────┐                ┌───────────────────────────────────────┐
  │ 1. download_glm.sh    │  weights on    │  serve_glm_lumi.sbatch                 │
  │    HF -> scratch      │─── scratch ───▶│  vLLM OpenAI server (Glm4MoeForCausalLM)│
  │                       │                │  node0 ── Ray ── node1 (TP=8 x PP=2)    │
  │ 3. distill / smoke    │                │  writes logs/glm_endpoint.txt           │
  │    client  ───────────┼── internal ───▶│  http://<node0-ip>:8000/v1             │
  │    (OpenAI API)       │    network     │                                         │
  └───────────────────────┘                └───────────────────────────────────────┘
```

Key facts:
- **Weights download runs on the login node** (compute nodes have no external internet); the server
  runs offline (`HF_HUB_OFFLINE=1`) off scratch.
- The server binds `0.0.0.0` on a compute node. **Clients run on the login node** (or another job)
  and hit `http://<node-ip>:<port>/v1` over LUMI's internal Slingshot network — the endpoint IP is
  written to `logs/glm_endpoint.txt`.

## Sizing (MI250X = 64 GB/GCD, 8 GCD/node, CDNA2 → **no native FP8**, use bf16)

| Variant | bf16 weights | GCDs (weights+KV+overhead) | Nodes |
|---|---|---|---|
| GLM-5.2 full (~355B) | ~710 GB | ~14–16 | **2** (TP=8 × PP=2) |
| GLM-5.2 Lite/Air (if published) | ~200–240 GB | ~4–6 | **1** (TP=8) |
| FP8 quant (experimental on MI250X) | ~355 GB | ~7–8 | 1 (may be slow — no FP8 HW) |

Start on **1 node** if a Lite/Air variant exists; otherwise **2 nodes** bf16.

## Steps

**1. Download the weights (login node).** Set the real GLM-5.2 repo id.
```bash
GLM_REPO=zai-org/GLM-5.2 bash scripts/download_glm.sh      # -> $SCRATCH/models/GLM-5.2
```

**2. Serve.**
```bash
# single node (Lite/Air or FP8):
sbatch --nodes=1 --export=ALL,MODEL=$SCRATCH/models/GLM-5.2,TP=8 slurm/serve_glm_lumi.sbatch
# full model, two nodes (bf16, TP=8 x PP=2):
sbatch --nodes=2 --export=ALL,MODEL=$SCRATCH/models/GLM-5.2,TP=8 slurm/serve_glm_lumi.sbatch
```
Watch `logs/<jobid>.out` for `ENDPOINT -> http://<ip>:8000/v1` (also in `logs/glm_endpoint.txt`).
Model load for a 355B is several minutes.

**3. Smoke-test (login node).**
```bash
python scripts/glm_smoke.py --wait 600     # polls /health, lists models, runs a completion
```

**4. Distill through it.** `scripts/distill_from_9b.py` already speaks the OpenAI API:
```bash
python scripts/distill_from_9b.py --prompts data/distill_seed_prompts.jsonl \
  --api-base $(cat logs/glm_endpoint.txt) --api-model glm-5.2 \
  --out data/distill/glm_to_2b.jsonl
```
(Run the client on the login node; no `GLM_API_KEY` needed for the local server — a dummy bearer.)

## Notes / expected first-run friction
- **Multi-node Ray** under Slurm+Singularity is the fragile part: the sbatch starts a Ray head on
  node0 and workers on the rest, then `vllm serve --distributed-executor-backend ray
  --pipeline-parallel-size <nodes>`. If workers don't join, check `NCCL_SOCKET_IFNAME=hsn0..3` and
  that the head IP resolved correctly. Single-node (no Ray) is the robust first target.
- **`--trust-remote-code`** is on (GLM ships custom modeling code); pin the revision for reproducibility.
- If VRAM is tight, lower `--max-model-len` or `--gpu-memory-utilization`, or add `--max-num-seqs`.
- The server holds the allocation for its `--time`; size that to the distillation batch, or run the
  client against it within the same window.
