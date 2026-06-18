# Pipeline Orchestration

The pipeline is **separate stages chained as one sequence** (not one merged training run —
see `docs/PLAN.md` for why). Stages run on LUMI as a **Slurm dependency chain with eval gates**.

```
(extend*) → SFT → [eval] → DPO → [eval] → GRPO → [eval]
              │        │       │       │        │       │
            BASE→sft  gate   sft→dpo  gate   dpo→grpo  gate
```
\*`extend` is **skipped for Qwen3.5** (native 262K). For a custom base, prepend a long-context
CPT stage (Megatron, the `openeuro-longctx-datamix` repo) before SFT.

## How it works
- **`pipeline/run_pipeline.sh <size>`** submits every stage with `sbatch --parsable
  --dependency=afterok:<prev>` and passes `STAGE_LOAD` / `STAGE_SAVE` / `CFG` via `--export`.
- **Checkpoint flow:** `BASE → output/<size>/sft → .../dpo → .../grpo`. Each stage saves
  **weights-only** (`save_only_model`), since the next stage reloads weights, not optimizer.
- **Eval gates:** after each stage an `eval` job runs (`slurm/eval_lumi.sbatch`). The **next
  stage depends on the eval job** (`afterok`), so if eval exits non-zero the chain halts.
  `eval/check_gate.py` defines the gate (e.g. **long-context RULER must not drop >2 pts**;
  core score must not drop below a floor). This is what prevents post-training from silently
  eroding the base's 262K context.

## Run it
```bash
# stage data + base model to LUMI scratch first (login node — no internet on compute nodes)
bash scripts/stage_data_lumi.sh P1
# validate the WHOLE chain on 2B (cheap), then run the real artifact
bash pipeline/run_pipeline.sh 2b
bash pipeline/run_pipeline.sh 9b
```

## Status of the scaffold
Skeletons in place: `pipeline/run_pipeline.sh`, `slurm/{sft,dpo,grpo,eval}_lumi.sbatch`,
`configs/{sft,dpo,grpo}_2b_smoke.yaml`, `configs/accelerate_fsdp.yaml`.
**TODO to make runnable:** (1) confirm the TRL entrypoints against `BirgerMoell/trl`;
(2) implement `eval/run_eval.py` + `eval/check_gate.py`; (3) build/extend the ROCm container
with TRL/vLLM(/veRL) and **validate vLLM on MI250X** (the GRPO risk); (4) stage base + data.

## Cost/time (see PLAN.md)
~300k GPU-h, deadline 2026-08-02. Budget is ample; validate on 2B first, then 9B. Front-load
the ROCm/vLLM container validation — it's the critical path, not compute.
