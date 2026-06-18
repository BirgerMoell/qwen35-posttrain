# STATUS / Handoff — qwen35-posttrain

**Updated:** 2026-06-18. Self-contained state so anyone (or a fresh session) can continue.

## Where things are
- **GitHub:** https://github.com/BirgerMoell/qwen35-posttrain (public, `main`)
- **Local clone:** `/Users/birgermoell/AI-Sweden/qwen35-posttrain`
- Read first: `docs/PLAN.md`, `docs/ORCHESTRATION.md`, `data/SOURCES.md`.

## Decisions locked
- **Base:** `Qwen3.5-9B-Base` (dense, native **262K** context → long context already in the base;
  post-training must *preserve* it). **Validate on `Qwen3.5-2B-Base` first.**
- **Pipeline:** `SFT → DPO → RLVR/GRPO → (tool/agentic) → safety → eval/merge`. No "extend"
  stage (long ctx is native).
- **Frameworks (LUMI/ROCm, MI250X):** TRL (SFT/DPO), veRL (GRPO/scale), vLLM-ROCm
  (rollouts + eval). Use the project's **`BirgerMoell/trl`** fork.
- **Orchestration:** Slurm `afterok` dependency chain with **eval gates** (an eval job can fail
  to block the next stage — guards long-context regression).
- **Compute:** ~**300k GPU-h, deadline 2026-08-02**. Full 9B pipeline ≈ low-thousands GPU-h
  (~1%). Budget is ample → **time + a working ROCm RL/vLLM stack are the real constraints.**

## Built (scaffold)
- `pipeline/run_pipeline.sh <2b|9b>` — chained runner with eval gates
- `slurm/{sft,dpo,grpo,eval}_lumi.sbatch` — stage skeletons (env: `STAGE_LOAD/STAGE_SAVE/CFG`)
- `configs/{sft,dpo,grpo}_2b_smoke.yaml`, `configs/accelerate_fsdp.yaml`
- `data/SOURCES.md`, `scripts/stage_data_lumi.sh`
- `eval/run_eval.py` + `eval/check_gate.py` (stubs)
- `docs/{PLAN,ORCHESTRATION,lumi_rocm_runbook}.md`

## Data (see data/SOURCES.md)
- SFT: `openeurollm/dolci-instruct-sft-tokenized` (70 langs), `openeurollm/dolci-think-sft-tokenized`,
  `allenai/Dolci-Instruct-SFT-Tool-Use`
- DPO: `openeurollm/Dolci-Instruct-DPO-translated`
- RLVR: `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, `allenai/Dolci-RL-Zero-Mix-7B`
- Long-ctx SFT: synthesize from project corpus + `nvidia/ChatQA2-Long-SFT-data`
- Project-native: `BirgerMoell/grpo-data-bootstrap`, `openeurollm-language-dataset-candidates`,
  `OneRuler-OELLM` (eval)

## Next actions / critical path (NOT done yet)
1. **Build/extend the ROCm container** (TRL + vLLM + veRL) and **validate vLLM on MI250X** —
   the GRPO gate; front-load it.
2. **Confirm TRL entrypoints** vs `BirgerMoell/trl` (skeletons assume `trl.scripts.{sft,dpo,grpo}`).
3. **Implement real `eval/run_eval.py`** (lm-eval + RULER via vLLM) + `check_gate.py` thresholds.
   Note: for any base-model long-ctx eval use **completion/log-likelihood mode** (not generation).
4. **Stage base model + data** to LUMI scratch (`scripts/stage_data_lumi.sh P1`).
5. **`run_pipeline.sh 2b`** to validate the whole chain end-to-end, then **`9b`**.

## LUMI specifics
Project `project_465002530`; `dev-g` (30 min) / `standard-g`; container
`/scratch/project_462000963/containers/laif-rocm-6.4.4-…sif`; work dir
`/scratch/project_465002530/users/bmoell/`. **No internet on compute nodes** (stage on a login
node). **Login nodes flaky** (transport errors → retry). Gotchas: export `WORLD_SIZE=$SLURM_NTASKS`;
weights-only saves between stages; HF many-file uploads hit a 128-commits/hr limit.

## Boundary
Long-context **extension** is a *separate* project (`openeuro-longctx-datamix`, the baby model on
LUMI). Not part of post-training; for Qwen3.5 it doesn't apply (native 262K).
