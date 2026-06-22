# RUNBOOK — Reproduce the EU Post-Training Pipeline

Single source of truth for running the OpenEuroLLM post-training pipeline on LUMI.
Written so a **human or an AI agent** can reproduce the work end to end. If something here
disagrees with reality, fix reality or fix this file — don't let them drift.

> **TL;DR** — On LUMI we fine-tune a strong 2026 instruct model (**Qwen3.5-9B**, with Gemma 4
> as a second track) on European multilingual data (Dolci suite) to improve EU languages
> without regressing English/reasoning. Validated end-to-end: **SFT and DPO both train and
> produce checkpoints**; eval harness produces real numbers (base Qwen3.5-9B gsm8k = 88%).

---

## 0. Coordinates

| What | Where |
|---|---|
| GitHub | `https://github.com/BirgerMoell/qwen35-posttrain` |
| LUMI project | `project_465002530` (compute) · data also on `project_462000963` |
| LUMI work dir | `/scratch/project_465002530/users/bmoell/` |
| Repo on LUMI | `/scratch/project_465002530/users/bmoell/qwen35-posttrain` (git pull to update) |
| Login | `ssh bmoell@lumi.csc.fi` (no internet on **compute** nodes — stage everything on login) |

---

## 1. Environment (the part that took the longest to get right)

**Container** (ROCm 6.4, PyTorch 2.9, has TRL 0.28 + vLLM 0.15 but old transformers 4.57.3):
```
/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
```

**Python overlay** — the container's `transformers 4.57.3` does NOT know the 2026
architectures (`qwen3_5`, `gemma4`). We install newer libs to a scratch dir and prepend it
to `PYTHONPATH` **inside** the container:
```
/scratch/project_465002530/users/bmoell/pylibs-overlay
```
Contains: `transformers>=5.5` (needs ≥5.2 for Qwen3.5, ≥5.5 for Gemma 4), `mistral_common`,
`weave`, `lm-eval[ifeval]` + `langdetect`/`nltk`/`immutabledict`, `flash-linear-attention`.

To (re)build the overlay on a login node:
```bash
C=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
O=/scratch/project_465002530/users/bmoell/pylibs-overlay
singularity exec -B /scratch/project_465002530:/scratch/project_465002530 "$C" \
  pip install --target="$O" "transformers>=5.5" mistral_common weave \
    "lm-eval[ifeval]" langdetect nltk immutabledict flash-linear-attention
```

**Critical bind-mount gotcha:** Singularity does NOT pass host `PYTHONPATH` in, and only sees
paths you `-B` bind. Every job must:
```bash
singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd "$C" bash -lc '
  export PYTHONPATH=/scratch/project_465002530/users/bmoell/pylibs-overlay:$PYTHONPATH
  ...'
```
(Note `/pfs` in the binds — compute nodes see `/scratch` as `/pfs/lustrep3/scratch`.)

---

## 2. Models & data (already staged on LUMI)

**Models** under `/scratch/project_465002530/users/bmoell/models/`:
- `Qwen3.5-9B` (instruct — has chat_template), `Qwen3.5-9B-Base`, `Qwen3.5-2B`, `gemma-4-E2B-it`
- Download on a login node via the container's python (`huggingface_hub.snapshot_download`),
  binding `-B /scratch/project_465002530:/scratch/project_465002530`.

**SFT data** — Dolci suite, already on `project_462000963`:
- Mix used: `…/SFTTrainer_format/multiling/openeurollm-sft-mix/tulu3-euroblocks-85-15/`
  (`euroblocks.jsonl` = EU multilingual, `tulu3-commercial.jsonl` = English replay; 85/15)
- **Pre-converted to Parquet** (memory-mapped — see gotcha #3):
  `/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-sft-parquet/train.parquet`
  (1,082,196 examples, `{messages}` format)

**DPO data**: `…/DPOTrainer_format/eng/Dolci-Instruct-DPO/train.jsonl` (260k `{prompt,chosen,rejected}` pairs)

**Eval data**: `…/posttraining_data/ArenaHard/{eng,fin}/train.jsonl`

---

## 3. The pipeline

Stages share `scripts/{sft,dpo}_train.py` (custom **text-only** TRL entrypoints — see gotchas)
and `configs/*.yaml` (flat TRL schema: ModelConfig + DatasetMixtureConfig + SFT/DPOConfig).

```bash
cd /scratch/project_465002530/users/bmoell/qwen35-posttrain && git pull

# SFT -> DPO chained (DPO runs afterok of SFT)
M=/scratch/project_465002530/users/bmoell/models/Qwen3.5-9B
O=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output
SFT=$(sbatch --parsable --time=24:00:00 \
  --export=ALL,STAGE_LOAD=$M,STAGE_SAVE=$O/qwen35-9b-sft,CFG=configs/sft_qwen35_9b.yaml \
  slurm/sft_lumi.sbatch)
sbatch --time=12:00:00 --dependency=afterok:$SFT \
  --export=ALL,STAGE_LOAD=$O/qwen35-9b-sft,STAGE_SAVE=$O/qwen35-9b-dpo,CFG=configs/dpo_qwen35_9b.yaml \
  slurm/dpo_lumi.sbatch
```

Smoke first (2B, 20 steps, `SMOKE=1`) to validate the stack cheaply — see `configs/*_2b_smoke.yaml`.

**Hardware per job:** 1 node = 8 MI250X GCDs, full fine-tuning with FSDP (FULL_SHARD,
`TRANSFORMER_BASED_WRAP`, `cpu_ram_efficient_loading`). Loss is healthy: SFT 1.30→0.96
(acc 0.71→0.76); DPO reward margins grow ~10× (validated on 2B).

---

## 4. Evaluation

```bash
# Standard suite (regression guard + ifeval) — run on base + each stage
sbatch --export=ALL,MODEL=$M,TAG=base,TASKS=gsm8k,ifeval,arc_challenge,hellaswag,mmlu \
  slurm/eval_lmeval_lumi.sbatch
sbatch --export=ALL,MODEL=$O/qwen35-9b-sft,TAG=sft,TASKS=... slurm/eval_lmeval_lumi.sbatch
python eval/compare.py --base base --stages sft,dpo      # delta table

# EU win-rate (the European delta) — generate candidate+baseline, LLM-judge head-to-head
sbatch --export=ALL,CANDIDATE=$O/qwen35-9b-sft,CAND_TAG=sft,BASELINE=$M \
  slurm/arena_eu_lumi.sbatch                              # -> eval/results/arena/sft_vs_base.json
```
Add `LIMIT=100` for a quick smoke. Judge defaults to self-judge (base model) — swap a larger
neutral judge via `JUDGE=` for a credible win-rate.

---

## 5. Gotchas & fixes (why each config setting exists)

Each of these cost a debug cycle. Don't undo them without understanding why.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `does not recognize qwen3_5/gemma4` | container transformers 4.57.3 too old | transformers 5.x overlay on `PYTHONPATH` |
| 2 | `EADDRINUSE` port 29500 | 8 srun tasks each ran `accelerate launch` | `--ntasks-per-node=1`; accelerate spawns 8 workers |
| 3 | CPU OOM at data load | 8 ranks each parse 1M-row JSON to Arrow | pre-convert to **Parquet** (memory-mapped, shared) |
| 4 | CPU OOM at model load | 8 ranks each load full multimodal model | text-only `AutoModelForCausalLM` + `cpu_ram_efficient_loading` |
| 5 | NCCL barrier timeout 1800s | rank-0 dataset preprocessing too slow | `ddp_timeout: 10800` + `dataset_num_proc` |
| 6 | GPU (HIP) OOM at seq 8192 | activations too big | `max_length: 4096` + `expandable_segments:True` |
| 7 | `Could not find …VisionBlock` (FSDP) | text-only model still lists vision in `_no_split_modules` | prune to classes present (in `*_train.py`) |
| 8 | `size mismatch … vec(...)` in DPO | `SIZE_BASED_WRAP` wraps embeddings standalone | `TRANSFORMER_BASED_WRAP` |
| 9 | DPO `KeyError 'images'` / processor error | multimodal model + processor on text DPO | text-only entrypoint, pass `AutoTokenizer` as `processing_class` |
| 10 | DPO works on Qwen, **not** Gemma-4-E2B | E-series is elastic/MatFormer (per-layer embeddings) — no clean text-only causal LM | use dense Gemma variants or Qwen for DPO; E-series DPO deferred |
| 11 | Slow SFT (~135 s/step) | Qwen3.5 hybrid arch (linear-attn/conv) falls back to torch | install `flash-linear-attention` (Triton, ROCm-capable) — in progress |

---

## 6. Reproduce from scratch (checklist)

1. `ssh bmoell@lumi.csc.fi`; `git clone` the repo into the work dir (or `git pull`).
2. Build the overlay (§1) on a login node.
3. Stage model + Parquet data (§2) on a login node.
4. Smoke: `SMOKE=1` 2B SFT + DPO (`configs/*_2b_smoke.yaml`) — must hit the training loop.
5. Real run: §3 (SFT→DPO chain).
6. Eval: §4 (base baseline, then stages, then `compare.py` + ArenaHard-EU).

---

## 7. Plans & status
- Strategy & two-track (instruct delta vs base full): `docs/PLAN.md`, `docs/PLAN_QWEN35.md`, `docs/PLAN_GEMMA4.md`
- Data sources: `data/SOURCES.md`
- Eval details: `eval/README.md`
