# Runbook — Post-training the OELLM-9B-256K long-context model

**Audience:** an autonomous agent (or engineer) picking up cold. Follow top-to-bottom.
**Goal:** run the full post-training pipeline (**SFT → SimPO**) on the OpenEuroLLM 9B 256K base model,
then evaluate — **general quality** (EU holdouts + Swedish rubric) **and long context** (needle-in-a-haystack).
The context extension is already done upstream (256K); the point here is (1) prove the pipeline works on
this base and (2) *measure* how well long context survives instruction tuning (currently unknown, because a
base model can't be instruction-evaluated).

Model card: https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude

---

## 0. Key facts (read before doing anything)

- **Architecture:** `model_type: qwen3`, `Qwen3ForCausalLM` — a **plain dense Qwen3 9B** (hidden 4096,
  36 layers, 32 heads / 8 KV GQA). Standard `AutoModelForCausalLM` — **no** fla-core / special kernels
  (unlike the Qwen3.5 hybrid models this repo mostly targets). `max_position_embeddings: 262144`,
  `rope_theta: 64_000_000` — **never change rope_theta**, it's what makes 256K work.
- **⚠️ Tokenizer is GEMMA-style, not Qwen.** Chat markers are `<start_of_turn>` / `<end_of_turn>`
  (NOT `<|im_start|>`), eos is `<eos>`, and pad is `<pad>`. The tokenizer has 262,145 entries while
  the model embedding shape is 263,168; preserve both. `<|im_start|>` etc. map to unk.
  This is the single most important gotcha — the wrong template silently trains garbage.
- **It's a BASE model** — no chat template, not instruction-tuned. SFT is what makes it usable *and*
  unlocks the ability to evaluate long-context instruction-following.

## 1. Environment / locations (LUMI, project_465002530, user bmoell)

```
REPO      /scratch/project_465002530/users/bmoell/qwen35-posttrain      # this git repo, synced from GitHub
BASE      /scratch/project_465002530/users/bmoell/longctx-extend/prelude_256k_hf   # raw base weights (pytorch_model.bin, 18GB)
MODEL     /scratch/project_465002530/users/bmoell/models/oellm-9b-256k-chat        # ALREADY STAGED: symlinks to weights + Gemma chat_template.jinja  <-- USE THIS as STAGE_LOAD
CONTAINER /scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY   /scratch/project_465002530/users/bmoell/pylibs-overlay        # transformers 5.12 (PYTHONPATH)
HF_CACHE  /scratch/project_465002530/users/bmoell/hf_cache
DATA      /scratch/project_465002530/users/bmoell/posttrain-data/       # SFT + preference parquets
```
Work from the login node (compute nodes have no internet). Container/overlay/HF_HOME are wired inside the
sbatch files already — you normally don't invoke the container by hand.

**The staged MODEL dir already exists** (created with the correct Gemma chat template). If it's gone,
recreate it: symlink `pytorch_model.bin`, `tokenizer.json`, `tokenizer.model` from BASE; copy `config.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`; and add `chat_template.jinja` with
the Gemma template (system merged into first user turn, role `assistant`→`model`, using
`<start_of_turn>`/`<end_of_turn>`; `add_generation_prompt` → `<start_of_turn>model\n`).

## 2. VALIDATE the chat template FIRST (cheap, do not skip)

Render real training rows through the staged tokenizer+template on the **login node** (instant). Confirm
you see `<start_of_turn>user … <end_of_turn>` / `<start_of_turn>model …`, roles mapped, sane token counts.
If this looks wrong, STOP — do not launch a multi-hour SFT on a broken template.

```bash
C=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
singularity exec -B /pfs,/scratch,/appl,/opt/cray "$C" bash -lc '
export PYTHONPATH=/scratch/project_465002530/users/bmoell/pylibs-overlay:${PYTHONPATH:-}
python - <<PY
import pyarrow.parquet as pq
from transformers import AutoTokenizer
D="/scratch/project_465002530/users/bmoell/models/oellm-9b-256k-chat"
t=AutoTokenizer.from_pretrained(D, trust_remote_code=True)
assert t.chat_template, "NO CHAT TEMPLATE — fix the staged dir"
parquet=pq.ParquetFile("/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-sft-parquet/train.parquet")
rows=next(parquet.iter_batches(batch_size=3, columns=["messages"])).to_pylist()
for r in rows:
    print(t.apply_chat_template(r["messages"], tokenize=False)[:300], "\n---")
PY'
```

## 3. Create the configs (in the repo)

The pipeline is data-driven via YAML (same schema as the Qwen3.5 configs). Create these two files.

**`configs/sft_oellm9b_256k.yaml`**
```yaml
# SFT — OELLM-9B-256K (Qwen3 dense + Gemma tokenizer). Standard EU instruction mix; 4 nodes x 8 GPU.
model_name_or_path: /scratch/project_465002530/users/bmoell/models/oellm-9b-256k-chat   # overridden by STAGE_LOAD
torch_dtype: bfloat16
attn_implementation: flash_attention_2  # required for boundary-safe packed training
datasets:
  - path: parquet
    data_files: {train: /scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-sft-parquet/train.parquet}
    split: train                   # full general EU set (~1.08M), no-think
  - path: parquet
    data_files: {train: /scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-reasoning-sft-parquet/train.parquet}
    split: "train[:250000]"        # ~20% reasoning (think-format)
output_dir: /scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft
eos_token: <end_of_turn>
max_length: 4096                   # standard. NOTE: short SFT may or may not degrade 256K — that's what we measure.
packing: true
dataset_num_proc: 6
ddp_timeout: 18000
max_steps: 3000
per_device_train_batch_size: 1
gradient_accumulation_steps: 4     # global batch 128 on 32 procs
learning_rate: 6.0e-6
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
logging_steps: 20
save_steps: 1000
save_total_limit: 3
report_to: none
```

**`configs/simpo_oellm9b_256k.yaml`**
```yaml
# SimPO — OELLM-9B-256K. Reference-free preference on EU exam pairs. STAGE_LOAD overrides to the SFT ckpt.
model_name_or_path: /scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft
torch_dtype: bfloat16
attn_implementation: flash_attention_2
datasets:
  - path: parquet
    data_files: {train: /scratch/project_465002530/users/bmoell/posttrain-data/exam-dpo-parquet/train.parquet}
    split: train
output_dir: /scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-simpo
loss_type: simpo
cpo_alpha: 0.0
simpo_gamma: 0.5
beta: 2.0
max_length: 1024
max_prompt_length: 768
max_steps: 1500
dataset_num_proc: 8
ddp_timeout: 18000
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
logging_steps: 20
save_steps: 500
save_total_limit: 3
report_to: none
```

Commit both configs to the repo before launching.

## 4. Launch SFT (stage 1)

`slurm/sft_lumi.sbatch` is env-parameterized (`STAGE_LOAD` / `STAGE_SAVE` / `CFG`). It runs
`scripts/sft_train.py` (plain `AutoModelForCausalLM` + FSDP), one `accelerate` launch per node × 8 GPU
workers. **Point STAGE_LOAD at the staged chat dir**, not the raw base.

```bash
cd /scratch/project_465002530/users/bmoell/qwen35-posttrain
O=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output
SFT=$(sbatch --nodes=4 --gpus-per-node=8 --time=0-20:00:00 \
  --export=ALL,RANK0_META_LOAD=1,STAGE_LOAD=/scratch/project_465002530/users/bmoell/models/oellm-9b-256k-chat,STAGE_SAVE=$O/oellm9b-256k-sft,CFG=configs/sft_oellm9b_256k.yaml \
  slurm/sft_lumi.sbatch | awk '{print $NF}')
echo "SFT job $SFT"
```

**Watch:** `logs/<jobid>.err` for a falling train loss (`{'loss': ...}` lines). First loss should be
finite and decrease; if it's NaN or flat, the template/tokenizer is wrong — kill and fix step 2.
9B on 4 nodes ≈ several hours for 3000 steps. `save_steps: 1000` writes checkpoints.

**If it times out mid-run:** checkpoints use `--save_only_model` (no optimizer state) so you **cannot**
optimizer-resume. Warm-start instead: make a `*_finish.yaml` with a low constant LR and fewer steps, and
set `STAGE_LOAD` to the last `checkpoint-*` dir (see `sft_qwen35_4b_eu_v3_finish.yaml` for the pattern).

## 5. Launch SimPO (stage 2), chained after SFT

```bash
SIMPO=$(sbatch --nodes=2 --gpus-per-node=8 --time=0-10:00:00 --dependency=afterok:$SFT \
  --export=ALL,STAGE_LOAD=$O/oellm9b-256k-sft,STAGE_SAVE=$O/oellm9b-256k-simpo,CFG=configs/simpo_oellm9b_256k.yaml \
  slurm/simpo_lumi.sbatch | awk '{print $NF}')
echo "SimPO job $SIMPO (afterok:$SFT)"
```
(Optional GRPO stage — verifiable exam-MCQ reward — mirrors `grpo_lumi.sbatch` + a
`grpo_oellm9b_256k.yaml`. Not required for the first pass.)

## 6. Evaluate (the deliverable)

Run these on the **SimPO** output (and optionally the SFT output for a delta). All are transformers-based
(no vLLM). Set eos to `<end_of_turn>` for generation if the model over-runs (Gemma chat convention).

**A. General quality — EU holdouts (deterministic, test_public):**
```bash
python scripts/eval_eu_holdouts.py --model $O/oellm9b-256k-simpo \
  --data data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/test_public.jsonl \
  --out eval/results/oellm9b_256k_holdouts.json --batch-size 32
```
Run inside the container with the overlay (see `slurm/eval_batch_lumi.sbatch` — set
`DRIVER` to a one-line script that calls the above; submit `--nodes=1`).

**B. Real generations — Swedish rubric:**
```bash
python scripts/eval_swedish_rubric.py --model $O/oellm9b-256k-simpo \
  --prompts data/swedish_rubric_prompts.jsonl --out eval/results/oellm9b_256k_svrubric.json --tag oellm9b-256k
```
Then score the generated answers against the rubric (språkkvalitet / korrekthet / instruktion /
hjälpsamhet) — an LLM judge or manual. This is where you sanity-check the model actually follows Swedish
instructions coherently.

**C. Long context — needle-in-a-haystack (the headline for THIS model):**
There is **no long-context eval script in the repo yet — build one** (`scripts/eval_niah.py`, transformers,
device_map="auto"): embed a unique "needle" fact at controlled depths (0%/25%/50%/75%/100%) inside filler
context at several lengths (**8K / 32K / 128K / 256K**), ask the instruction-tuned model to retrieve it,
score exact-match. Report a depth×length grid. This directly answers "does the 256K context survive SFT?"
Keep `rope_theta=64M`; generate with the chat template; use `max_new_tokens` small (the answer is short).
Compare, if useful, against the base model (but note base can't follow the retrieval instruction well —
that's exactly why SFT was needed).

## 7. Publish (only on a confirmed win)

Discipline from this repo: **republish only when a same-split head-to-head shows a win.** Since there's no
prior instruct version of this model, the bar is: coherent instruction-following (rubric) + non-degenerate
long context (NIAH grid). If good:
1. Recast to bf16: `sbatch --export=ALL,SRC=$O/oellm9b-256k-simpo slurm/recast_bf16_lumi.sbatch` → `*-bf16`.
2. Set `generation_config.json` eos to `<end_of_turn>` id + keep `rope_theta=64M`.
3. Upload: `python scripts/upload_model_hf.py --ckpt $O/oellm9b-256k-simpo-bf16 --repo birgermoell/<name> --card model_cards/<card>.md`
   (needs HF write token in env; NEVER hardcode it in a file).
4. Write a model card noting the 256K context + the NIAH grid + rubric numbers.

## 8. Known gotchas (from prior runs on this cluster)

- **Gemma tokenizer** — wrong chat template = silent garbage. Validate (step 2) before every fresh run.
- **`SLURM_GPUS_ON_NODE`** reports physical GCD count (8) even if you ask for fewer — the sbatch already
  handles `GPUS_PER_NODE`; for a big model needing fewer ranks/node, pass `--gpus-per-node=N` AND
  `--export=...,GPUS_PER_NODE=N`.
- **`--save_only_model`** → no optimizer-resume; warm-start from the consolidated checkpoint instead.
- **Rank-0-only loading is mandatory here.** A one-node smoke with the default per-rank load reached
  508 GB RSS and was OOM-killed (job `20288313`). `RANK0_META_LOAD=1` completed the same smoke with a
  260 GB transient peak (job `20288700`). Keep it enabled for the 4-node run.
- **Packed SFT requires FlashAttention 2.** Without `attn_implementation: flash_attention_2`, TRL 0.28
  warns that flattened packed samples can cross-attend and contaminate one another. The exact staged-model
  path completed a five-step packed FA2 smoke in job `20289762` (loss 1.245 → 1.268, all finite) and saved
  the chat template plus the unchanged 262,144-token / theta-64M model metadata.
- **Consolidated FSDP saves are FP32.** Even with bf16 training, this stack writes a ~36.4 GB FP32 model
  (and records `dtype: float32`). This is expected and reloads into bf16 when `torch_dtype: bfloat16` is
  explicit. Run the recast step in section 7 before publishing.
- **LUMI login nodes flap / maintenance** — compute jobs survive, but multi-node distributed jobs die on
  TCPStore drops; just resubmit.
- **eos for generation:** base uses `<eos>`; for a Gemma-style chat model set generation eos to
  `<end_of_turn>` or it may not stop cleanly.

## 9. Success criteria

1. SFT loss falls smoothly; SimPO completes; both save models.
2. EU holdouts overall accuracy is materially above the base (base ≈ pretrained-only, weak on instructions).
3. Swedish rubric: coherent, in-Swedish, instruction-following answers.
4. NIAH grid: retrieval works at 8K/32K and degrades gracefully (or not) toward 256K — **report the grid
   whatever it shows**; a degradation at 256K after short SFT is itself the finding this experiment exists to produce.
```
