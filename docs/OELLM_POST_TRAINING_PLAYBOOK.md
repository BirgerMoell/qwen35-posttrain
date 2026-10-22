# OpenEuroLLM 9B post-training playbook

A complete, reproducible record of how the OpenEuroLLM 9B 256K base model is post-trained into an instruct, reasoning
and tool-using assistant on Leonardo (CINECA): **Stage 1 Instruct SFT → Stage 2 Reasoning SFT → Stage 3 Tool-use SFT
→ Stage 4 DPO → Stage 5 RL**. Every number here comes from a run artifact (manifest, log, eval report); file paths are
relative to the `qwen35-posttrain` repository unless marked as Leonardo paths.

This document is maintained as the run progresses. **Status (2026-10-22):** Stages 1–3 done and published; Stage 4
DPO data in preparation; Stage 5 RL probe in its Leonardo canary phase.

| Stage | Checkpoint (HF, private) | Revision | Leonardo cost | EUROPA Eval | EUROPA Alignment |
|---|---|---|---:|---:|---:|
| Base | `openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b` | sha256-pinned copy | — | — | — |
| 1 Instruct | `birgermoell/oellm-9b-256k-stage1-instruct-v1` | `58e42ab4` | 3,153 local-h | 0.600 | 0.455 |
| 2 Reasoning | `birgermoell/oellm-9b-256k-stage2-reasoning-v1` | `2db72459` | 2,811 local-h | 0.609 | 0.605 |
| 3 Tool use | `birgermoell/oellm-9b-256k-stage3-tooluse-v1` | v1 `0dc97efa`, **v1.1 `88875629`** | 541 local-h | 0.616 | 0.642 |
| 4 DPO | in preparation | | est. 4.7–7K local-h | | |
| 5 RL | probe in canary phase | | probe ≤ 320 local-h | | |

Reference points on the same EUROPA protocol: Neonkraft instruct SFT (same base family) 0.762, threeway DPO 0.840,
Qwen3.6-35B-A3B 0.881. Our gap is analysed in §8.3 and addressed in Stage 4.

---

## 1. Environment

### 1.1 Cluster, accounts, budget
- **Leonardo Booster** (CINECA): 4 × A100 64 GB per node, 32 CPU cores, 512 GB RAM; partition `boost_usr_prod`
  (24 h job limit); CPU builds on `lrd_all_serial` (4 cores, 30 GB, nearly free).
- Account `aifac_f02_352` (400K local-h, shared). **Working budget for this project: ~50K local-h.**
  Billing: 1 GPU-hour = 8 local-h, so an 8-node × 4-GPU job costs 256 local-h per wall-hour.
  Measure own usage: `sacct -u $USER -A aifac_f02_352 -S <date> -X -P -o ElapsedRaw,AllocTRES` (sum
  `ElapsedRaw/3600 × billing`); `saldo` is account-wide.
- Spent by 2026-10-22 evening: ~7.35K local-h (Stage 1 3,153; Stage 2 2,811; Stage 3 541; tool-data translation 438;
  evaluations, diagnostics, canaries the rest).
- Login: `step ssh login '<email>' --provisioner cineca-hpc` (certificate expires; renew when SSH says
  "Permission denied"). Host-key alias: `ssh -o HostKeyAlias=login.leonardo.cineca.it pmoell00@login07-ext.leonardo.cineca.it`.

### 1.2 Storage layout
| Location | Use | Limits |
|---|---|---|
| `/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight` | SFT code (flat), data, final BF16 exports, logs | **1 TB shared project quota**; keep only final exports and datasets |
| `/leonardo_work/AIFAC_F02_352/pmoell00/europa` | EUROPA harnesses, pinned pylibs, HF cache | small |
| `$SCRATCH = /leonardo_scratch/large/userexternal/pmoell00` | RL env (`oellm-rl/`), DPO staging (`oellm-dpo/`), rollouts, checkpoints, archives | PB-scale, **40-day purge** |
| `/leonardo_scratch/fast/AIFAC_F02_352` | project fast scratch | full; do not use |

Full SFT checkpoints with optimizer state are ~135–170 GB; delete them once the BF16 export and checks pass. Anything
long-term goes to Hugging Face. Use `setsid nohup` for long commands on login nodes; the login node kills processes
that use too much CPU or memory (it killed a 1.4M-row staging pass), so heavy CPU work goes to `lrd_all_serial`.

### 1.3 Software
- Container: `/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif`
  (glibc 2.39; torch 2.9 NV, transformers 5.3.0, TRL 0.29.1, accelerate, flash-attn, liger).
- The host is RHEL 8.8 / glibc 2.28: wheels needing glibc ≥ 2.31 (vLLM) must be installed and run **inside** the
  container. The RL environment is a `uv` venv built inside the container (§10.1).
- Compute nodes are offline: download models/datasets on a login node into an HF cache, run jobs with
  `HF_HUB_OFFLINE=1`. HF token only via stdin (`cat ~/.cache/huggingface/token | ssh … 'export HF_TOKEN=$(cat); …'`),
  never written to disk on the cluster.

---

## 2. Base model and tokenizer

- Base: `openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`, Qwen3ForCausalLM 9B, 256K context
  (RoPE θ = 64M), vocab 263,168, untied embeddings. Staged sha256-pinned at `…/base-256k-anneal300b`.
- Tokenizer: `…/sft-migration-preflight/tokenizer`, `tokenizer.json` sha256 `5123f74b254340ff…` (pinned in every
  data manifest; every launcher asserts it).

| Token | ID | | Token | ID |
|---|---:|---|---|---:|
| `<\|im_start\|>` | 3 | | `<tool_response>` | 15 |
| `<\|im_end\|>` (EOS) | 4 | | `</tool_response>` | 16 |
| `<tool_call>` | 11 | | `</think>` | 17 |
| `</tool_call>` | 12 | | `<think>` | 18 |
| `<tools>` | 13 | | pad | 262144 |
| `</tools>` | 14 | | | |

IDs 11–18 are added tokens (non-special). **They start with nearly identical embedding and `lm_head` rows**; see §7.4
for why this matters. The threeway DPO model uses different IDs (262144+) and is incompatible: never reuse its
tokenizer, template, weights or tokenized data.

---

## 3. Chat templates

All stages use our templates only (`templates/`), tested in `tests/test_chat_template_v2*.py`.

| Template | sha256 (prefix) | Used for |
|---|---|---|
| `oellm_qwen3_two_stage_v2.jinja` | `b4854e3a1d8fe7f8` | Stages 1–2 |
| `oellm_qwen3_two_stage_v2_1.jinja` | `b96d9bbad3f81d32` | Stage 3 onward (adds Qwen3/Hermes tools); release template |
| `oellm_qwen3_two_stage_v2_1_rl_think.jinja` | — | RL only: thinking ON unless `enable_thinking=False` (RL backends call the template without kwargs) |

Invariants (enforced by tests and data builders):
- Hard `enable_thinking` switch. Default: generation prompt `<|im_start|>assistant\n`, labels `answer<|im_end|>\n`, no
  think tokens anywhere. Thinking: prompt ends `<|im_start|>assistant\n<think>\n`, labels
  `trace\n</think>\n\nanswer<|im_end|>\n`.
- `<think>` is always prompt-side: the model never learns to open a think block. No empty think blocks.
- Reasoning lives in `reasoning_content`; inline `<think>` in content is rejected by the template.
- Earlier assistant turns carry no reasoning, except (v2.1) assistant steps after the last user turn in a tool loop.
- Tools (v2.1): `<tools>` JSON list in the system turn; assistant emits `<tool_call>{"name":…,"arguments":{…}}</tool_call>`;
  results come back as `<tool_response>` blocks in a user turn. Tool-free conversations render byte-identically to v2
  (verified on 2,843 real rows), so Stage-2 replay tokens are reused unchanged in Stage 3.

---

## 4. Data principles (all stages)

1. **Pre-tokenize offline** into Parquet (`input_ids`, `assistant_masks`, provenance columns) with the pinned
   tokenizer and template; record both sha256 values in `manifest.json`. Launchers refuse to train if either differs.
2. **Reject, never truncate**: rows over the stage's max length are dropped and counted.
3. **Label guards in the trainer** (`scripts/sft_train.py`): `SFT_FORBID_THINK_LABELS=1` (Stage 1: abort on any
   supervised think token) or `SFT_FORBID_LABEL_IDS` (Stage 2: `18`; Stage 3: `18,15,13`, i.e. never supervise `<think>`,
   `<tool_response>`, `<tools>`). The collator aborts the batch; it never silently masks.
4. Deterministic selection by hashing row ids (`sha256(id + salt)`), so mixes are reproducible and held-out buckets are
   stable (e.g. Stage-3 validation = `sha256(id + "#val") < 0.005`).
5. Every mix ships a manifest with per-source rows, rendered tokens and supervised tokens.

---

## 5. Stage 1 — Instruct SFT

**Goal:** a strong non-reasoning instruct model; prepare everything for Stage 2.

**Data** `data/stage1-mix-v1` (builder `scripts/build_oellm_stage1_mix.py`, launcher
`slurm/build_oellm_stage1_mix_leonardo.sbatch`): **3,135,379 train rows, 2.365B rendered / 1.707B supervised tokens**;
3,270 validation rows. Max 4,096 tokens per row (2,826 overlength + 13,528 invalid rows rejected).

| Source | Train rows | Rendered tokens | Notes |
|---|---:|---:|---|
| OpenEuroLLM-Excellent-SFT direct rows (dolci, nemotron chat/code/math, …) | ~2.63M | ~1.9B | curated core; mostly English |
| Translated Dolci (`cs de el es fi fr it nl pl ro sv uk`) | 477,998 | 451M | each source-family id assigned to exactly one language by hash |
| Aya (human), non-English targets | 22,274 | 3.1M | ≤ 8K per language, answers ≥ 50 chars |
| Model identity v1 | 782 × 2 exposures | 0.17M | "OpenEuroLLM", in many languages |

**Training** `configs/sft_oellm_stage1_instruct_v1_leonardo.yaml`, `slurm/sft_oellm_stage1_instruct_v1_leonardo.sbatch`:
8 nodes × 4 A100, FSDP full shard (`accelerate_fsdp.yaml`), TRL SFTTrainer with pre-tokenized data
(`assistant_only_loss` flipped off by the wrapper because TRL rejects it for non-conversational data; the collator still
applies `assistant_masks`), BFD packing into 16K padding-free bins (no cross-document attention), liger fused
cross-entropy, bf16, gradient checkpointing; 1 epoch, LR 8e-6 cosine to 10 %, 3 % warmup, weight decay 0, grad-norm
1.0, seed 42; eval batch 1 (batch 8 OOMs on the 263K vocab). ~0.5M tokens per step.

**Run:** job `61558734`, 12 h 19 min, 3,153 local-h; train loss 0.508, token accuracy 0.857, 2.365B tokens.
Post job: FP32 → BF16 export with the template bundled, generation check (24 multilingual prompts).

**Result:** EUROPA Eval 0.600, Alignment 0.455. Identity data issue found afterwards (rows naming other assistants),
fixed in the published `birgermoell/openeurollm-instruct-sft-v1` (private, `d12df83`).

---

## 6. Stage 2 — Reasoning SFT

**Goal:** add an opt-in thinking mode without degrading default mode.

**Data** `data/stage2-mix-v2` = `birgermoell/openeurollm-reasoning-instruct-v1` (public, `9273ba4`) mixed by token
budget (builder `scripts/build_oellm_stage2_mix.py`): **372,792 reasoning rows / 1.738B tokens + 791,520 direct replay
rows / 599M tokens** (replay fraction 0.25), max 32,768 tokens. Per-source token budgets (excerpt): Nemotron multilingual
de/es/fr/it 75M each; OpenThoughts3 150M; Nemotron chat 100M; SmolTalk multilingual8 think 100M; translated Dolci 12
languages 45M each; Dolci-32B Persona/Think Precise IF 90M/60M; WildChat 100M; safety sets (WildJailbreak, WildGuardMix,
CoCoNot) all; plus multilingual traces, MedQA reasoning traces (MIT) and Swedish Kimi traces in full.
Known property: most source traces are English even under non-English prompts.

**Training** `configs/sft_oellm_stage2_reasoning_v1_leonardo.yaml`: from the Stage-1 BF16 export; 32K packing; LR 5e-6
cosine to 10 %; label guard `SFT_FORBID_LABEL_IDS=18`; otherwise as Stage 1.

**Run:** job `61618341`, 10 h 59 min, 2,811 local-h; train loss 0.711, token accuracy 0.794, 2.337B tokens.

**Result:** default mode EUROPA Eval 0.609, Alignment 0.605 (identity fixed). **Thinking mode is weak**: on the quick
reasoning eval 53.5 % of traces never close within 6K tokens and thinking accuracy (0.40) is below default (0.59).
A 16K-token rerun of the 305 unclosed prompts showed it is a **termination problem, not verbosity**: 81 % still never
close at 16K; of those, 52 % are verbatim loops. → Stage-3 replay keeps only traces ≤ 8K tokens; RL targets closure.

---

## 7. Stage 3 — Tool-use SFT

### 7.1 Dataset `birgermoell/openeurollm-tool-use-v1` (public, `e30b5449`)
**158,703 conversations** (148,285 English + 10,418 translated into 16 European languages), ~328M tokens, normalized
from Toucan-1.5M SFT, oellm-eu-tooluse, When2Call, Nemotron-SFT-Agentic-v2, ToolMind graphsyn, ToolACE (licenses in the
dataset card). ~30 % of rows teach *not* calling (ask for missing info 19.5K, cannot answer 13.2K, no call needed
11.2K, no tools 2.5K). Builder `scripts/build_openeurollm_tool_use.py` with adapters `scripts/tooluse_adapters.py`
(schema normalization, validation of names/arguments/results, category labels, literal-tag and meta-answer rejection);
relabel pass `scripts/relabel_tool_use.py`; translation `scripts/translate_tool_use.py` (Ollama `gemma4:31b`
sha256 `6316f062…`, `think: false`, user/system/final text only, arguments must survive verbatim).

### 7.2 Mix `data/stage3-mix-v1` (`scripts/build_oellm_stage3_mix.py`)
| Part | Rows | Tokens |
|---|---:|---:|
| Tool pool (per-source budgets: Toucan 70M, Nemotron tool calling 50M, ToolMind 25M, others all) | 112,877 | 219M |
| Stage-2 replay, reasoning (**traces ≤ 8,192 tokens only**) | 30,243 | 85M |
| Stage-2 replay, direct | 105,684 | 80M |
| Identity + safety (always kept) | 36,933 | 27M |
| Validation (tool, 0.5 % id-hash bucket) | 808 | 1.7M |

### 7.3 Training
`configs/sft_oellm_stage3_tooluse_v1_leonardo.yaml`: from Stage-2 BF16; template v2.1; max length 32,768; LR 3e-6
cosine to 10 %; `SFT_FORBID_LABEL_IDS=18,15,13`; save every 300 steps. Job `61746313`, 2 h 7 min, 397 steps,
541 local-h; train loss 0.717, eval loss 0.823 → 0.820 (no regression on replay).

### 7.4 Defect and repair: collapsed tag rows → v1.1
v1 closed every tool call with `<tool_call>` instead of `</tool_call>`. The data was verified clean (full scan: 160,990
supervised calls, all well-formed). Cause: the `lm_head` rows of IDs 11 and 12 start nearly identical, and AdamW moves a
weight by at most ~LR per step, so 397 steps at 3e-6 moved their difference only 0.018 → 0.025 (Stage 2's many think
tags at 5e-6 for ~2,250 steps had separated 17/18 to 0.15). Teacher-forced on held-out calls: closer top-1 **0 %**,
logit gap −0.3 (vs +14 for `</think>`).

Repair (`scripts/repair_tag_head_rows.py`, 1 GPU, 12 min): freeze everything, collect final hidden states at supervised
positions of 3,000 Stage-3 training rows (every tag target + 64 ordinary targets per row), and re-fit only `lm_head`
rows 11–18 with the exact full-vocabulary cross-entropy (the non-tag log-sum-exp is a per-position constant) plus an L2
pull to the original rows; 400 Adam steps. Held-out validation rows:

| Target | Top-1 before | after |
|---|---:|---:|
| `</tool_call>` | 0.0 % | 99.6 % |
| `<tool_call>` | 75.9 % | 98.8 % |
| `</think>` | 84.7 % | 100 % |
| tag wrongly top-1 at ordinary positions | 0.008 % | 0.093 % |

End to end (held-out tool check, 361 rows): valid JSON 0 → 99.5 %, valid arguments 0 → 98.1 %, call/no-call decision
50 → 87 %, think tags in default answers 6.9 → 0 %; thinking mode unchanged at equal budget (21/24 close).
**Check this for every future stage** (row distances of paired tags + teacher-forced tag top-1).

### 7.5 Results (v1/v1.1)
EUROPA Eval 0.616, Alignment 0.642. Quick reasoning (thinking): accuracy 0.402 → 0.435, unclosed 53.5 % → 48.9 %
(MATH-500 0.09 → 0.16, EU math 0.06 → 0.11); default mode flat (0.588 → 0.581).

---

## 8. Evaluation protocol

### 8.1 Harnesses
- **EUROPA Eval** (432 items, 36 languages, 12 capabilities) and **EUROPA Alignment** (576 items, 16 scenarios):
  HF backend, bf16, temperature 0, seed 42, reasoning off; judge `gemma4:31b` sha256 `6316f0629137…` via Ollama 0.32.8
  on Leonardo (blobs copied from the reference machine: the registry tag was re-pointed). Launcher
  `…/europa/europa_eval_stage1_leonardo.sbatch` with `MODEL`, `REV`, `NAME`; pre-download the model into
  `…/europa/hf-cache` on a login node (compute nodes are offline). ~2 h on 1 node.
- **Quick reasoning eval** `scripts/eval_reasoning_quick.py` (GSM8K 150, MGSM en/de/es/fr 200, MATH-500 100, oellm
  EU12 math 120): both modes, unfinished-trace rate, lengths. ~3 h on 1 node.
- **Generation check** `scripts/eval_stage1_generation_check.py`: 24 multilingual prompts, both modes; use
  `--max-new-tokens 4096` for thinking (the 768 default truncates traces).
- **Held-out tool check** `scripts/eval_tool_calling_heldout.py` (§7.4).
- **EUROPA Reasoning** (new, public: github.com/BirgerMoell/europa-reasoning): 24 parametric templates × 36 languages,
  default and reasoning mode, `\boxed{}` scoring, rubric judge for explanations, process metrics (trace closure, loops,
  budget exhaustion, reasoning length, trace script). Launcher `scripts/hpc/run_leonardo.sbatch` (vLLM target + Ollama
  judge on one node).

### 8.2 Gates
Stage acceptance: no think tags in default-mode answers; identity correct; EUROPA Eval ≥ previous stage; Alignment not
regressed; stage-specific checks (tool parse rate, trace closure).

### 8.3 Why our EUROPA Eval trails the reference SFT (0.616 vs 0.762)
The gap is format compliance and direct-mode arithmetic, not knowledge: numeric reasoning 0.06 vs 0.58, tool-recovery
(single letter) 0.49 vs 0.92, 122 malformed answers vs 38. We answer `B. At ten.` to "answer only with the letter" and
`18 - 7 = 11` to "answer only with the number". Root cause in the instruct data: of 9,685 strict-format user turns in
3.15M rows, 99 % of targets ignore the constraint (mostly Excellent-SFT zebra puzzles distilled with an explanation
after the answer). Remedy: format-compliance preference pairs in Stage 4 (§9.2); next instruct-data version rewrites or
drops those targets.

---

## 9. Stage 4 — DPO (in preparation)

### 9.1 Recipe: replicate the threeway run, fix tokenizer and template
Reference `birgermoell/oellm-9b-threeway-dpo-full-v1` (LUMI, 32 MI250X GCDs, 43 h): TRL DPO, sigmoid loss, β 0.1,
LR 5e-7 cosine, 3 % warmup, Adam (0.9, 0.999, 1e-8), weight decay 0, per-device batch 1, global batch 128, max length
2,048 with truncated pairs dropped, 1 epoch, seed 42, bf16, FSDP, gradient checkpointing. 1,441,547 train pairs /
15,824 validation pairs after filtering; held-out preference accuracy 0.878.

Our changes: pairs rendered with **our tokenizer and template v2.1** (default mode, no think tokens); policy and
reference = Stage-3 v1.1; every source pinned; offline pre-tokenization with guards; format-compliance pairs added.

### 9.2 Data
- `scripts/stage_oellm_dpo_v1.py` (login node `--prefetch-only`, then offline on `lrd_all_serial` via
  `slurm/build_oellm_dpo_v1_leonardo.sbatch`): HelpSteer3 `preference` @ `f6d14577` (sign of `overall_preference`,
  ties dropped, native validation: 36,299 / 1,920 / 2,257 ties), Dolci-Instruct-DPO-translated @ `a57841a6`
  (12 languages, every 100th row per language held out; counts identical to the reference, e.g. cs 116,249 / 1,175,
  en 243,752 / 2,463), UltraFeedback cleaned @ `770076f0` (duplicate user turn removed; every 100th row held out).
- `scripts/build_oellm_dpo_v1_pretok.py`: prompt / chosen / rejected ids exactly as TRL 0.29 computes them (prompt with
  generation prompt; completion = render(prompt + response) minus the prompt), template byte-identical to the model's
  bundled template, hashes recorded; drops empty responses, render errors, prefix mismatches, pairs over 2,048 tokens,
  and any pair containing IDs 11–18.
- Format-compliance pairs (to be built): strict-format prompts (number, letter, single word, JSON only, N lines, no
  explanation) in the 36 languages; chosen = verified bare answer, rejected = Stage-3 v1.1's own verbose/wrong answer;
  no EUROPA items; checks that unconstrained answers keep their length.

### 9.3 Training plan
`scripts/dpo_train.py` with a `DPOTrainer` subclass that accepts the pre-tokenized columns as is (TRL 0.29 always
re-tokenizes otherwise), template/tokenizer hash checks and a label guard; 8 nodes, two chained 24 h jobs with resume.
Estimated 19–28 h, 600–900 GPU-h (4.7–7K local-h); a ≤ 20 local-h smoke measures throughput first.
Gates: Alignment ≥ 0.85, Eval ≥ Stage 3 with malformed answers ≤ 40, over-refusal check, no length inflation or collapse,
tag-row check (§7.4).

---

## 10. Stage 5 — RL (oellm-rlvr + TMAX)

### 10.1 Environment on Leonardo
`$SCRATCH/oellm-rl`: TMAX (open-instruct fork, commit `3f80d370`) venv built with `uv sync --locked --no-dev` inside the
container (torch 2.10.0+cu128, vLLM 0.19.1, transformers 5.4.0, Ray 2.53.0, DeepSpeed 0.18.4) plus `beaker-py` (a
dev-group dependency imported at runtime). Control plane `oellm-rlvr` (configs, Slurm rendering, verifier patches via
`sitecustomize` `OELLM_PATCH_*`).

### 10.2 Thinking-mode probe (`configs/leonardo-grpo-math-oellm9b-think-probe.yaml`)
Question: can DAPO with a closed-trace-or-zero reward cut loops and raise accuracy?
- Start: Stage-3 v1.1 with the RL thinking-default template (`models/stage3-tooluse-v1-think`, HF revision `88875629`).
- Reward: MathVerifier on the final answer; `OELLM_PATCH_REQUIRE_THINK_CLOSE` returns an empty answer unless the trace
  has exactly one `</think>`, so a boxed answer inside a loop earns nothing.
- Data (`oellm-rlvr/scripts/leonardo_think_probe_data.sh`): 2,008 prompts in one pre-mixed parquet (1,008
  oellm-math-rlvr EU-language + 400 English at difficulty 1–3, 600 GSM8K train), pinned revisions.
- DAPO, β 0, LR 1e-6, 16 prompts × 8 samples, 6K responses, 48 updates; 2 nodes (4 ZeRO-3 learners, 4 vLLM engines).
- Leonardo-specific fixes found by canaries (each cost one short 2-node attempt):
  1. `beaker-py` missing (dev group) → installed.
  2. Ray placement pending forever: TMAX reserves 10 CPUs per learner GPU (40 per node) but booster nodes have 32 →
     `RAY_NUM_CPUS=64` (logical) in `scripts/ray_node.sh`.
  3. OOM at the weight sync: whole-model ZeRO-3 gather (~18 GB) on top of ~31 GB → `--gather_whole_model False`.
  4. OOM at the training step (263K-vocabulary logits) → `--deepspeed_offload_optimizer True` with a prebuilt
     `DeepSpeedCPUAdam` (`TORCH_EXTENSIONS_DIR`) and `OELLM_PATCH_POLICY_CPU_ADAM` (TMAX used torch AdamW for the policy
     even under offload, which DeepSpeed refuses).
- Evaluation before/after: quick reasoning eval (thinking accuracy and unclosed rate), EUROPA Reasoning process metrics.

### 10.3 Production RL plan
`docs/OELLM_STAGE5_RL_PLAN.md`: tiered pools (math/code verifiable first, agentic later), measured-difficulty bins,
selected-vs-candidate checkpoints, format-following rewards.

---

## 11. Published artifacts

| Artifact | Visibility | Revision |
|---|---|---|
| `birgermoell/oellm-9b-256k-stage1-instruct-v1` | private | `58e42ab4` |
| `birgermoell/oellm-9b-256k-stage2-reasoning-v1` | private | `2db72459` |
| `birgermoell/oellm-9b-256k-stage3-tooluse-v1` | private | `88875629` (v1.1); `0dc97efa` (v1, do not use for tools) |
| `birgermoell/openeurollm-instruct-sft-v1` (dataset) | private | `d12df83` |
| `birgermoell/openeurollm-reasoning-instruct-v1` (dataset) | public | `9273ba4` |
| `birgermoell/openeurollm-tool-use-v1` (dataset) | public | `e30b5449` |
| github.com/BirgerMoell/europa-reasoning (benchmark) | public | main |
| `qwen35-posttrain`, `oellm-rlvr` (code) | GitHub | main |

---

## 12. Lessons and pitfalls

1. **Template and tokenizer drift is the most expensive bug.** Pin both by sha256 in every data manifest and assert in
   every launcher. Never ship another run's tokenizer/template (the threeway release's template emits empty think blocks).
2. **New special tokens can stay collapsed** (§7.4). Low-LR SFT cannot separate rows that start equal; the loss curve
   hides it. Check paired-tag row distances and teacher-forced tag accuracy after every stage.
3. **Default-mode format discipline must be in the data**: if "answer only with X" targets add explanations, the model
   learns to ignore the instruction (§8.3).
4. **Moving reasoning into thinking mode removes it from default mode**: direct-mode multi-step arithmetic dropped;
   keep some short worked answers in direct replay.
5. **Thinking loops are a termination problem**: more tokens do not help (81 % of unclosed traces still unclosed at
   16K). Filter long traces from replay, reward closure in RL.
6. **Check evaluation budgets before calling a regression** (a 768-token generation cap made v1.1 look worse).
7. **TRL specifics**: `assistant_only_loss` is rejected for pre-tokenized data but `assistant_masks` still apply;
   eval batch must be 1 with a 263K vocab; DPO re-tokenizes unless the trainer is subclassed.
8. **Leonardo specifics**: glibc 2.28 host (build and run vLLM inside the container); offline compute nodes; login nodes
   kill heavy processes; 32-core nodes vs. Ray CPU reservations; 64 GB A100s need per-parameter weight sync and optimizer
   offload for 9B RL with 4 learners.
9. **Machine translation needs automatic script checks**: Gemma inserted CJK, Hangul, Thai, Arabic and Cyrillic
   fragments into 11 of 35 EUROPA Reasoning packs; greedy retries are deterministic (vary temperature, not only seed).
10. **Storage**: delete full checkpoints after export; archive to personal scratch, publish to HF.

---

## 13. Reproduction checklist

1. Environment (§1): container, tokenizer and base staged; `step ssh login`; HF token via stdin only.
2. Stage 1: `sbatch slurm/build_oellm_stage1_mix_leonardo.sbatch` → check manifest → `sbatch
   slurm/sft_oellm_stage1_instruct_v1_leonardo.sbatch` (resubmit to resume) → post/export job → EUROPA evals.
3. Stage 2: build reasoning dataset and mix (`slurm/build_openeurollm_reasoning_instruct_leonardo.sbatch`,
   `slurm/build_oellm_stage2_mix_leonardo.sbatch`) → `sbatch slurm/sft_oellm_stage2_reasoning_v1_leonardo.sbatch` →
   export → quick reasoning eval both modes.
4. Stage 3: `slurm/build_openeurollm_tool_use_leonardo.sbatch` (+ `translate_tool_use_leonardo.sbatch`) →
   `slurm/build_oellm_stage3_mix_leonardo.sbatch` → `sbatch slurm/sft_oellm_stage3_tooluse_v1_leonardo.sbatch` →
   `slurm/post_oellm_stage3_tooluse_v1_leonardo.sbatch` → **tag-row repair** (`scripts/repair_tag_head_rows.py`) →
   held-out tool check, generation checks (4,096 tokens), EUROPA evals → upload.
5. Stage 4: prefetch (`python scripts/stage_oellm_dpo_v1.py --prefetch-only` on a login node) →
   `sbatch slurm/build_oellm_dpo_v1_leonardo.sbatch` → format pairs → DPO smoke → full DPO (two chained jobs).
6. Stage 5: `$SCRATCH/oellm-rl` env (`sync.sh` inside the container), probe data script, `oellm-rlvr render-slurm`,
   canary (2 updates) → probe.

## 14. Changelog
- 2026-10-20: took over from Codex; template v2; Stage 1 launched.
- 2026-10-21: Stage 1 done; identity fix; reasoning dataset; Stage 2 done; tool dataset and template v2.1.
- 2026-10-22: Stage 3 done; tag-row repair → v1.1; storage cleanup; EUROPA Reasoning published; EUROPA Eval gap traced
  to strict-format SFT targets; DPO data staging; RL canaries on Leonardo.
