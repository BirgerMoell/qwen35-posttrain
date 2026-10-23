# OpenEuroLLM 9B post-training plan: instruct → reasoning → tools → DPO → RL

**Status 2026-10-23 (evening).** This is the master plan: what the post-training pipeline is for, every decision taken
and why, what is done, what runs now, and what comes next, with budgets and gates. It is written so that the whole
pipeline can be reproduced and continued by someone new.

| Document | Role |
|---|---|
| this file | the plan: goals, decisions, stage plans, budget, gates, open questions |
| [`OELLM_POST_TRAINING_PLAYBOOK.md`](OELLM_POST_TRAINING_PLAYBOOK.md) | the record: exact data, configs, hashes, job IDs, costs, results, pitfalls, reproduction commands |
| [`OELLM_RL_RUN_PROGRAM.md`](OELLM_RL_RUN_PROGRAM.md) | Stage 5 run by run: questions, run cards, results, stop rules, merging policy |
| [`OELLM_STAGE5_RL_PLAN.md`](OELLM_STAGE5_RL_PLAN.md) | Stage 5 data pools, rewards, difficulty windows, environment tiers |
| `oellm-rlvr` repo | RL control plane (configs, Slurm rendering, verifier patches, data scripts) |

---

## 1. Goal and principles

Turn the OpenEuroLLM 9B 256K base (`openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`) into an assistant that
answers well in the 36 OpenEuroLLM languages, has an opt-in reasoning mode that finishes its traces, calls tools
reliably, follows format instructions, is safe, and is measurably better at verifiable tasks after RL.

Principles that every stage follows:
1. **Selected vs candidate**: a stage starts from the previous stage's *selected* checkpoint; a candidate is promoted
   only if its paired evaluations pass the gates (§8). Otherwise the next stage starts from the previous selection.
2. **One tokenizer, our templates**: tokenizer `tokenizer.json` sha256 `5123f74b…`, templates v2 (Stages 1–2) and v2.1
   (Stage 3 on) pinned by sha256 in every data manifest and asserted by every launcher. Never another run's tokenizer
   or template (the threeway DPO release ships a broken template).
3. **Pre-tokenize offline, reject never truncate**, label guards in every trainer (no supervised `<think>`, tool
   responses or tool lists), manifests with per-source counts.
4. **Every job above ~20 GPU-h gets a local-h estimate before submission**; one question per experiment.
5. **Both modes stay trained**: default mode must never emit think tags; thinking mode must close its trace.
6. **Models private** until evaluation and license review; datasets and benchmarks public.

## 2. Status dashboard

| Stage | Status | Checkpoint / job | Cost (measured) | Key result |
|---|---|---|---:|---|
| 1 Instruct SFT | ✅ done | `birgermoell/oellm-9b-256k-stage1-instruct-v1` `58e42ab4` | 3,276 | EUROPA Eval 0.600, Alignment 0.455 |
| 2 Reasoning SFT | ✅ done | `…-stage2-reasoning-v1` `2db72459` | 2,811 | Eval 0.609, Align 0.605; thinking mode loops (53 % unfinished) |
| 3 Tool-use SFT | ✅ done | `…-stage3-tooluse-v1` **v1.1 `88875629`** (tag-row repair) | 541 | Eval 0.616, Align 0.642; tool JSON 99.5 %, decision 0.87 |
| 4 DPO | ⏳ running/queued | step 500 of 11,264 saved; resume job 61922663 waiting for 8 nodes | 1,035 so far | preference accuracy 0.80 at step ~400 |
| 5 RL | 🔬 ablations | RL-0, RL-0b, A2a done; A1 language and A5 code running/queued | 926 so far | thinking accuracy 0.435 → 0.53, unfinished 49 % → 34 % |

Budget: **9.34K of ~50K local-h** spent since 2026-10-20 (`sacct`, own jobs; account total 400K is shared).

## 3. Decisions log (dated, with reasons)

| Date | Decision | Why |
|---|---|---|
| 10-20 | Two-stage SFT with a hard `enable_thinking` switch (template v2); `<think>` always prompt-side | the model never learns to open a think block; default mode stays clean |
| 10-21 | Stage 2 mix by token budget, 25 % direct replay | keep default mode while adding reasoning |
| 10-21 | Stage 3 = tool SFT with ~30 % "do not call" rows, translated slice in 16 languages; template v2.1 | When2Call lesson: over-calling; tool sources are English |
| 10-22 | Repair collapsed `lm_head` rows of tag tokens 11–18 → Stage-3 **v1.1** (v1 never emitted `</tool_call>`) | low-LR SFT cannot separate rows that start equal; repair is cheap and verified |
| 10-22 | Stage 4 = replicate the threeway DPO run (1.44M pairs) with our tokenizer/template, **full size** | reference run lifted EUROPA Eval 0.762 → 0.840 |
| 10-22 | Add multilingual **format-compliance** DPO pairs | EUROPA Eval gap traced to strict-format SFT targets (99 % ignore "answer only with …") |
| 10-23 | DPO at 2 pairs per GPU per micro-step | 1 pair/GPU was FSDP all-gather bound (25.3 s/step ⇒ ~20K local-h) |
| 10-23 | RL ablations on Stage-3 v1.1 in parallel with DPO; RL program with gates and stop rules | RL work does not wait for DPO; one question per run |
| 10-23 | Merging policy: checkpoint averaging; Stage-3→DPO interpolation as DPO fallback; never with v1 | cheap insurance against DPO hurting thinking/tool use |
| 10-23 | Keep the non-stop penalty (−0.5 for unfinished traces) in RL | A2a: fewest unfinished traces, best MATH-500 |
| 10-23 | **Reasoning-language policy pending A1**: answers must be in the user's language (hard rule); the trace language is decided by A1's accuracy cost | RL-0 switched EU-language traces to English (24.8 % → 1.1 % in prompt language) |
| 10-23 | Code track: EvalPlus baseline + single-turn code RL (A5) with a contained execution server | only ~7K code-reasoning SFT rows; code is verifiable |
| 10-23 | oellm-eval (project harness, v0.3.0) at stage boundaries; comparison set: our base, Neonkraft SFT, threeway DPO, Apertus-8B-Instruct, Qwen3-8B | project comparability and knowledge-retention checks |

## 4. Stages 1–3 (done) — summary

Full details, data tables and reproduction commands are in the playbook §5–7.

- **Stage 1 Instruct SFT**: 3.14M rows / 2.37B tokens (OpenEuroLLM-Excellent-SFT direct rows, translated Dolci in 12
  languages, Aya, identity), 16K packing, LR 8e-6, 1 epoch, 8 × 4 A100, 12 h 19 min.
- **Stage 2 Reasoning SFT**: 373K reasoning rows / 1.74B tokens + 792K direct replay / 0.60B, 32K packing, LR 5e-6,
  11 h. Thinking mode learned but loops: a termination problem (81 % of traces unclosed at 6K stay unclosed at 16K).
- **Stage 3 Tool-use SFT**: `birgermoell/openeurollm-tool-use-v1` (158.7K conversations, 16 languages) + Stage-2 replay
  (reasoning traces ≤ 8K only), LR 3e-6, 397 steps, 2 h 7 min; then the tag-row repair (v1.1).

Known weaknesses entering Stage 4: format compliance in default mode (EUROPA numeric 0.06, 122 malformed answers),
thinking-mode termination, reasoning traces mostly English, Basque and Georgian weakest on knowledge benchmarks.

## 5. Stage 4 — DPO (in progress)

**Recipe** (replication of `birgermoell/oellm-9b-threeway-dpo-full-v1`, only tokenizer/template changed):
TRL 0.29 DPO, sigmoid loss, β 0.1, LR 5e-7 cosine, 3 % warmup, global batch 128, max length 2,048 (longer pairs
dropped at build time), 1 epoch, seed 42, bf16 + FSDP full shard, gradient checkpointing; policy and reference =
Stage-3 v1.1.

**Data (built 10-23, job 61856512):** HelpSteer3 `preference` @ `f6d14577` (ties dropped), Dolci-Instruct-DPO-translated
@ `a57841a6` (12 languages, every 100th row held out), UltraFeedback-cleaned @ `770076f0` → **1,441,722 train /
15,827 validation pairs** (reference 1,441,547 / 15,824). Pre-tokenized with our tokenizer and template v2.1 in default
mode; pairs containing tag IDs 11–18 dropped.

**Trainer:** `scripts/dpo_train_pretok.py` (TRL subclass that trains on the pre-tokenized ids; manifest, template and
tokenizer hash checks; vectorized shard scan; per-batch tag guard), config `configs/dpo_oellm_stage4_v1_leonardo.yaml`,
launcher `slurm/dpo_oellm_stage4_v1_leonardo.sbatch` (self-queued continuation jobs, resume from the latest
checkpoint, smoke mode with a resume test).

**Run so far:** 2-node smoke passed (incl. resume). Full run at 1 pair/GPU reached step 500 (checkpoint saved, 136 GB)
at 25.3 s/step — all-gather bound, ~79 h / ~20K local-h for the epoch — and was stopped. It resumes from step 500 at
**2 pairs/GPU, grad-accum 2** (same global batch and recipe), job 61922663, queued since 10-23 morning (≈10K jobs pending
on the partition). Health at step ~400: loss 0.43, preference accuracy 0.80, margin 1.2.
**Estimate:** 35–45 h on 8 nodes, ~9–11K local-h for the full epoch (to be replaced by the measured speed after the
first 30 min of the resumed job). The user approved the full set (10-23).

**Still to do in Stage 4:**
1. **Format-compliance pairs** (part of the plan, not yet built): strict-format prompts (number, letter, one word,
   JSON-only, N lines, no explanation) in the 36 languages, EUROPA items excluded; chosen = verified bare answer,
   rejected = Stage-3 v1.1's own verbose/wrong answer (vLLM, ~2–4 GPU-h). They go into a short follow-up DPO pass on the
   Stage-4 checkpoint (or the next DPO version), since the main run is already training.
2. **Gates** (§8) on the final export; also the tag-row check and the thinking-mode/tool checks, because all DPO pairs
   are default-mode and tool-free.
3. **Merging fallback**: average of the last 2–3 checkpoints vs the best single; if thinking or tools regress,
   interpolate Stage-3 v1.1 → DPO at α ∈ {0.5, 0.75, 1.0} before falling back to Stage 3.
4. Export BF16 with the v2.1 template, upload private, copy off `$SCRATCH` (40-day purge).

## 6. Stage 5 — RL

Run-by-run detail in [`OELLM_RL_RUN_PROGRAM.md`](OELLM_RL_RUN_PROGRAM.md); pools and rewards in
[`OELLM_STAGE5_RL_PLAN.md`](OELLM_STAGE5_RL_PLAN.md).

**Stack:** `oellm-rlvr` + TMAX (open-instruct fork `3f80d370`), DAPO, β 0, 2 nodes per run (4 ZeRO-3 learners with
CPU-offloaded `DeepSpeedCPUAdam`, 4 vLLM engines); qualified on Leonardo after six canaries (fixes in playbook §10).
Measured cost: ~2.2 local-h per 128 rollouts at ~3.8K tokens; a 48-update ablation ≈ 50–80 local-h.

**Ablation results so far** (quick reasoning eval, held out, thinking mode):

| Run | Change | Accuracy | Unfinished | MATH-500 | EU math | Trace in prompt language |
|---|---|---:|---:|---:|---:|---:|
| Stage-3 v1.1 | — | 0.435 | 48.9 % | 0.160 | 0.108 | 24.8 % |
| RL-0 | math DAPO, unclosed ⇒ 0 | 0.512 | 39.3 % | 0.140 | 0.308 | 1.1 % |
| RL-0b | RL-0 + 48 updates | 0.532 | 37.4 % | 0.180 | 0.367 | 0.4 % |
| A2a | non-stop penalty −0.5 | 0.523 | 34.0 % | 0.210 | 0.300 | 1.5 % |

Default-mode accuracy unchanged in all (0.575–0.593 vs 0.581). Running: **A1** (language-consistency reward on the
EU-language rows) and **A5** (single-turn code RL, 2,004 `oellm-code-rlvr` problems, contained execution server).
Still to build: A3 (default-mode strict-format/numeric prompts in RL, per-row mode switch), A4 (single-turn tool reward).

**Production sequence** (from the selected Stage-4 checkpoint): W1 warm-up (4K responses, math + exam-MCQ + strict
format + light IF) → W2 core (8K, + harder math, code, IF, tool calls) → W3 frontier (16K, hardest pools) → T1/T2 tool
RL (single-turn, then multi-turn with SkyRL) → F final selection. Each window: difficulty profiling on the current
checkpoint (pass@8 bins), realized-mix logging, checkpoints every 16 updates, gates before promotion, stop rules
(entropy collapse, length blow-up, > 25 % at length cap, < 20 % mixed groups, default-mode think tags, language drift,
trace legibility).

**Reasoning-language policy:** hard rule — the final answer is in the user's language. Trace language: if A1 shows a
cost ≤ 2 points, W1–W3 use the language-consistency reward; otherwise traces may be in English but must stay legible
natural language (legibility monitored: detector confidence, glued-word share, perplexity under the Stage-3 model),
and native reasoning is revisited after SFT on good native traces.

## 7. Code track

- Data so far: Stage 1 ~120K code rows (~7 % of tokens, English); Stage 2 only ~7.2K code-reasoning rows.
- Baseline: HumanEval+ / MBPP+ (EvalPlus 0.3.1) in both modes for Stage-3 v1.1 and RL-0 (`scripts/eval_code_evalplus.py`,
  `slurm/eval_code_evalplus_leonardo.sbatch`); model code is scored in a contained container (no home or project
  mounts). LiveCodeBench and multilingual-instruction code tasks are next.
- RL: A5 (above); afterwards easy code in W2, harder (DeepCoder-type) in W3. If pass@8 is near zero on a pool, a small
  code-reasoning SFT top-up comes first.
- Agentic coding (terminal/SWE, Tier C) only after the main windows and an SFT warm-start on agent trajectories.

## 8. Evaluation protocol and gates

| Harness | What | When | Cost |
|---|---|---|---|
| EUROPA Eval (432 items, 36 languages) + Alignment (576) | capabilities incl. strict format; safety/over-refusal | every stage | ~2 h, 1 node |
| Quick reasoning eval (570 items, both modes) | accuracy, unfinished rate, length, per language | every RL run, every stage | ~1.4 h, 1 node |
| Trace-language audit (lingua) | share of traces in the prompt language | every RL run | CPU |
| EUROPA Reasoning (864 items, 36 languages) | reasoning accuracy + process metrics (closure, loops, budget, trace script) | stage boundaries, RL windows | ~1 node-h per mode |
| Held-out tool check (361 rows) | call decision, JSON validity, argument validity, think leaks | Stage 3+, after DPO, tool RL | short |
| Code: EvalPlus (HumanEval+ 164, MBPP+ 378) | pass@1 base/plus, both modes | baseline, after code RL, stage boundaries | ~1 node-h |
| **oellm-eval v0.3.0** (Belebele, Global-MMLU, INCLUDE, MultiBLiMP, Global-MGSM) | knowledge/language retention, project comparability | stage boundaries | ~180 local-h per model |
| Tag-row check | teacher-forced top-1 of tag tokens 11–18 | after every training stage | minutes |

**Gates** (all stages): no think tags in default-mode answers; identity correct; EUROPA Eval ≥ previous stage;
Alignment not regressed (Stage 4: ≥ 0.85); oellm-eval within noise of the previous stage; stage-specific checks (tool
parse rate, trace closure, malformed answers ≤ 40 after DPO; RL windows: target holdouts up on a paired test, no
retention regression, closure targets). Selection always on frozen holdouts, never on training reward.

**Reference comparisons:** public oellm-eval scores (`openeurollm/prelude-base-eval-scores`) for Prelude, datamix,
Apertus-8B, Olmo-3-7B, Llama-3.1-8B, Qwen3-8B-Base, Qwen3.5-9B-Base (playbook §8.1). Stage-3 v1.1 leads all European and
open base models in the file on Belebele/Global-MMLU/INCLUDE and trails Qwen3/3.5 by 5–9 points. Own runs (10-23,
array 61949325): our base checkpoint, Neonkraft SFT, threeway DPO (with our tokenizer), Apertus-8B-Instruct, Qwen3-8B.

## 9. Budget

| Block | Spent (measured) | Remaining plan (estimate) |
|---|---:|---:|
| Stages 1–3 (SFT, translation, data, exports) | ~7.1K | — |
| Stage 4 DPO | 1.0K | ~9–11K (full epoch at the new speed) + format-pair pass ~0.3K |
| Stage 5 RL ablations | 0.9K | ~0.3K (A1, A5, A3/A4) |
| Stage 5 production (W1–W3, T1/T2, final) | — | ~12–14K |
| Evaluations (all harnesses, comparison models) | ~0.3K | ~2K |
| **Total** | **9.34K** | **~24–28K → ~34–37K of the 50K working budget** |

Every job above ~20 GPU-h gets an estimate first; the DPO estimate is replaced by the measured speed once the resumed
job runs.

## 10. Risks and open questions

| Risk / question | Mitigation / decision point |
|---|---|
| DPO queue wait (8 nodes on a full partition) | keep queued; if it waits > 24 h, consider 4 nodes (better per-GPU efficiency, longer wall time) |
| DPO weakens thinking mode or tool use (all pairs default-mode, tool-free) | post-DPO gates; Stage-3 → DPO interpolation; T1 tool RL |
| RL drifts traces to English / less legible text | A1 decision; language and legibility stop rules; answer-language hard rule |
| Small, easy RL pools saturate (RL-0b flattening) | W1–W3 use larger, harder pools with measured difficulty |
| Code RL signal too sparse | A5 mixed-group rate; SFT top-up if needed |
| Strict-format habit relearned by a future SFT | rewrite/drop the 9.4K non-compliant targets in the next instruct-data version |
| oellm-eval container gaps (PolyMath, Macedonian INCLUDE) | run PolyMath from our own venv; report upstream |
| `$SCRATCH` 40-day purge | copy selected exports to work storage and HF (private) |

## 11. Next steps (in order)

1. Collect A1 (language), A5 (code) and the code baseline; run the trace-language audit; decide the language policy.
2. Collect the oellm-eval comparison (5 models) and add it to the playbook table.
3. DPO resume starts → record measured speed and cost → run to the end with self-queued continuations.
4. Build the format-compliance pairs while DPO trains; build A3 (default-mode RL rows) and A4 (tool reward).
5. Post-DPO: export, gates, merging checks, select the Stage-4 checkpoint.
6. Profile W1 pools on the Stage-4 checkpoint → W1 → W2 → W3, each gated; T1 inside W2 if A4 passes.
7. SkyRL qualification on Leonardo for T2 (multi-turn tools with error recovery, multilingual Wikipedia search).
8. Final selection, release evaluation (fresh-seed EUROPA Reasoning variant), model card, private upload.

## 12. Release hygiene (all stages)

Pinned source revisions, per-row source and license, deterministic selection, rejection reports, pre-tokenized data
equal to the audited tokens, private model uploads until evaluation and license review, one template file per stage
with tests proving existing data renders identically, HF token only via stdin, no personal paths or account names in
public repositories.
