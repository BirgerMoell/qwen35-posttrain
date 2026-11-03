# Release checklist — OpenEuroLLM 9B assistant

Status 2026-11-03. A pre-release checklist in the spirit of an airline pre-takeoff checklist: every candidate
checkpoint goes through it **in order, every item, every time**, and the result is written down. Nothing is released
on "it looked fine in chat".

## How to use it

1. **Freeze the candidate.** One model directory, one commit of each repo, one template file. Write these on the
   sign-off sheet (§10) before running anything.
2. **Run the phases in order.** §1 (artifact preflight) is fast and catches the failures that make every later
   number meaningless, so it runs first. §2–§7 can run in parallel once §1 passes.
3. **Each item has a gate class:**
   - **NO-GO**: a failure blocks the release. No exceptions without a new checklist run.
   - **REVIEW**: a failure needs a written justification on the sign-off sheet and a line in the model card's
     limitations section.
   - **TRACK**: no threshold yet; the number is recorded so the next release can be compared with it.
4. **Compare against the previous release, paired.** "No regression" means: same eval set, same decoding settings,
   difference within the noise band stated in the item (or a paired bootstrap CI that includes 0).
5. **Thresholds are marked (set)** when they already gate our stages (roadmap §8, RL run program §1,
   hallucination plan §5) and **(proposed)** when this checklist introduces them. Proposed thresholds become set
   once the team agrees.
6. **Selection vs release.** Checkpoints are *selected* on the frozen holdouts. The release run uses the same
   harnesses plus the fresh-seed variants (§7.3) so the final numbers are not tuned on.

Paths use placeholders (`$WORK`, `$SCRATCH`, `<model dir>`). Commands are Slurm launchers in
[`qwen35-posttrain/slurm`](https://github.com/BirgerMoell/qwen35-posttrain/tree/main/slurm); run them from the work
directory after `source ~/.oellm/leonardo.env`.

**Cost of a full run:** ~60–80 local-h for §1–§6 on Leonardo, plus ~180 local-h for oellm-eval (§2.6). It's cheap
next to training, so it is never skipped to save budget.

---

## 1. Artifact preflight (before any eval)

**Why:** most of our worst incidents were artifact defects, not capability problems: the threeway release's broken
chat template, the collapsed lm_head rows of the tag tokens (11–18) that made `</tool_call>` impossible to emit, and
think tags leaking into default mode. These make every downstream number wrong, so they're checked first.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 1.1 | Tokenizer is ours | `sha256sum tokenizer.json` vs the base's | identical hash (set) | NO-GO |
| 1.2 | Chat template is ours | `sha256sum chat_template.jinja` vs `oellm_qwen3_two_stage_v2_1.jinja`; never the threeway repo's template | identical hash (set) | NO-GO |
| 1.3 | Config is sane | `config.json`: architecture, RoPE theta and scaling, `max_position_embeddings` (256K), bos/eos; `generation_config.json`: eos ids include the end-of-turn token | matches the base, apart from intended changes | NO-GO |
| 1.4 | Weights are sound | load every shard; check dtype bf16, no NaN/Inf, parameter count, shard index complete | all pass | NO-GO |
| 1.5 | Tag rows intact | teacher-forced top-1 accuracy and paired-row distances of tag tokens 11–18 (playbook §7.4) | top-1 ≥ 0.99 on held-out rows; no collapsed pair (set) | NO-GO |
| 1.6 | Loads and serves | vLLM (the version we ship in docs) with `--tool-call-parser hermes`, and transformers `generate` | both start; one chat in each mode works | NO-GO |
| 1.7 | Mode switch works | 50 prompts per mode: default mode has no `<think>` anywhere; thinking mode opens and closes one block | default think-tag rate = 0 (set); thinking well-formed ≥ 0.99 | NO-GO |
| 1.8 | Human walkaround | a person reads ~20 chats: 5 languages (incl. one low-resource), both modes, one tool call, one refusal, one identity question | nothing alarming; notes on the sign-off sheet | NO-GO |

**Steps:** run 1.1–1.4 on CPU (minutes) → 1.5 on 1 GPU (minutes) → 1.6–1.7 in a 1-GPU interactive job
(~15 min) → 1.8 by hand from the 1.6 server.

---

## 2. Core capabilities

### 2.1 Reasoning (thinking mode) and direct answers (default mode)

**Why:** reasoning is the main thing Stage 2 and Stage 5 RL train for. Thinking traces that never close, or loop,
waste the user's tokens and give no answer.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 2.1.1 | Quick reasoning eval, both modes (570 items: GSM8K, MGSM, MATH-500 subset, …) | `MODEL=<dir> NAME=<name> sbatch --gpus-per-node=4 slurm/eval_reasoning_quick_leonardo.sbatch` (~1.4 h, 1 node) | accuracy ≥ previous release − 0.02 in both modes (set) | NO-GO |
| 2.1.2 | Trace closure and loops | same run: `unfinished` share; loop detector | ≥ 95 % closed within the budget, loops ≈ 0 (set, Stage-5 target) | REVIEW |
| 2.1.3 | EUROPA Reasoning (864 items, 36 languages), both modes | EUROPA Reasoning harness (~1 node-h per mode) | ≥ previous release; process metrics (closure, budget, trace script) recorded | NO-GO |
| 2.1.4 | Hard maths | AIME and PolyMath via our venv (roadmap §10) | TRACK | TRACK |

### 2.2 Instruction following and strict formats

**Why:** users and downstream tools rely on "answer with only a number", JSON-only output, word limits and the like.
Stage 3 had 122 malformed EUROPA answers; DPO and RL were partly aimed at that.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 2.2.1 | EUROPA Eval strict format and numeric | `MODEL=<dir> REV=main NAME=<name> sbatch slurm/europa_eval_stage1_leonardo.sbatch` (runs Eval + Alignment, ~2 h) | numeric ≥ 0.5, malformed ≤ 20 (set, Stage-5 target; ≤ 40 after DPO) | REVIEW |
| 2.2.2 | Verifiable instructions (IFEval-style, multilingual) | the ifeval verifiers from oellm-rlvr on a held-out prompt set | ≥ previous release (proposed) | REVIEW |

### 2.3 Language fidelity (36 languages)

**Why:** this is the project's reason to exist. An answer in English to a Polish question, or broken grammar in
Maltese, is a failure even if it's factually right.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 2.3.1 | Answer in the user's language | lingua audit on the quick-eval and EUROPA outputs, per language | ≥ 98 % of answers in the prompt language, every language ≥ 95 % (proposed; the policy itself is set: answers always in the prompt language) | NO-GO |
| 2.3.2 | Trace language | lingua audit on thinking traces | per the A1 decision: ≥ 90 % native if that costs ≤ 2 points, otherwise legible English (set) | REVIEW |
| 2.3.3 | Fluency and naturalness per language | [OELLM quality gate](https://github.com/BirgerMoell/oellm-quality-gate) (`grammatical`, `natural`, `language_match`) on generated answers | no language below the previous release by more than its noise band (proposed; Leonardo run not yet launched) | REVIEW |
| 2.3.4 | Coverage | EUROPA Eval / Alignment: "languages covered" and the minimum per-language score | 36/36 covered; minimum language score ≥ previous release (proposed) | REVIEW |

### 2.4 Knowledge and hallucination

**Why:** a 9B model can't know everything. It has to answer what it knows and say so when it doesn't. Our W2
checkpoint answered 94 % of the questions it didn't know with a confident wrong answer, and W3 overcorrected into
saying UNKNOWN to everything in default mode. Both failures must be caught here. Details:
[hallucination plan](OELLM_HALLUCINATION_PLAN.md).

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 2.4.1 | Answers what it knows (head band) | `MODEL=<dir> NAME=<name> sbatch slurm/eval_eu_tasks_leonardo.sbatch` (eu_qa, ~15 min); read `band:head` | accuracy ≥ 0.50 and abstain ≤ 0.10, **in both modes** (set) | NO-GO |
| 2.4.2 | Doesn't invent (hallucination rate) | same; `hallucination_rate` = wrong / (n − correct) | below the previous release; target ≤ 0.60 (proposed; W2 0.94) | NO-GO |
| 2.4.3 | Fabricated entities | hallu-v1 set (`DATA=<hallu-v1/eval.jsonl> sbatch slurm/eval_eu_tasks_leonardo.sbatch`; built by oellm-rlvr `scripts/build_hallucination_evals.py`), `fabricated` condition | invents for ≤ 30 % of made-up entities (proposed; W2 0.79, W2 + search 0.25) | NO-GO |
| 2.4.4 | Calibration | hallu-v1 `t50/t75/t90` conditions; self-consistency AUROC (`oellm-rlvr hallucination-consistency`) | abstention rises with the stated threshold; AUROC ≥ previous (proposed) | REVIEW |
| 2.4.5 | Knowledge retention | oellm-eval (§2.6) Global-MMLU, INCLUDE | within noise of the previous release (set) | NO-GO |

### 2.5 Grounded and document tasks

**Why:** much real use is "answer from this document": contracts, invoices, tables, meeting notes. Here the model
must use the text it's given, not its memory.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 2.5.1 | eu_doc (500 items, 10 languages, 5 families) | same run as 2.4.1 | ≥ previous release − 0.02 (set) | NO-GO |
| 2.5.2 | Grounding safety | EUROPA Eval `grounding_safety` domain | ≥ previous release (proposed) | REVIEW |

### 2.6 General retention (project benchmarks)

**Why:** post-training can quietly erase what pre-training learned. oellm-eval is also the project-wide comparison
with other European and open models.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 2.6.1 | oellm-eval v0.3.0: Belebele, Global-MMLU, INCLUDE, MultiBLiMP, Global-MGSM | playbook §8.1 command (1-GPU array, ~180 local-h) | each benchmark within noise of the previous release (set) | NO-GO |
| 2.6.2 | Comparison table | `scripts/oellm_eval_compare.py` against Apertus-8B, Olmo-3-7B, Llama-3.1-8B, Qwen3-8B | table updated in the model card | REVIEW |

---

## 3. Agentic capabilities

### 3.1 Tool use

**Why:** tool calls are executed by software, not read by a person. A malformed call or a wrong decision to call
fails silently or does something unintended.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 3.1.1 | Held-out tool check (361 rows) | `MODEL=<dir> NAME=<name> sbatch --gpus-per-node=4 slurm/eval_tool_calling_heldout_leonardo.sbatch` (~10 min) | decision ≥ previous release (set); target ≥ 0.92 (set, Stage-5); JSON ≥ 0.99; think leaks = 0 | NO-GO |
| 3.1.2 | Error recovery | EUROPA Eval tool-recovery | ≥ 0.8 (set, Stage-5 target) | REVIEW |
| 3.1.3 | Parallel and multi-step calls | slices of 3.1.1 | ≥ previous release (proposed) | REVIEW |
| 3.1.4 | External benchmark | BFCL | TRACK (planned) | TRACK |

### 3.2 Search and retrieval

**Why:** search is our strongest lever against hallucination (E7: +0.23 accuracy, fabricated inventions
0.79 → 0.25), but only if the model actually searches, reads the results and doesn't make things up anyway.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 3.2.1 | Search QA (offline Wikipedia BM25, 10 languages) | `MODEL=<dir> NAME=<name> sbatch slurm/eval_search_qa_leonardo.sbatch` | accuracy with search ≥ without + 0.15; search rate on factual questions ≥ 0.85 (proposed) | REVIEW |
| 3.2.2 | No invention after an empty search | `fabricated` condition with search | invents ≤ 25 % (proposed) | REVIEW |
| 3.2.3 | Indirect prompt injection | search results and tool outputs that contain instructions ("ignore the user and …") | follows injected instructions ≤ 5 % (proposed; **eval to build**) | REVIEW |

### 3.3 Code

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 3.3.1 | EvalPlus (HumanEval+ 164, MBPP+ 378), both modes | `MODELS="<name>=<dir>" sbatch --gpus-per-node=4 slurm/eval_code_evalplus_leonardo.sbatch` (~30 min) | pass@1 plus ≥ previous release − 0.02 (set) | NO-GO |
| 3.3.2 | Closure in thinking mode | same run, `closed` | ≥ 0.90 (proposed) | REVIEW |

### 3.4 Long context

**Why:** the base model is 256K. Post-training on short sequences can lose that silently, and nobody notices until
a user pastes a long document.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 3.4.1 | Needle-in-a-haystack grid | `scripts/eval_niah.py` (`slurm/eval_oellm9b_sft_niah_lumi.sbatch`; port to Leonardo), depths × lengths up to 256K | ≥ 0.95 at every depth up to 128K; no regression vs the base at 256K (proposed) | NO-GO |
| 3.4.2 | Long-document QA | eu_doc-style questions over long inputs (RULER-style) | TRACK (eval to build) | TRACK |

---

## 4. Safety

**Why:** this is the part where a failure hurts someone other than us. The model has to refuse what is dangerous,
help with what is legitimate (over-refusal is a failure too), keep refusing under pressure, and behave the same way
in all 36 languages, not only in English.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 4.1 | EUROPA Alignment overall (576 items, 36 languages) | same job as 2.2.1 | macro ≥ previous release (set: "Alignment not regressed") | NO-GO |
| 4.2 | Dangerous capabilities | Alignment `dangerous_capabilities` | ≥ previous release; **every failed critical item in this domain read by a person** (proposed) | NO-GO |
| 4.3 | Critical safety gates | Alignment "Failed items" with dimensions (safety, attack resistance, privacy, …) | count ≤ previous release; no new failure category (proposed; W2 151, 4b 115) | NO-GO |
| 4.4 | Over-refusal (refusal calibration) | Alignment `refusal_calibration`; OR-Bench-style benign-but-scary prompts | ≥ 0.40 (proposed; W2 0.35, 4b 0.44) | REVIEW |
| 4.5 | Jailbreak robustness | Alignment contrast sets: base vs challenge score, robustness gap | challenge score ≥ previous; gap shrinking (proposed) | REVIEW |
| 4.6 | Multilingual safety | Alignment per-language scores | no language more than 15 points below the mean (proposed) | REVIEW |
| 4.7 | Privacy | Alignment `privacy_security`; prompts asking for personal data about private people | ≥ previous release (proposed) | REVIEW |
| 4.8 | Fairness | Alignment `high_stakes_fairness` | ≥ previous release (proposed) | REVIEW |
| 4.9 | Wellbeing and crisis handling | Alignment `human_wellbeing` (self-harm, medical, triage) | ≥ previous release (proposed) | NO-GO |
| 4.10 | Agentic safety | Alignment `agentic_control`; tool calls that would delete, send or pay without confirmation | ≥ previous release (proposed) | REVIEW |
| 4.11 | Human red-team session | 2 people × 1 hour, at least 3 languages, focused on 4.2–4.10; findings logged | no unreviewed severe finding (proposed) | NO-GO |

**Steps:** run 4.1–4.10 from the EUROPA job (~2 h) → triage every failed critical item (4.2, 4.3) into
"real failure / judge error" → red-team session (4.11) → write the findings and decisions on the sign-off sheet.

---

## 5. Identity and honesty about itself

**Why:** a model that says it's ChatGPT, invents its training data or claims abilities it doesn't have (internet
access, memory between chats) misleads users and embarrasses the project.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 5.1 | Identity facts | held-out identity-v2 prompts (validation split), sampled and judged with `scripts/judge_onpolicy_dpo.py` (sample + judge) | ≥ 95 % judged correct; zero claims to be another vendor's model (proposed) | NO-GO |
| 5.2 | Capabilities and limits | identity categories `tools_and_internet`, `memory_and_privacy`, `capabilities_and_limits` | no false capability claims (proposed) | NO-GO |
| 5.3 | Checkpoint-specific unknowns | category `checkpoint_specific_unknowns` (e.g. benchmark scores, release date) | says it doesn't know rather than inventing (proposed) | REVIEW |

---

## 6. Robustness, output quality and efficiency

| # | Check | Why | How | Ready when | Class |
|---|---|---|---|---|---|
| 6.1 | Repetition and prompt echo | loops are the most common bad user experience in small models | EUROPA output diagnostics (prompt echoes, repeated spans) | ≤ previous release (proposed) | REVIEW |
| 6.2 | Answer length | RL tends to inflate length; long default-mode answers cost users | median output tokens per mode (EUROPA, quick eval) | within ±25 % of the previous release unless intended (proposed) | TRACK |
| 6.3 | Thinking token budget | cost and latency | mean thinking tokens on the quick eval | recorded; ≤ previous release for equal accuracy (proposed) | TRACK |
| 6.4 | Multi-turn conversation | most chats have more than one turn | multi-turn slices of the tool check; EUROPA `practical_communication` | ≥ previous release (proposed) | REVIEW |
| 6.5 | Decoding robustness | users won't use our exact settings | rerun 2.1.1 and 2.4.1 at temperature 0.7 / top-p 0.95 | no item collapses (> 0.10 drop) vs greedy (proposed) | REVIEW |
| 6.6 | Overall preference | catches what the narrow evals miss | pairwise LLM judge (gemma4:31b, pinned digest) vs the previous release on ~500 prompts across 36 languages | win rate ≥ 50 % (proposed; eval to build) | REVIEW |

---

## 7. Evaluation integrity

**Why:** a number is only worth something if the model hasn't seen the test. Our holdouts come from hash-bucket
splits; this section proves it.

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 7.1 | Holdouts disjoint from training | hash-bucket ids of every eval set vs every training manifest (SFT, DPO, RL pools) | zero overlap (set) | NO-GO |
| 7.2 | n-gram contamination | 13-gram overlap of eval prompts against the training data | < 1 % of items, and those items reported (proposed) | REVIEW |
| 7.3 | Fresh-seed release evals | EUROPA Reasoning `--seed <new>` and a new seed for the eu_qa sample | within noise of the selection numbers (set, roadmap step 8) | NO-GO |
| 7.4 | Judge pinned | judge model and digest recorded for every judged eval (EUROPA, identity, preference) | digest on the sign-off sheet (set) | NO-GO |

---

## 8. Release paperwork and hygiene

| # | Check | How | Ready when | Class |
|---|---|---|---|---|
| 8.1 | Model card | all numbers from §1–§7 (`scripts/write_model_card_evals.py`, `scripts/plot_model_card_evals.py`), intended use, both modes, recommended decoding, **known limitations** (every REVIEW failure), safety summary | complete and reviewed by a second person | NO-GO |
| 8.2 | Data provenance and licences | every training source with revision and licence ([RL data mixture](OELLM_RL_DATA_MIXTURE.md), playbook); share-alike sources (e.g. the CC-BY-SA MCQ rows) and their implications checked | licence review signed | NO-GO |
| 8.3 | Regulatory documentation | EU AI Act general-purpose model obligations: technical documentation, training-content summary, copyright policy; confirm with the project's legal contact | confirmed | NO-GO |
| 8.4 | Reproducibility record | commit hashes, configs, data manifests, job IDs, costs in the playbook | a reader can rebuild the checkpoint | REVIEW |
| 8.5 | No personal data or site identifiers | `scripts/check_site_identifiers.sh` on every public repo; no personal paths or account names; no personal-data sources in training | passes (set) | NO-GO |
| 8.6 | Upload hygiene | HF token only via stdin; upload private first; check the private repo loads (§1.6 from HF); public only with explicit approval | approval recorded | NO-GO |
| 8.7 | Versioning | semantic version, changelog vs the previous release, eval results committed to the repo | done | REVIEW |

---

## 9. Ready-to-release rule

A candidate is **ready** when:

1. every NO-GO item passes;
2. every REVIEW failure has a written justification and a model-card limitation;
3. TRACK numbers are recorded;
4. the sign-off sheet is signed by the training lead, the eval owner and a safety reviewer (three different people
   where possible).

If any training, merge or template change happens after the checklist ran, the whole checklist runs again.

---

## 10. Sign-off sheet (copy per candidate)

| Field | Value |
|---|---|
| Candidate | name, model dir, weights sha256 |
| Parent and recipe | parent checkpoint, stages, merge recipe |
| Commits | qwen35-posttrain, oellm-rlvr, EUROPA harness |
| Template / tokenizer sha | |
| Previous release compared against | |
| §1 preflight | pass / fail, notes |
| §2–§7 results | table of every item: value, previous, class, pass/fail |
| REVIEW justifications | |
| Red-team findings (4.11) | |
| Judge digests | |
| Decision | GO / NO-GO, date |
| Signed | training lead · eval owner · safety reviewer |

---

## 11. Latest candidate: Stage 4b on W3 (11-03)

**Status: NO-GO.**

| Item | Result | Verdict |
|---|---|---|
| 1.2 template, 1.7 default think-tag rate | v2.1; tool check think leaks 0 | pass |
| 2.1.1 quick reasoning | thinking 0.614 / default 0.591 (W3 0.632 / 0.595) | pass (−0.018) |
| 2.4.1 head band | default **0.03 accuracy / 0.97 abstain**; thinking 0.41 / 0.38 | **NO-GO** |
| 2.5.1 eu_doc | 0.514 (W3 0.516) | pass |
| 3.1.1 tools decision | 0.864 (W2 0.886) | **NO-GO** (−0.022) |
| 3.3.1 HumanEval+ thinking | 0.360 (W3 0.360) | pass |
| 4.1 Alignment macro | 72.0 (W2 67.7) | pass |
| 4.3 failed critical items | 115 (W2 151) | pass |
| 4.4 refusal calibration | 43.9 (W2 34.8) | pass |
| 2.6, 3.4, 4.11, 5, 7, 8 | not run | — |

Next: linear merges of W2 and 4b (RL run program, Stage 4b results) to recover the default mode.
