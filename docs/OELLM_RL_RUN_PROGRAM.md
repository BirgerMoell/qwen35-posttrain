# Stage 5 — RL run program

Status 2026-10-23 (evening): RL-0, RL-0b, A2a done; A1 running; A5 (code) queued; A3/A4 not yet built. This is the **run-by-run** plan for Stage 5: which RL runs we make, in what order, what each one
must answer, what it costs, and how its result changes the next run. The data pools, rewards and environment tiers are
specified in [`OELLM_STAGE5_RL_PLAN.md`](OELLM_STAGE5_RL_PLAN.md); results go into the
[playbook](OELLM_POST_TRAINING_PLAYBOOK.md) §10 as each run finishes.

## 1. What RL must deliver

The target is a 9B model that is clearly better than Stage 4 at things we can verify, in all 36 languages, without
losing what SFT and DPO built. Concretely, by the end of Stage 5:

| Capability | Today (Stage-3 v1.1) | Stage-5 target | Measured by |
|---|---|---|---|
| Thinking traces that finish | 51 % closed at 6K (quick eval); 81 % of unclosed traces stay unclosed at 16K, half are loops | ≥ 95 % closed within the window budget, loops ≈ 0 | quick reasoning eval, EUROPA Reasoning process metrics |
| Reasoning accuracy (thinking mode) | 0.435 (quick eval) | clear paired gain on frozen holdouts | MATH-500, AIME, MGSM, PolyMath, EUROPA Reasoning |
| Reasoning in the user's language | 24.8 % of non-English traces in the prompt language (RL-0: 1.1 %) | answers always in the user's language; traces per A1 (≥ 90 % native if the cost is ≤ 2 points, otherwise legible English) | lingua audit on quick-eval traces; EUROPA Reasoning trace-script metric |
| Direct-mode answers and strict formats | EUROPA numeric 0.056, 122 malformed | numeric ≥ 0.5, malformed ≤ 20 | EUROPA Eval |
| Tool use | decision 0.870, JSON 0.995; EUROPA tool-recovery 0.486 | decision ≥ 0.92, recovery ≥ 0.8 | held-out tool eval, EUROPA Eval, BFCL |
| Retention | EUROPA Eval 0.616 / Alignment 0.642 (before DPO) | no regression vs the Stage-4 checkpoint | EUROPA Eval/Alignment, identity, default-mode think-tag rate = 0 |

## 2. Budget envelope

Working budget 50K local-h (account total 400K, shared). Used by 2026-10-23 evening: 9.34K (`sacct`), of which RL
926. Planned DPO: ~9–11K for the full epoch (revised after the speed measurement).
**RL envelope: ~25K local-h** with a ~5K reserve, so the program can absorb one failed window or one extra ablation.

Cost model (measured, canary 6, job 61859252, 2 nodes): 16 prompts × 8 samples, mean response 3.8K tokens, max 6,144:
**~125 s per update ≈ 2.2 local-h per 128 rollouts**. Cost scales roughly with generated tokens, so

    local-h per update ≈ 2.2 × (rollouts / 128) × (mean response tokens / 3,800)

Estimates below use this formula; each run re-measures it and the playbook records the real number.

## 3. Principles every run follows

1. **One question per run.** Ablations change one thing against a fixed baseline; production windows change only what
   the ablations decided.
2. **A run card before submission** (template in §9): parent checkpoint, hypothesis, data with the realized mix,
   reward, settings, budget, live stop rules, end-of-run evaluation, decision rule.
3. **Select on frozen holdouts, never on training reward.** Checkpoints every 16 updates; the selected one is the best
   on the target holdout among those that pass retention.
4. **Profile difficulty on the current checkpoint** (pass@8 bins, Stage-5 plan §4) before each window, and check the
   realized mix per batch.
5. **Check the special-token rows after every run**: the teacher-forced top-1 measurement that
   `repair_tag_head_rows.py` records as `before` (a measure-only flag is still to be added). RL updates are tiny, so
   the rows should not collapse again, but we confirm the closers still win at closing positions.
6. **Both modes stay trained.** Every production window includes default-mode prompts (no think tokens allowed);
   default-mode think-tag rate must stay 0.
7. **Ablations run on Stage-3 v1.1 while DPO trains**, so RL work does not wait for DPO. Their conclusions are
   relative (A vs B on the same parent) and are re-checked once on the DPO checkpoint in W1.

## 4. The run sequence

```
 RL-0 think probe ──► A-series ablations (parallel with DPO) ──► Stage-4 DPO checkpoint
                                                               │
                         W1 warm-up (4K) ◄─────────────────────┘
                              │ gate
                         W2 core (8K)
                              │ gate
                         W3 frontier (16K)
                              │ gate
                  T1 single-turn tools ──► T2 multi-turn tools / search (SkyRL)
                              │ gate
                         F final selection, release evals
```

### RL-0 — Think probe (done: job 61868620, 48 updates, 2 h 51 min, ~50 local-h)

**Result (quick reasoning eval, held out, vs Stage-3):** thinking accuracy 0.435 → **0.512**, unfinished
48.9 % → **39.3 %**, mean thinking length 3,478 → 2,970 tokens; default mode 0.581 → 0.593 (retained). Largest gains on
EU-language math (0.108 → 0.308; unfinished 86 % → 58 %) and MGSM (0.59 → 0.66); **MATH-500 flat** (0.16 → 0.14,
80 % unfinished at 6K). Decision: closure responds to RL (+9.6 points closed, accuracy up) → proceed; run A2 because
hard problems still do not finish; the update count is not capped by the episode counter (48 of 48 ran).

**Follow-ups submitted 2026-10-23** (2 nodes each, ~50 local-h each, quick eval of step 48 queued automatically):
- RL-0b `think-cont` (job 61923006): 48 more updates from RL-0 step 48, seed 43 — is the gain still rising?
- A2a `think-nonstop` (job 61923004): RL-0 with one change, truncated rollouts get −0.5 instead of 0.

**Results of RL-0b and A2a (quick reasoning eval, thinking mode, held out; 570 items):**

| Model | Accuracy | Unfinished | Tokens | GSM8K | MGSM | MATH-500 | EU math | Trace in prompt language | Default-mode acc. |
|---|---|---|---|---|---|---|---|---|---|
| Stage-3 v1.1 | 0.435 | 48.9 % | 3,478 | 0.673 | 0.590 | 0.160 | 0.108 | 24.8 % | 0.581 |
| RL-0 (48 updates) | 0.512 | 39.3 % | 2,970 | 0.727 | 0.660 | 0.140 | 0.308 | 1.1 % | 0.593 |
| RL-0b (RL-0 + 48) | **0.532** | 37.4 % | 2,848 | 0.700 | 0.680 | 0.180 | **0.367** | 0.4 % | 0.593 |
| A2a (−0.5 if unfinished) | 0.523 | **34.0 %** | **2,649** | 0.687 | **0.690** | **0.210** | 0.300 | 1.5 % | 0.575 |

Readings: more updates on the same small pool still add a little (+2 points) but are flattening; the non-stop
penalty gives the shortest, most often finished traces and the best MATH-500 at equal update count to RL-0
(+1.1 points overall vs RL-0, +5 on MATH-500 with 120/100-item subsets, so within noise individually). Every RL
variant reasons in English. Decision for W1: keep the non-stop penalty (−0.5); the language question is A1's.

- A1 `think-lang` (job 61928425): RL-0 with one change, the 1,008 EU-language rows scored by `multilingual_math`
  (correct AND trace in the prompt language AND valid think/box form; lingua detector). Pool
  `oellm-rlvr/scripts/leonardo_a1_language_data.sh` (sha256 `958f0939…`).

**Language finding (lingua on the quick-eval thinking traces, 270 non-English rows):** the share of traces in the
prompt language fell from **24.8 % (Stage 3) to 1.1 % (RL-0)**. Stage 3 reasoned natively on the fully native
oellm-math-eu prompts (cs, fi, hu, nl, sv 10/10, el 9/10) but those traces rarely closed; RL-0 answers the same prompts
with English traces that close and score. MGSM prompts carry an English instruction, and Stage 3 already reasoned in
English there. So part of RL-0's EU-math gain is a switch to English: every later run must log the trace-language
rate, and A1 decides whether the language term goes into W1–W3.

**A1 result (quick eval, thinking mode, held out; lingua audit):**

| | Stage 3 | RL-0 | A1 (language reward on EU rows) |
|---|---:|---:|---:|
| Overall accuracy | 0.435 | 0.512 | 0.514 |
| EU math (fully native prompts, 120): accuracy | 0.108 | 0.308 | 0.192 |
| EU math: unfinished | 85.8 % | 57.5 % | 73.3 % |
| EU math: trace in prompt language | 56 % (67/120) | 2.5 % (3/120) | **65 % (78/120)** |
| MGSM (English instruction line) | 0.590 | 0.660 | 0.720 |
| MATH-500 | 0.160 | 0.140 | 0.230 |
| GSM8K | 0.673 | 0.727 | 0.687 |

Reading: the language reward works (native traces 2.5 % → 65 % on native prompts, above Stage 3) and overall accuracy is
unchanged, but on the prompts where it binds, accuracy is 11.6 points below RL-0 (still +8.4 over Stage 3) because native
traces finish less often (73 % unfinished). The cost is a termination weakness of native-language reasoning, which comes
from mostly-English reasoning SFT data, not from the reward. Against the decision rule (cost ≤ 2 points ⇒ adopt), the
language reward is **not** adopted for W1 as is. Options for the user/project to decide (roadmap §6): (a) English
traces allowed, answers in the user's language, legibility monitored; (b) add native-trace SFT (a "bridge" of verified
native-language reasoning traces) before RL, then re-test A1; (c) language reward with a softer weight (e.g. correct but
off-language = 0.5) to trade accuracy against native traces. Differences per benchmark are within noise at 100–200
items; the EU-math gap (37 vs 23 correct of 120) is the clearest effect.

Per-language breakdown of A1 (native EU math, ~8 prompts per language, so noisy): de/fr/es/en keep 0.68–0.80 accuracy
with 16–26 % unfinished; cs/hu/nl/sv/bg/fi fall to 0.0–0.1 with 60–100 % unfinished at the 6,144-token limit. The cost
is concentrated in the smaller languages and looks like a token-budget/termination problem rather than wrong
reasoning. Check: the 114 unfinished native-language items re-generated at 16,384 tokens (job 62021559,
`eval/reasoning-quick/think-lang-step48-unfinished-native-16k`). If most of them finish and are correct, native
reasoning becomes a per-language budget/SFT question (option b) rather than a capability limit.

**Decision (2026-10-23): option (a) with an answer-language check.** W1–W3 allow traces in any language; the
EU-language math rows use `multilingual_math` with label `"check": "answer"` (oellm-rlvr `f6552b6`): reward 1 only if
the answer is correct, the think/box form is valid and the final answer's prose (after `</think>`, boxes and LaTeX
removed) is in the prompt language; answers with < 20 letters of prose (a bare `\boxed{}`) pass. Pool:
`CHECK=answer scripts/leonardo_a1_language_data.sh` → `data/answer-language/` (2,008 rows, sha `0b745cd2…`). On the
existing quick-eval outputs, 92–96 % of finished non-English answers already pass (RL-0: 156/172, A2a: 183/192), so
the term rarely binds and needs no separate ablation; it mainly stops two drifts seen in RL-0/RL-0b/A2a: English
answers to German/French/Hungarian prompts and Russian answers to Bulgarian prompts. Native-language reasoning becomes a
later opt-in (system-prompt switch, high-resource languages first) on top of the English-reasoning model.

#### RL-0 card (as planned)

| | |
|---|---|
| Parent | Stage-3 v1.1, RL thinking-default template |
| Question | Does DAPO with "unclosed trace ⇒ reward 0" make traces close, without hurting accuracy? |
| Data | 2,008 prompts: 1,408 oellm-math (multilingual) + 600 GSM8K |
| Settings | DAPO, β 0, LR 1e-6 constant, 16 × 8, max response 6,144, `total_episodes` 6,144, 2 nodes |
| Cost | ≤ 320 local-h (estimate ~140) |
| Live signals | closed-trace rate per update, mean length, `sequence_lengths_max` hits, entropy, mixed-group fraction |
| End eval | quick reasoning eval (thinking, 6K) vs Stage-3 v1.1; EUROPA Reasoning both modes |
| Decision | closed rate up ≥ 10 points with accuracy flat or up → termination is an RL problem, proceed. Closed rate flat → add explicit length shaping (A2) before W1. Accuracy down → lower LR (5e-7) in all later runs. |
| Open item | canary logged ~200 episodes per update (expected 128; dynamic-sampling refills are counted), so 6,144 episodes may give ~30 updates, not 48. Record the real update count; set later runs by `max_updates`, not episodes. |

### A-series — ablations (2 nodes each, on Stage-3 v1.1, in parallel with DPO)

Each runs 32–48 updates at 16 prompts × 8 samples, 6K max response, against the RL-0 settings as the baseline arm.

**Status:**

| Run | Status | Result / note |
|---|---|---|
| A1 language (arm c, `think-lang`, job 61928425) | ✅ done | not adopted; **answer-language check (arm b) adopted for W1–W3** without its own run (it binds on < 10 % of correct answers); 16K re-eval of unfinished native items queued (62021559) |
| A2a non-stop penalty (`think-nonstop`, job 61923004) | ✅ done | fewest unfinished (34 %), best MATH-500; **adopted for W1–W3** |
| A2b loop penalty | not needed yet | revisit if loops persist at 8K/16K |
| A3 both modes (`think-mixed`, canary 61969304 → full 61969305) | submitted 10-23 night | template `templates/oellm_qwen3_two_stage_v2_1_rl_mixed.jinja` (thinking by default, no thinking block for rows with `rl_mode: default`); verifiers `math_direct` / `mcq_letter_direct` (bare correct answer only, no think tags; `OELLM_PATCH_DIRECT_MODE_VERIFIERS`); pool `oellm-rlvr/scripts/leonardo_a3_mixed_mode_data.sh` = RL-0 rows + 430 math + 430 letter-only default-mode rows (letter rows mostly bg/hr: the format pairs used most other exams_qa rows); A2a recipe |
| A4 tool single-turn (`think-tool`, canary 61968632 → full 61968633) | submitted 10-23 night | verifier `tool_call` (`OELLM_PATCH_TOOL_CALL_VERIFIER`): 0.2 well-formed + 0.3 tool names + 0.5 arguments, no-call rows 1 only without a call, scored after exactly one closed trace; pool `scripts/leonardo_a4_tool_data.sh` = 1,000 English `grpo` + 500 non-English single-call + 500 no-call rows from the Stage-3 tool dataset (held-out bucket excluded), tool blocks re-rendered with our template |
| **A5 code** (`think-code`, job 61949439) | queued | added 10-23: single-turn code RL with the contained `code_api` server; canary passed (16 % all-tests pass) |

| Run | Question | Arms | Data | Cost | Decides |
|---|---|---|---|---|---|
| **A1 language** | Can RL make the model reason in the prompt language, and what does it cost in accuracy? | (a) baseline; (b) answer-language multiplier; (c) answer + trace-language multiplier | oellm-math EU24 + exam-MCQ, non-English only | 2 × ~150 local-h (arm a = RL-0) | whether W1–W3 use the trace-language multiplier, and for which languages (drop any language that loses > 3 points) |
| **A2 length** (only if RL-0 does not fix closure) | Which shaping fixes termination? | (a) DAPO soft overlong punishment; (b) (a) + loop penalty (n-gram repeat ratio ⇒ 0) | RL-0 data | 2 × ~150 local-h | the structure guards for all windows |
| **A3 both modes** | Does mixing default-mode prompts keep and improve the direct mode? | (a) thinking only; (b) 70 % thinking + 30 % default-mode strict-format and numeric prompts | math + strict-format set | ~150 local-h (arm a = RL-0) | the default-mode share in W1–W3; expected to fix EUROPA numeric |
| **A4 tools single-turn** | Does the ToolRL reward on the 46K GRPO split improve held-out tool decisions? | one arm | oellm-eu-tooluse-v1 `grpo` + no-call items | ~150 local-h | whether tools enter W2 or wait for T1 |
| **A5 code** | Is there learning signal in single-turn code RL, and does it transfer to HumanEval+/MBPP+? | one arm | 2,004 `oellm-code-rlvr` train problems (12 families, d1–4) | ~80 local-h | whether code enters W2 as planned or needs an SFT top-up first |

Ablation total: ~600–900 local-h.

### W1 — Warm-up window (Stage-4 DPO checkpoint)

| | |
|---|---|
| Question | Does the RL-0 + ablation recipe transfer to the DPO checkpoint, at the production batch size? |
| Data | math (oellm EU24 ≥ 40 % non-English, DeepMath 3–5), exam-MCQ (letter-only answers; **only sources that are neither evaluation benchmarks nor license-unclear**: drop `belebele`, `global_mmlu`, `mmmlu`, `xcopa` — we report them in oellm-eval — and every `unknown`/NC/needs-review source; today that leaves mainly `exams_qa` and `slovak_financial_exam`), strict format, IF ≤ 2 constraints; 30 % easy / 60 % medium / 10 % hard by measured pass@8 |
| Settings | 64 prompts × 8, max response 4K, 48 updates, decisions from A1–A3 |
| Cost | profiling ~150 + training ~450 local-h |
| Gate | target holdouts up (paired test), closed rate ≥ 90 %, no retention regression |

### W2 — Core window

| | |
|---|---|
| Data | + DeepMath 5–7, POLARIS, PrismMath, oellm-code d1–3, IF ≤ 5 constraints, tool calls (if A4 passed); 10 / 55 / 35 % |
| Settings | 64 × 8, max response 8K, 64 updates, re-profiled on the W1 export, fresh optimizer |
| Cost | profiling ~250 + training ~1.5K local-h |
| Gate | as W1, plus code holdouts and IFEval |

### W3 — Frontier window

| | |
|---|---|
| Data | + DeepMath 7–9, DeepCoder, code d4–5, hard multilingual math, re-profiled former 0/8 items; 5 / 45 / 50 % |
| Settings | 64 × 16, max response 16K, 64 updates |
| Cost | profiling ~400 + training ~4.5K local-h |
| Gate | AIME / LiveCodeBench / PolyMath gains; closure at 16K ≥ 95 % |

Scaling rule: if a window's holdout curve is still rising at its last checkpoint, extend it in steps of 32 updates
from the reserve rather than starting the next window.

### T1 — Single-turn tool RL (if not already in W2)

oellm-eu-tooluse-v1 `grpo` (46K, **English only, exactly one call per row, no no-call rows** — checked 10-23) plus
multilingual and no-call rows from the Stage-3 tool dataset (as in A4), Nemotron function-calling pivot; ToolRL reward;
default and thinking mode. ~800 local-h. Gate: held-out tool eval decision ≥ 0.92, BFCL irrelevance, no regression.

### T2 — Multi-turn tools and search (SkyRL)

Mock-tool loops with injected tool errors (the EUROPA tool-recovery gap), conversational tool use with simulated
users, and a search agent over a **local multilingual Wikipedia index** (the European angle: retrieval in the user's
language). Needs SkyRL qualified on Leonardo first (it is qualified on LUMI); qualification canary ~100 local-h.
Training ~3–5K local-h. Gate: EUROPA tool-recovery ≥ 0.8, τ²-bench (eval only), no retention regression.

### F — Final selection and release

Full eval suite on the last two promoted checkpoints (both modes), the special-token check, the identity audit, and
a fresh-seed EUROPA Reasoning variant (`--seed`) as the uncontaminated release check. ~300 local-h.

## 5. Budget summary

| Block | Local-h (estimate) |
|---|---|
| RL-0 think probe | ~140–320 |
| A-series ablations | ~600–900 |
| W1 / W2 / W3 (incl. profiling) | ~600 / ~1.75K / ~4.9K |
| T1 / T2 (incl. SkyRL canary) | ~800 / ~3–5K |
| Evaluation across the program | ~1.5K |
| Final selection | ~300 |
| **Total** | **~14–16K**, leaving ~9–11K of the 25K envelope for window extensions |

## 6. Live monitoring and stop rules (all runs)

Stop the run (and keep the last good checkpoint) if any of these hold for 4 consecutive updates:

| Signal | Stop when |
|---|---|
| Entropy | falls below 30 % of its first-update value (collapse) |
| Mean response length | rises > 50 % with no reward gain (padding / looping) |
| Hits at max length | > 25 % of rollouts |
| Mixed groups | < 20 % of groups have mixed rewards (no learning signal; re-profile) |
| Default-mode think tags | any |
| Verifier errors | > 2 % of rollouts |
| Language (A1 onward) | trace-language rate falls > 10 points |

Every window also gets a manual audit of the 20 highest-reward rollouts (reward hacking: answers inside the trace,
several boxed answers, prompt echo).

## 7. Merging and combining checkpoints

Merging is cheap (minutes on CPU with mergekit, no GPU) and works when the models share a parent, tokenizer and
template and differ by small deltas, which holds for our DPO and RL stages (LR 5e-7 to 1e-6). The cost is the
evaluation, so a merge is only made where it answers a concrete question.

**Standing (every run):**

| When | Merge | Compared against | Cost |
|---|---|---|---|
| End of DPO and of every RL window | uniform average of the last 2–3 saved checkpoints | the best single checkpoint | one eval pass |
| After DPO, **only if** a gate shows thinking-mode or tool regression | linear interpolation Stage-3 v1.1 → DPO, α ∈ {0.5, 0.75, 1.0} (α = 1 is DPO) | DPO and Stage 3 | three eval passes |

The DPO interpolation is the first fallback for the risk that all-default-mode, tool-free DPO pairs weaken thinking or
tool use; it replaces "start RL from Stage 3" unless every α fails the gates.

**Options (used only if the situation arises):**
- **Recovery after a specialist run**: if T1/T2 tool RL regresses reasoning, combine task vectors of the reasoning and
  tool checkpoints over their common parent (task arithmetic first; TIES/DARE if the deltas conflict). Mixed-data RL
  with a controlled mix stays the main path.
- **Delta transplant onto a new SFT checkpoint**: when the instruct data is rebuilt (strict-format targets fixed), add
  the (DPO + RL) delta to the new SFT checkpoint and evaluate before deciding to re-run DPO and RL. Works less well the
  more the new SFT checkpoint differs.
- **Final soup**: average of the last promoted W3 and T2 checkpoints, before the release evaluation.

**Never merge**: models from a different base, tokenizer or RoPE configuration (use distillation instead); models
trained for different template contracts (e.g. Stage 2 with Stage 3); **Stage-3 v1 (`0dc97efa`)**, whose collapsed tag
rows would partly undo the v1.1 repair.

**Rules for every merge**: the output keeps our tokenizer and the v2.1 template (never mergekit's tokenizer union);
mixing weights are chosen on the selection holdouts, never on the release evaluations; the merged checkpoint passes
the same gates as any checkpoint (both modes, tool eval, EUROPA Eval/Alignment/Reasoning, trace closure, tag-row
check) and is recorded with its recipe (inputs with revisions, method, weights, mergekit version).

## 8. Order of work

Done: RL-0 → RL-0b, A2a (results in §4); A1 and A5 submitted.

Next:
1. ✅ Language policy decided (A1 section): traces any language, answer in the prompt language. Pending: 16K re-eval 62021559.
2. A5 result + EvalPlus on its final export → decide code in W2 (or SFT top-up).
3. Build A3 (per-row default mode) and A4 (tool reward); run both on Stage-3 v1.1 while DPO trains.
4. Build W1 pools: oellm-math-rlvr EU24 (≥ 40 % non-English) + DeepMath 3–5, exam-MCQ (35 languages, letter answers),
   strict-format set, IF ≤ 2 constraints; decontaminate against every eval; profiling job (pass@8) ready to run on the
   DPO checkpoint the day it passes its gates.
5. W1 → W2 → W3, each gated; T1 inside W2 if A4 passes; legibility and language monitors from W1 on.
6. In parallel with W2/W3: SkyRL qualification canary on Leonardo, mock-tool and Wikipedia-search environments.

## 9. Run card template

```
Run ID / job IDs:
Parent checkpoint (path, HF revision):
Question (one sentence):
Data: pools, rows, languages, realized mix (logged), decontamination report
Reward: verifier versions, multipliers, guards
Settings: algorithm, LR, prompts x samples, max response, updates, nodes
Budget: estimate (formula §2) and actual
Live stop rules: §6 (+ run-specific)
End evaluation: holdouts and retention suite, both modes
Decision rule: what result leads to which next run
Result: numbers, selected checkpoint, decision taken
```
