# Stage 5 — RL run program

Status 2026-10-25: RL-0, RL-0b, A2a, A1, A3, A4, A5 done (results below); Stage-4 DPO v2 trained, evals running. This is the **run-by-run** plan for Stage 5: which RL runs we make, in what order, what each one
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
**16K result (job 62244120, 10-25):** with 16,384 tokens only 22 % of the 114 items finish and 8.8 % are correct
(de/fr best: 33–44 % unfinished, 22 % correct; sv/nl 100 % unfinished at 16,384 tokens). Native traces run away or loop
rather than running out of budget, so a larger budget does not rescue them; this supports option (a). Native reasoning
needs good native-trace SFT data first.

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
| A3 both modes (`think-mixed`, canary 61969304 → full 61969305) | ✅ done 10-24: training reward `mcq_letter_direct` 0.06 → 0.21, `math_direct` 0 throughout (the model never samples a bare answer; format DPO first). Quick eval: thinking 0.532 / unfinished 34 % (= best of A2a/RL-0b), default 0.593 (RL-0 0.593); mixing default-mode rows costs nothing; adopt per-row default-mode rows for W1 after the DPO format pass | template `templates/oellm_qwen3_two_stage_v2_1_rl_mixed.jinja` (thinking by default, no thinking block for rows with `rl_mode: default`); verifiers `math_direct` / `mcq_letter_direct` (bare correct answer only, no think tags; `OELLM_PATCH_DIRECT_MODE_VERIFIERS`); pool `oellm-rlvr/scripts/leonardo_a3_mixed_mode_data.sh` = RL-0 rows + 430 math + 430 letter-only default-mode rows (letter rows mostly bg/hr: the format pairs used most other exams_qa rows); A2a recipe |
| A4 tool single-turn (`think-tool`, canary 61968632 → full 61968633) | ✅ done 10-24: no effect — training reward 0.58 → 0.52; held-out tool decision 0.867 (Stage 3: 0.870), JSON 0.994; thinking 0.432 / unfinished 46 %. Not adopted; tool RL needs a harder/curriculum pool (T1) | verifier `tool_call` (`OELLM_PATCH_TOOL_CALL_VERIFIER`): 0.2 well-formed + 0.3 tool names + 0.5 arguments, no-call rows 1 only without a call, scored after exactly one closed trace; pool `scripts/leonardo_a4_tool_data.sh` = 1,000 English `grpo` + 500 non-English single-call + 500 no-call rows from the Stage-3 tool dataset (held-out bucket excluded), tool blocks re-rendered with our template |
| **A5 code** (`think-code`, job 61949439) | ✅ done 10-24 (walltime at ~step 30/48; evaluated step 24) | training: all-tests pass 0.16 → ~0.30, truncated 73 % → ~30 %. EvalPlus HumanEval+/MBPP+ thinking 0.256/0.265 (Stage 3 0.189/0.243, RL-0 0.250/0.251), closed 57/61 %; direct 0.323/0.344 (unchanged). Math thinking 0.481. Small gain, mostly closure; code in W2 with a longer walltime (~8 h for 48 updates) plus a code-reasoning SFT top-up |

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

Done: RL-0 → RL-0b, A2a, A1 (+16K re-eval), A3, A4, A5 (results in §4 and the status table).

**Queued 10-25 evening (from the Stage-4 candidate = linear merge 0.5 × Stage-3 v1.1 + 0.5 × DPO v2,
model dir `models/stage4-merge-a05-mixed` = merge weights + RL mixed template).** The P1/A4b runs planned from DPO v2
were cancelled before they started once the merge restored tool use (roadmap §5).
- **W1a** `w1a` (canary 62278865 → full 62278866), the first combined window, overnight on 2 nodes: 2,048 prompts =
  128 updates, one pass (`oellm-rlvr/scripts/leonardo_w1a_data.sh`, sha `191e288e`): 560 EU-language math with the
  answer-language check (`multilingual_math`, check=answer), 540 English/GSM8K math, 300 `math_direct` + 300
  `mcq_letter_direct` default-mode rows, 348 code rows (`code_stdio`, contained execution server); non-stop penalty
  −0.5; RL-0 recipe otherwise (DAPO, β 0, LR 1e-6, 16 × 8, 6K responses); checkpoints every 32 updates; walltime 11 h.
  Estimate ~7–9 h, ~500–600 local-h. Evals on the newest checkpoint: quick reasoning (62278869), held-out tool
  (62278870), EvalPlus (62278871). Questions: do the A-series gains add up in one run (thinking accuracy ≥ 0.53,
  unfinished ≤ 34 %, code thinking > 0.256/0.265), does the default-mode bare-answer reward move from the merge (it
  carries half of the format DPO), and do tools and the default mode hold?
- **A4b** `merge-tool` (canary 62278867 → full 62278868): the multi-turn tool pool (`leonardo_a4b_tool_data.sh`) from
  the merge, ~250 local-h; evals tool held-out (62278872) and quick (62278873). Question: learning signal and held-out
  tool check above the merge (decision 0.878, JSON 0.985, args 0.951).
- **A4b result (10-26, job 62278868, 48 updates, 1 h 51 min):** training reward 0.35 → 0.42 (correct 0.60 → 0.65) —
  a learning signal, unlike A4. Held-out tool check (step 48) vs the merge: decision 0.878 → 0.881, names 0.767 →
  **0.784** (best of all models), JSON 0.985 → **0.995**, arguments 0.951 → 0.969, finished 98 %; reasoning held
  (thinking 0.442 / unfinished 49 %, default 0.577). **Adopt the A4b pool for the tool share of W2.**
- **W1a in progress (10-26 morning):** update 73/128 after 3 h; letter-only default answers 0.39 → 0.53, thinking math
  0.55 → 0.57, code 0.07 → 0.12. `math_direct` stays 0: of 2,784 sampled default-mode replies to "final answer only"
  math prompts **none** is bare (median ~290 characters of working before the `\boxed{}`), so RL has nothing to
  reinforce. Not a verifier bug (checked the decoded rollouts). Next lever: a shaped reward for these rows (correct
  final `\boxed{}` scaled by brevity, full credit only when bare) or more bare-answer DPO/SFT data.
- **W1a result (10-26, job 62278866, 128 updates, 4 h 26 min, ~280 local-h; step 128):**

  | Model | default | thinking acc. | unfinished | MATH-500 | EU math | MGSM | GSM8K |
  |---|---|---|---|---|---|---|---|
  | Stage-3 v1.1 | 0.581 | 0.435 | 48.9 % | 0.160 | 0.108 | — | — |
  | merge α 0.5 | 0.568 | 0.426 | 47.4 % | 0.160 | 0.158 | 0.585 | 0.607 |
  | A3 (best ablation) | 0.593 | 0.532 | 34.2 % | 0.180 | 0.283 | 0.700 | 0.740 |
  | **W1a** | 0.574 | **0.619** | **15.4 %** | **0.270** | **0.433** | **0.780** | **0.787** |

  EvalPlus (pass@1 plus, HumanEval+ / MBPP+): thinking **0.348 / 0.328**, traces closed 94 % / 86 % (Stage 3: 0.189 /
  0.243, ~45 % closed; A5: 0.256 / 0.265); direct 0.372 / 0.362 (Stage 3: 0.335 / 0.357). Held-out tool check held
  (decision 0.878, names 0.778, JSON 0.991, args 0.958, finished 99.2 %). Training: code 0.06 → 0.28, `math_direct`
  0.01 → 0.09 (bare answers appear late), letters 0.44 → 0.50. The combined window is far better than any single
  ablation: thinking mode now beats direct mode, and unfinished traces fell from 49 % to 15 %.
- **EUROPA (10-26, jobs 62323272/62323274):** Eval / Alignment — Stage 3 0.616 / 0.642 (malformed 122 / 72); merge
  0.655 / 0.660 (111 / 57); **A4b 0.670 / 0.670 (100 / 43)**. Quantitative reasoning 0.04 → 0.31, agentic control
  0.49 → 0.68; regressions: grounding & safety 0.70 → 0.58, refusal calibration 0.44 → 0.35. Reference threeway DPO
  0.840 (malformed 9). Next levers: refusal-calibration and grounding preference/RL data; strict-format IF in W2.
- **Combination (queued after W1a, job 62324489):** task arithmetic `merge + (A4b − merge) + (W1a − merge)`
  (`slurm/merge_task_arithmetic_leonardo.sbatch`; all three share 399 tensors, bf16), then quick (62324491), tool
  (62324492), EvalPlus (62324493) and EUROPA (62324496); EUROPA also on W1a's final step (62324495). The W2 base is
  the best of merge / W1a / A4b / combination on these (tools ≥ merge, thinking accuracy and closure ≥ W1a, EUROPA ≥
  merge).
- **W2 pieces built 10-26** (oellm-rlvr):
  - `math_direct_shaped` verifier: bare correct answer 1.0; correct last `\boxed{}` after working 0.2–0.6 by length
    (0.2 + 0.4·max(0, 1 − chars/800)); wrong 0. Gives RL a gradient towards bare answers.
  - `ifeval_direct` / `ifeval_think`: TMAX's IFEval/IFBench constraint checkers on the whole default-mode reply or on
    the answer after one closed trace (upstream `ifeval` empties default-mode replies under the closed-trace contract
    and cannot parse JSON `null` in labels). Integration-tested against the real TMAX module.
  - IF pool `scripts/leonardo_w2_if_data.sh`: 1,500 rows of nvidia/Nemotron-RL-instruction_following @ `bcdec57a`
    (ODC-BY), constraints validated against TMAX's registry (80 dropped), sha `ceac22cb`.
  - Multilingual tool slice: `qwen35-posttrain/scripts/translate_rl_tool_pool.py` (user turns only, protected argument
    strings, langid; gemma4:31b), job 62325109, 640 sampled A4b rows.
  - Math profiling `scripts/leonardo_w2_math_profile.sbatch` on the combination (job 62325199): ~1,160 oellm-math-rlvr
    level-3–5 prompts (40 per non-English language + 240 English) not in any pool or eval, 8 samples at 8K tokens,
    banded by solves out of 8.
  - Pool `scripts/leonardo_w2_data.sh` and config `configs/leonardo-grpo-math-oellm9b-w2.yaml`: 2,560 prompts —
    896 profiled math (solved 1–7 of 8), 192 `math_direct_shaped` + 192 `mcq_letter_direct`, 256 `ifeval_direct` + 128
    `ifeval_think`, 384 code, 512 tools (translated slice first); 7K responses, tool prompts ≤ 3K, concave length
    penalty + non-stop penalty, 160 updates; ~400–550 local-h from W1a's measured 2.1 min/update (walltime 12 h caps it at ~770). Canary first, as always.
- **W2 queued 10-26 15:30 from W1a step 128** (`models/w2-base-mixed`; combination was a tie: EUROPA 0.733/0.733 vs
  W1a 0.735/0.711, thinking 0.598 vs 0.619): pool build on `lrd_all_serial` after profiling 62329893 and translation
  62325109 (62346864; falls back to unprofiled math if profiling fails) → W2 62346870 → quick / tool / EvalPlus
  62346873 / 62346877 / 62346880. The canary was dropped at 15:55 (full partition: every chained job queues
  separately; a crash in W2 itself surfaces equally early), and the build no longer waits for the translation (the
  translated tool slice is optional). The partition was full all day
  (≈ 7,900 pending jobs), so the chain was queued early to accumulate priority.
- **Specialist-and-merge test (queued 10-26 evening).** Question: can parallel specialists + merging match a joint
  window (useful on a full partition, where two 2-node runs schedule more easily than one long run)? Evidence so far:
  joint W1a ≫ single-skill ablations; the naive merge + ΔW1a + ΔA4b kept W1a's level but not A4b's tool gain.
  - **T1** `t1-tool` (62358554): the A4b tool recipe from W1a step 128 (same base as W2), 96 updates on the A4b pool +
    translated slice (translation 62358480 as a 1-GPU job, 480 rows; build 62358553). ~240 local-h. Evals tool / quick.
  - **Merges** after W2 and T1 (`slurm/merge_task_arithmetic_leonardo.sbatch`, base W1a step 128):
    W1a + ΔW2 + w·ΔT1 for w = 1.0 / 1.5 / 2.0 (62358559 / 62358562 / 62358566) and TIES (density 0.2, λ 1; 62358570),
    each with quick + tool evals. Compare with W2 alone: if a merge holds W2's reasoning/code and reaches T1's tool
    scores, specialists + merging become a parallel training mode; otherwise tools stay inside the joint windows.
- Each canary (256 episodes, 1 h) gates its full run (`afterok`); a cleanup job cancels the full run and its evals if
  the canary fails.

Next:
1. ✅ Language policy decided (A1 section): traces any language, answer in the prompt language. ✅ 16K re-eval done.
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
