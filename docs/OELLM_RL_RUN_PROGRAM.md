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
- **T1 result (10-27, job 62358554):** 96 updates in 3 h 09 min (~200 local-h); training reward ~0.35-0.40 → ~0.45-0.50,
  mean solved length 290 → 480 tokens. Evals queued (62358557 / 62358558).
- **W2 started 10-27 ~12:30** (62354506): update 63 / 160 after 1 h 37 min (~1.5 min/update), truncation ~0-1 %,
  mean response 600-900 tokens (length penalty active), reward ~0.38 → ~0.45.
- **EU task suites (new evals and RL pools, 10-27)** in the spirit of the Artificial Analysis index, sized for a 9B
  model; generator + programmatic check in `oellm-rlvr/src/oellm_rlvr/eu_tasks.py`, so eval and RL reward are the
  same code (`eu_qa`, `eu_doc`; `OELLM_PATCH_EU_TASK_VERIFIERS`). Eval `scripts/eval_eu_tasks.py` +
  `slurm/eval_eu_tasks_leonardo.sbatch` (1 GPU, both modes, ~16 min).
  - `eu_qa` (AA-Omniscience spirit): Wikidata (CC0) facts about European people, places, works and organisations,
    7 question types x 17 countries x 3 popularity bands (subject sitelinks >= 30 / 10-29 / 3-9), 10 languages;
    reward +1 correct, 0 `ANSWER: UNKNOWN`, -1 wrong. Eval 600 (60 per language, 200 per band), RL 30,791.
  - `eu_doc` (GDP.pdf / AA-Briefcase spirit): generated invoices (VAT), staff tables, contracts (end dates), room
    bookings (clashes, minutes) and letters -> JSON, locale-formatted numbers and dates; one question variant per family
    held out of RL. Eval 500 (120 held-out variants), RL 6,000.
  - Published (public, CC BY 4.0): `birgermoell/oellm-eu-tasks` @ `50092422` (configs `qa` / `doc`, splits test / train).
  - Baselines (strict `ANSWER:` line; in brackets: localized answer keywords accepted):

    | Model / mode | QA acc. | head acc. | abstain | halluc. | QA index | doc score |
    |---|---|---|---|---|---|---|
    | W1a default | 0.273 | 0.515 | 3 % | 95 % | -42.0 | 0.378 (0.464) |
    | W1a thinking | 0.270 | 0.505 | 5 % | 93 % | -40.8 | 0.351 (0.399) |
    | A4b default | 0.265 | 0.500 | 3 % | 96 % | -43.7 | 0.360 |
    | A4b thinking | 0.238 | 0.460 | 3 % | 96 % | -49.0 | 0.289 (30 % unfinished) |
    | Qwen3-8B default | 0.157 | 0.310 | 15 % | 82 % | -53.3 | 0.742 |
    | Qwen3-8B thinking | 0.205 | 0.425 | 24 % | 70 % | **-35.5** | **0.947** |

    Our models know more European facts (head 0.51 vs 0.31-0.43) but almost never abstain; on documents they lose to
    Qwen3-8B on format (the `ANSWER:` keyword is translated: "Respuesta:", "Odpowiedź:") and on multi-step table
    arithmetic, and our thinking mode is below default mode there.
- **W3 queued 10-27** (`configs/leonardo-grpo-math-oellm9b-w3.yaml`): the W2 recipe from W2's final checkpoint
  (staging + pool job 62430100 after W2, `scripts/leonardo_w3_stage.sbatch` → `models/w3-base-mixed`), 160 updates on
  2,560 prompts (`scripts/leonardo_w3_data.sh`, dry-run sha `e019c61d`): 512 eu_qa (171/171/170 by band), 640 eu_doc
  (128 per family), 512 math, 256 IF, 256 code, 384 tools; ~40 % of the EU slices in thinking mode. Training 62430101;
  evals quick / tool / code 62430102 / 62430104 / 62430105, EU suites 62430250 (and W2 on the EU suites 62430244).
  Estimate ~300 local-h. Questions: does the QA index rise through abstention without losing head accuracy (watch for
  over-abstaining), does the doc score close the gap to Qwen3-8B (incl. held-out variants), do W2's skills hold?
- **W2 result (10-27, job 62354506, 160 updates, 3 h 41 min, ~30 GPU-h; step 160):** thinking 0.630 (W1a 0.619),
  unfinished 14.0 % (15.4 %), MATH-500 thinking 0.31 (0.27), EU math thinking 0.50 (0.43), default 0.575 (0.574);
  tools decision / names / JSON / args 0.886 / 0.789 / 1.000 / 0.972 (best in the lineage; W1a 0.878 / 0.778 / 0.991 /
  0.958); EvalPlus plus direct 0.390 / 0.365 (W1a 0.372 / 0.362), thinking 0.299 / 0.333 (0.348 / 0.328); EUROPA Eval /
  Alignment 0.727 / 0.717 (malformed 53 / 2; W1a 0.735 / 0.711); EU suites QA index −39.3, doc 0.376 (≈ W1a, not
  trained on). About two thirds of sampled groups had zero reward variance (mostly all wrong) and were refilled.
  **W2 is the new best checkpoint → public release** `birgermoell/oellm-9b-256k-stage5-w2-rl-step160` (card generated by
  `scripts/plot_model_card_evals.py --release w2` + `scripts/write_model_card_evals.py --release w2`; oellm-eval
  knowledge run 62530211 to be added).
- **Specialist-and-merge test result (10-28): negative.** T1 (tool specialist from W1a) did not improve tools (decision
  0.864, names 0.745 vs W1a 0.878 / 0.778) and kept reasoning (thinking 0.611). Merges W1a + ΔW2 + w·ΔT1, thinking /
  tool decision / names: w 1.0 0.618 / 0.856 / 0.740; w 1.5 0.591 / 0.850 / 0.734; w 2.0 0.621 / 0.820 / 0.698; TIES
  0.598 / 0.875 / 0.762; all below W2 alone (0.630 / 0.886 / 0.789). Adding a delta that learned nothing only adds
  noise. Decision: keep skills inside joint windows; specialists only where the specialist itself beats the joint run.
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

## 10. Continual improvement loop (from W4)

Status 2026-10-29. After W3 the model improves in repeated rounds instead of a fixed run sequence. More data is not
the constraint (open verifiable data and our own generators are effectively unlimited); **difficulty fit, coverage
and release gates are.** W2 left 69 % of its prompts unsolved at k = 8 (no GRPO signal), so every round re-profiles
its pool on the checkpoint it will train.

### 10.1 One round

1. **Measure** the current checkpoint: the eval suite (knowledge, EU task suites, IF, tools, code, identity, safety,
   both modes) plus environment baselines: svenska-rl (`slurm/svenska_rl_baseline_leonardo.sbatch`) and NeMo Gym
   (`slurm/nemo_gym_baseline_leonardo.sbatch`).
2. **Diagnose** 2–3 weak areas from per-task scores, solve rates and the previous run's filtered/unsolved ratios.
3. **Build the pool by pass rate** on that checkpoint (`oellm-rlvr` `pass-rate-sample/-score/-select`,
   `scripts/leonardo_pass_rate.sbatch`): k = 8 samples with the RL template, the training verifiers per `dataset`
   (same `OELLM_PATCH_*` flags), keep prompts whose reward varies within the group (optionally a solve-rate band),
   cap per dataset, decontaminate against every eval. Record the profile and the model in the pool manifest.
4. **Train** one RL window (~150–250 updates, the W2 recipe: DAPO, async, non-stop and length penalties). Behaviours
   RL cannot reach go to targeted DPO/SFT (identity, safety: Stage 4b).
5. **Gate the release:** the targeted areas improve and nothing on the guard set (knowledge, identity, safety, IF,
   multilingual, both modes) regresses beyond its margin. Then model card, HF/Ollama release.
6. **Record** the round in this document (run card §9) and update the next round's candidates.

### 10.2 Candidate data

| Source | Adds | Status / requirement |
|---|---|---|
| EU task suites (`eu_qa`, `eu_doc`) | calibrated knowledge with UNKNOWN, EU document tasks | in W3 |
| svenska-rl (8 environments, contract v0.2) | Swedish office work: invoices, payment matching, scheduling, redaction, procurement, stock | baseline done (below); typed ones need a `svrl_typed` verifier in compat |
| Translated verifiable prompts (math, logic, tool tasks into the 12 EU languages) | multilingual reasoning; answers language-independent, verifiers unchanged | qwen3.6 translation pipeline exists; spot-check |
| NVIDIA Nemotron-RL (CC-BY/ODC-BY): instruction following, Structured Outputs, QA-Abstention, SysBench, ReasoningGym | schema/format compliance, system-prompt following, abstention against hallucination | NeMo Gym baseline in progress; judge-based ones use the local Gemma judge |
| Procedural puzzles (ReasoningGym-style) | logic, planning, counting with difficulty knobs | unlimited |
| Long-context synthetic tasks (32K–128K retrieval/aggregation) | uses the 256K context in RL | needs long-rollout throughput tests |
| Multi-turn tool environments (svenska-rl tool episodes, Workplace Assistant) | agentic work over several steps | needs a multi-turn RL runtime (SkyRL/Harbor adapter or NeMo RL); T1 showed single-call tool RL saturated |
| Rubric-judged writing (EU-language e-mails, summaries, plain language) | quality without an exact answer | later; reward-hacking audit first |

### 10.3 Order

| Round | Content | Requirement |
|---|---|---|
| W3 | EU task suites + the W2 mix | queued (2 nodes); move to 1 node if it keeps waiting |
| Stage 4b DPO | identity (v2.1) + safety pairs (reduced: ~2,000 prompts × 4 samples), **both sampled on the W3 checkpoint** (on-policy; the 711 W2 identity pairs are a pilot) | after W3 lands (decided 10-29); gate on EUROPA refusal calibration (0.35) and grounding & safety (0.60), which regressed in the merge, not in RL |
| W4 | + Structured Outputs, QA-Abstention, SysBench, svenska-rl typed tasks, translated math/logic; pool by pass rate | `svrl_typed` verifier patch; baselines |
| W5 | multi-turn tool RL (svenska-rl tool episodes, Workplace Assistant) | multi-turn runtime |
| W6 | long-context RL | throughput tests |
| later | rubric RL for writing quality | judge audit |

Cost: one 2-node window of ~24 h ≈ 190 GPU-h ≈ 1.5K local-h; measurement and pool building add ~100–200 local-h per
round. Compute is not the bottleneck (§2); curation and the multi-turn runtime are.

### 10.4 First measurements

**svenska-rl L1 baseline, W2 (= Agora 9B), 64 eval tasks per environment, greedy, job 62626669 (17 min, 1 GPU),
svenska-rl a8a81dc; re-scored after the answer-parsing fix in the next commit (JSON keys without diacritics,
markdown around `SVAR:`).**

| Environment | Mean reward | Solved | Note |
|---|---|---|---|
| `document_redaction` | 0.79 | 0.25 | finds most items; exact lists are rare |
| `payment_matching` | 0.66 | 0.36 | good candidate band |
| `inventory_reorder` | 0.21 | 0.00 | answers 0 for every article: equals the all-zero shortcut (0.22) |
| `payment_registration` (tools) | 0.15 | 0.02 | often stops after one tool call |
| `invoice_amount_due` | 0.11 | 0.11 | arithmetic over many lines, rounding to whole kronor wrong |
| `meeting_slot` | 0.11 | 0.11 | |
| `supplier_quotes` | 0.09 | 0.09 | |
| `meeting_reschedule` (tools) | 0.06 | 0.03 | |

Most environments sit in or near the useful band for GRPO; the low ones need a pass-rate profile at k = 8 (greedy
accuracy understates sampled variance) before they go into a pool.

### 10.5 Reward-coverage bug in W1a/W2 (found 10-29, fixed before W3)

Building the pass-rate filter showed that several verifier patches never took effect in training. Each
`OELLM_PATCH_*` flag installed its own post-import hook on `open_instruct.ground_truth_utils`; the first hook on
`sys.meta_path` loaded the module and ran only its own patch, so only the last-registered verifier patch applied.
Separately, `ifeval_think` subclassed `ifeval_direct` and `build_all_verifiers` only instantiates direct subclasses.
Training logs ("No reward function found for dataset …"): **W1a** skipped `multilingual_math` (6,305); **W2** skipped
`ifeval_direct`, `ifeval_think`, `math_direct_shaped`, `mcq_letter_direct`, `multilingual_math` (22,993), and the
closed-trace check was off. Those rows got no task reward, so their groups mostly carried no signal: wasted samples,
not wrong gradients (the non-stop penalty still applied). **W2's "69 % unsolved" includes these unscored rows and
overstates the difficulty problem**; re-measure with the pass-rate filter. Fixed in oellm-rlvr `bfcdb83` and
`089ce7b` (regression tests); a compute-node check and the pass-rate scoring smoke (job 62637819: all W3 datasets
scored, 0 errors) passed, then W3 (62430101) was released. W3 is the first window with every verifier active.

Activating every verifier exposed two launch failures, both fixed:
- **10-31 (62680105):** Ray could not pickle the reward config. `multilingual_math` was registered for the first time,
  and it held a `threading.Lock` on a patch-local class; Ray cloudpickles such classes by value. oellm-rlvr
  `47c03dc` moves the detector to `language_audit.shared_detector`; a check confirms all 34 W3 verifiers pickle.
- **11-01 (62990735), step 1:** the ifeval verifiers need NLTK `punkt_tab`, which offline nodes lack. Configs and
  `leonardo_pass_rate.sbatch` now set `NLTK_DATA=$OELLM_SCRATCH/oellm-rl/nltk_data`.
  - The W3 pass-rate profile hit the same error: 389 ifeval scores errored, so its `incomplete` ifeval counts are
    this bug, not difficulty.
  - With the fix, 80/80 ifeval samples score.
  - W3 was resubmitted as 63215513.

- **W3 result (11-02, job 63215513, 160 updates, 3 h 08 min on 1 node; step 160):**

  | Metric | W2 | W3 |
  |---|---|---|
  | Quick reasoning, thinking / default | 0.630 / 0.575 | **0.632 / 0.595** |
  | EvalPlus HumanEval+ / MBPP+ (thinking) | 0.299 / 0.333 | **0.360 / 0.341** |
  | Held-out tools: decision / names / JSON / args | 0.886 / 0.789 / 1.000 / 0.972 | 0.850 / 0.776 / 1.000 / 0.978 |
  | eu_doc score, default / thinking | 0.376 / 0.347 | **0.516 / 0.510** |
  | eu_qa default: acc / abstain / wrong | 0.28 / 0.05 / 0.67 | **0.003 / 0.995 / 0.002** |
  | eu_qa thinking: acc / abstain / wrong | 0.26 / 0.04 / 0.70 | 0.18 / 0.67 / 0.16 |
  | eu_qa head band: acc / abstain (thinking) | 0.51 / 0.01 | 0.38 / 0.42 |

  - **Abstention collapse.** In default mode W3 answers UNKNOWN to 99.5 % of the questions, including the
    well-known ones. The ternary reward on a tail-heavy pool did this: 216 of 512 eu_qa prompts were never solved at
    k=8, so UNKNOWN (0) beat guessing (expected about −0.8) almost everywhere. Nothing in the pool rewarded answering
    known facts over abstaining, so the policy generalised to "always abstain". This is the failure TruthRL reports
    for rule-based rewards.
  - The thinking mode keeps some knowledge (index +1.8 vs −43.7), but head-band accuracy falls from 0.51 to 0.38.
  - **The other skills improved or held:** documents +0.14, code +0.06, reasoning flat, tools −0.04 on decisions.
  - **Decision (11-02):** keep W3 as the base for its document, code and reasoning gains, and repair the abstention
    in **Stage 4b DPO on W3**. W3 is staged as `oellm-upload/oellm-9b-256k-stage5-w3-rl-step160` (the RL
    checkpoint with the v2.1 template). Data jobs:
    - **Abstention repair (63249019):** W3 samples k=8 on the 31K eu_qa train prompts plus 600 fabricated prompts.
      "Known" means W2 answered the question consistently and correctly; for those, chosen = a correct answer and
      rejected = W3's UNKNOWN. That is 60 % of the pairs, the rest being UNKNOWN over a wrong guess.
    - **Identity (63249021):** the 768 identity-v2 prompts, re-sampled and judged on W3.
    - **Reduced safety (63249024):** a 2,001-prompt stratified subset of safety-v1 (762 comply, 667 refuse,
      572 grounded), k=4, judged by gemma4:31b.

    Estimate for the data: ~200 local-h. The 4b DPO mixes these with general Stage-4 pairs (~100 local-h on
    2 nodes). Its pair mix is set after the W2 abstention pilot reports.
  - **Abstention repair pairs built (63254527):** 2,222 pairs. 1,333 are known (chosen = a correct answer, rejected
    = W3's UNKNOWN; W2 knew 1,431 of the questions), 862 unknown (UNKNOWN over a wrong guess) and 27 fabricated (W3
    already abstains on nearly all of them).
  - **Chain queued 11-02 (all afterok):**
    1. assemble 63255365 (repair + identity + safety + 4,000 general pairs; waits for 63249021 and 63249024);
    2. DPO 63255366 (2 nodes, LR 5e-7, 1 epoch; config `dpo_stage4b_w3_leonardo.yaml`);
    3. export 63255367;
    4. evals: hallu-v1 63255368, EU suites 63255369, quick 63255370, tools 63255372, EvalPlus 63255373, EUROPA
       63255374, search QA 63255375.

    Estimate ~275 local-h after the data jobs.
  - **Requeued 11-02 21:00:** identity finished (689 pairs), but safety (63249024) hit its 5 h limit in the judge
    phase. It judged 1,000 of 10,512 samples (about 250 per hour; finishing would cost ~1,200 local-h). Safety pairs
    are built from the ~250 prompts judged so far (pairs + pretok, 63281619), and the chain is requeued: assemble
    63281620 → DPO 63281621 → export 63281622 → evals hallu-v1 63281624, EU 63281625, quick 63281627, tools
    63281628, EvalPlus 63281630, EUROPA 63281642, search QA 63281644. Speeding up the judge (Ollama parallelism or
    vLLM serving) is a precondition for a full safety refresh.
  - **4b gate, vs W3 and W2:**
    - eu_qa head-band accuracy ≥ 0.50 and abstention ≤ 0.10 in both modes;
    - standard hallucination rate below W2's 0.94;
    - fabricated inventions below W2's 0.79;
    - eu_doc, code and reasoning within 0.02 of W3;
    - tools and EUROPA no worse than W2.

  - **4b results (11-03): gate failed on the abstention repair; safety and identity improved.** Data: 2,110
    repair pairs, 689 identity, 46 safety (from 257 judged prompts), 4,000 general. DPO final reward accuracy 0.73,
    margin 0.19.

    | | W2 | W3 | 4b |
    |---|---|---|---|
    | eu_qa default: head acc / abstain | 0.53 / 0.03 | 0.005 / 0.995 | **0.03 / 0.97** |
    | eu_qa thinking: head acc / abstain | 0.52 / 0.01 | 0.38 / 0.42 | 0.41 / 0.38 |
    | eu_doc (default) | 0.376 | 0.516 | 0.514 |
    | quick reasoning thinking / default | 0.630 / 0.575 | 0.632 / 0.595 | 0.614 / 0.591 |
    | HumanEval+ thinking | 0.299 | 0.360 | 0.360 |
    | tools decision / names | 0.886 / 0.789 | 0.850 / 0.776 | 0.864 / 0.784 |
    | EUROPA Eval macro | 71.7 | – | 71.2 |
    | EUROPA Alignment macro / refusal calibration / failed items | 67.7 / 34.8 / 151 | – | **72.0 / 43.9 / 115** |

    - The DPO learned its pairs, but a 0.19 margin cannot move W3's default mode: its UNKNOWN mode is far deeper
      than W2's answer mode, which the same recipe moved easily (E4-pilot). DPO at this strength is not a repair
      for an RL-induced collapse.
    - Next (cheap, ~30 local-h): linear merges (1 − α)·W2 + α·4b at α = 0.5 (merge 63369335, eu 63369336) and
      α = 0.7 (63369337, eu 63369338). They test whether W2's healthy default mode comes back while 4b's safety,
      doc and code gains survive. The full suite runs on the winner only.
    - Search QA on 4b crashed on a stray think tag in a tool-call turn. The harness now strips it; rerun 63369113.

  - **W2 × 4b merges and 4b search QA (11-03).**

    | | default head acc / abstain | thinking head acc / abstain | eu_doc default / thinking |
    |---|---|---|---|
    | W2 | 0.53 / 0.03 | 0.52 / 0.01 | 0.376 / – |
    | merge α = 0.5 | 0.34 / 0.49 | 0.50 / 0.12 | 0.473 / 0.488 |
    | merge α = 0.7 | 0.03 / 0.98 | 0.43 / 0.37 | 0.500 / 0.540 |
    | 4b | 0.03 / 0.97 | 0.41 / 0.38 | 0.514 / 0.514 |

    - Merging trades the collapse against W3's gains: α = 0.5 half-recovers the default mode but loses 0.04 on
      eu_doc, and α = 0.7 is still collapsed. No merge passes both gates.
    - **4b with search, thinking mode: accuracy 0.50, wrong 0.14, fabricated inventions 0.04** (W2 with search:
      0.52 / 0.32 / 0.31). This is the most honest configuration so far: the same accuracy with half the wrong
      answers and almost no inventions. Default mode with search still abstains (accuracy 0.15, searches 64 %).
    - Conclusion: the default-mode collapse comes from the W3 RL reward and needs fixing in RL (ternary reward with
      known-prompt guards, search available; E8). Approved 11-03 (~400 local-h).
  - **E8 built (11-03).** Parent: the α = 0.5 merge (quick reasoning 0.640 / 0.588, tools 0.889, HumanEval+
    0.384 / 0.366). Pool (oellm-rlvr `scripts/build_e8_pool.py`, 2,560 prompts, 736 with the search tool): 640
    default-mode search QA (W2 strata 160 known / 160 mid / 320 unknown), 96 made-up entities with search, 448
    closed-book QA (288 known, the guard), 32 made-up closed-book, 1,344 replay. Search runs inside the rollout loop
    (one server per node, observation format tested token for token against template v2.1); QA is graded on the
    final turn only. Search rows are default mode only.
  - **E8 canary passed (63410422, 1 node, 2 updates, 17 min).** All 10 languages loaded in the search server.
    - 46 of 80 search-row rollouts called `wiki_search` and got Wikipedia excerpts back; their rewards span
      +1 / 0 / −1 (27 / 29 / 24), so there is learning signal.
    - Made-up entities with search: 3 of 8 still answered (−1).
    - Rows without the tool never called it; tool_call (0.66), math, code, doc and IF rewards look normal.
    - Malformed tool-call JSON from the model is logged by vLLM's Hermes parser and treated as no call.
  - **E8 full run queued 11-04: 63493790** (2 nodes, 160 updates, walltime 12 h, ~300–400 local-h).
  - **E8 trained (11-04, 2 h 13 min, 160 updates).** Training rewards, first 40 updates vs last 40:

    | Slice | First 40 | Last 40 |
    |---|---|---|
    | eu_qa with search | −0.00 | **+0.23** |
    | search rate on search rows | 58 % | 99 % |
    | eu_qa closed-book (incl. known guard, made-up entities) | 0.10 | 0.20 |
    | eu_doc | 0.50 | 0.54 |
    | code | 0.47 | 0.54 |
    | math | 0.64 | 0.67 |
    | multilingual math | 0.51 | 0.56 |
    | tool_call | 0.46 | 0.47 |
    | ifeval_direct | 0.57 | 0.54 |

    Export `oellm-9b-256k-stage5-e8-rl-step160`. Evals 63500803–63500840: hallu-v1, EU, quick, tools, EvalPlus,
    EUROPA, search QA, european-rl.
  - **european-rl L1 baselines (finance_amount_due, 300 tasks, 6 languages):**

    | | default | thinking |
    |---|---|---|
    | merge α=0.5 | 0.08 | 0.09 |
    | W2 | 0.06 | 0.10 |
    | W3 | 0.11 | 0.11 |

    - The errors are real arithmetic slips (e.g. 581.88 × 25.5 % written as 148.25), not parsing.
    - Levels 1–3 sit at 8–25 %, inside the learnable band; levels 4–5 are near 0.
    - Thinking mode leaves 22–43 % without an answer line within 6K tokens.
    - A first expert run should start at levels 1–3.

### 10.6 Hallucination (11-02)

Plan, measurements and experiments are in `OELLM_HALLUCINATION_PLAN.md`.

**Findings so far:**
- W2 knows more than Qwen3-8B on eu_qa (accuracy 0.28 vs 0.16) but almost never abstains: its hallucination rate is
  0.94.
- Its own self-consistency predicts correctness well (AUROC 0.87), so abstention training has large headroom.

**Consequences for this program:**
- RL windows from W4 keep eu_qa plus fabricated-entity prompts at 10–20 %, with "known" prompts as a guard against
  over-abstention.
- Stage 4b adds on-policy abstention pairs (sampled on W3) to the identity and safety pairs.
- The hallucination gate (plan §5) joins the standard gates.

### 10.7 Multi-teacher on-policy distillation (MOPD) as the integration step (proposed 11-03)

Source: Ma et al., *MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training*,
arXiv 2606.30406 (June 2026).

**The method:**
1. Train one RL specialist per domain from the same SFT checkpoint, in parallel.
2. Train a single student from that checkpoint on its own rollouts. Each prompt is routed to its domain's specialist,
   which scores every token. The loss is the per-token reverse KL.
3. The policy-gradient form is GRPO with one change: the per-token advantage is
   `clip(log π_teacher(y_t) − log π_student(y_t), ±5)`.

**Reported results (Qwen3-30B-A3B; math, instruction following, SWE):**
- Normalised score, where 1.0 = the specialists: MOPD 0.94, mixed-domain RL 0.88, task arithmetic 0.86, cascade RL
  0.78, linear weight averaging 0.33.
- MOPD reaches the teacher plateau after ~25–30K samples per domain; mixed RL needs 150–180K.
- Same-origin teachers are essential. Initial per-token KL is ~0.04 for same-origin teachers vs ~0.19 for an outside
  teacher (Qwen3-235B). The outside teacher made math worse, and its top-k variant collapsed at step 18.
- A second round (retrain specialists from the student, distil again) adds a further +0.05.

**Why it fits this program:**
- Our two integration tools are mixed RL windows and linear merges, the two baselines MOPD beats. We see the same
  failure modes:
  - W3 gained eu_doc and code but lost default-mode QA, and tools fell from 0.886 to 0.850.
  - The α = 0.5 W2 × 4b merge half-recovers QA but loses 0.04 on eu_doc.
- Our specialists are same-origin: one lineage, tokenizer and template.
  - W2: tools 0.886, healthy default-mode answering.
  - W3: eu_doc 0.516, code.
  - 4b: EUROPA Alignment 72.0, refusal calibration 43.9, identity.
  - E8: search and calibration, if it passes its gate.
- oellm-rlvr already has the machinery (`backend.kind: verl_opd`, `docs/on-policy-distillation.md`):
  - several teachers, routed by the row's `data_source`;
  - the policy-gradient `k1` loss with a clamp, i.e. MOPD's main form;
  - hybrid distillation + verifiable reward for single-turn rows.

**Constraints:**
1. **Distillation copies the teacher token by token, including its failures.** W2 answers 67 % of unknown facts
   wrongly, and W3 abstains on almost all of them. Neither may teach factual QA. eu_qa rows keep the verifiable
   +1 / 0 / −1 reward (hybrid mode), or use E8 as the teacher once it passes the hallucination gate.
2. **Multi-turn tool OPD is not implemented in the adapter.** Search stays in RL (E8); single-turn tool rows (A4b)
   can be distilled.
3. **verl is not installed on Leonardo.** The adapter has only passed the 2B contract smoke. Our `forward_kl_topk`
   is not the paper's bias-corrected reverse-KL top-k; the paper found the two forms close (0.94 vs 0.91), so the
   policy-gradient form is the default.
4. **The evidence is one preprint (v1) on one model family.** Our own pilot decides.

**Plan, after E8 (each step gated; approval before step 2):**

| Step | What | Cost |
|---|---|---|
| 1 | Install verl at the pinned commit on Leonardo; pass the CUDA 2B contract smoke | ~10 local-h |
| 2 | 9B pilot (details below) | ~150 local-h |
| 3 | If the pilot beats the α = 0.5 merge, a full integration window; E8 or its successor joins as the search/QA teacher | ~300–400 local-h |

The step-2 pilot:
- **Student:** the α = 0.5 merge.
- **Teachers:** W2 for tools, W3 for eu_doc and code, 4b for safety and identity prompts. They are frozen, and 2–3
  teachers share one A100 each for prefill.
- **Factual QA:** eu_qa and made-up entities keep the ternary reward instead of a teacher.
- **Batch:** N = 1 per prompt, advantage clip 5, ~15K samples.

**Pilot gate:** every integrated domain reaches ≥ 90 % of its teacher's gain over the student (the paper's
normalised score per domain), and the release checklist's NO-GO items pass. The paper's repeated rounds map onto
our window cadence: each round retrains specialists from the newest student, then distils again.


## 11. Next phase (planned 11-04)

**Where we stand:**
- **E8** fixed the default-mode collapse:
  - eu_qa head 0.54 / 0.10;
  - search QA: wrong 0.09, fabricated inventions 0.02;
  - tools 0.906, the best so far.
- **Still short for release:**
  - thinking-mode head abstain is 0.20 (gate ≤ 0.10);
  - EUROPA Eval is 68.5 vs 72.2 for the parent;
  - default-mode hallucination rate is 0.61 (proposed target 0.60).
- **E8's role:** a strong expert (honesty, search, tools), not yet a release candidate.
- **european-rl:** 36 languages, 3 environments, contract passed; the first baselines show arithmetic is weak (6–11 %).

| Phase | What | Cost (local-h) | Gate / decision |
|---|---|---|---|
| A. Close E8 | Collect quick reasoning and EUROPA Alignment; full release-checklist scoring; per-language and per-band error analysis | 0 (running) | E8 confirmed as the honesty/search/tools expert |
| B. MOPD integration pilot (§10.7) | B1: install verl on Leonardo, 2B contract smoke. B2: measure teacher–student per-token KL per domain (same-origin check). B3: 9B pilot, details below | ~10 + ~10 + ~150 | **approval for B3**; pilot beats both parents on the checklist |
| C. european-rl into training | C1: L1 baselines for formal_email (typed, ready) and meeting_reschedule (needs a multi-turn eval harness). C2: tool-episode environments served to RL like `wiki_search`. C3: first **EU office expert** run: finance levels 1–3, formal_email, meeting_reschedule | ~15 + 0 + ~300 | **approval for C3**; L3 (held-out +5 points, positive lower bound) on each environment, no checklist regression |
| D. Quality of the European data | Native review for the highest-traffic languages (partners); a person verifies the VAT snapshot; new holiday rule kinds (Sunday skip, observed shift, every-N-years); 2026 snapshot (BG euro, LT, RO, NL, BE changes) | 0 compute | review status per language moves from `llm` to `native:` |
| E. More environments | E-1 shortcut probes for the existing oellm-rlvr pools; office world: payment registration, invoice approval, GDPR access/erasure with a simulated customer; search in thinking mode (E-4); long dossiers (E-5) | CPU + small canaries | each passes contract v0.3 before use |
| F. Release track | When an integrated candidate passes every NO-GO item: oellm-eval retention (~180), red-team session, model card, licence and EU AI Act paperwork | ~250 | **public release needs explicit approval** |

**The phase-B3 pilot:**
- **Student:** E8 (strongest overall).
- **Teachers, same lineage:** 4b for safety and identity prompts (Alignment 72.0); W3 for documents and thinking-mode code; the 50/50 merge for EUROPA-style general capability.
- **Factual QA and search** keep the +1 / 0 / −1 reward, which also targets thinking-mode over-abstention.

**Order:** A → B1/B2 and C1 in parallel (cheap) → B3 → C3 from B3's student if B3 passes, otherwise from E8 → F. D and E run alongside, without compute.

### 11.1 Decisions (11-04)

- The MOPD pilot (phase B) is approved.
- The EU office expert run (phase C3) is to be planned.
- European text continues without native review (`_review: llm`).
- No release date is set.
- Arithmetic is added as an environment (european-rl `business_arithmetic`): the finance baselines failed mostly
  through exact decimal arithmetic (581.88 × 25.5 % → 148.25), which the existing math pools do not train.

### 11.2 Plan: E9, the EU office expert (phase C3)

**Purpose.** Train the office skills the European environments measure:
- exact business arithmetic in local formats;
- invoices under national VAT rules;
- formal correspondence under language-specific conventions;
- cross-country scheduling with tools (once served to RL);
- without losing E8's honesty, search and tool skills.

**Parent.** The MOPD pilot student if the pilot passes its gate (§10.7), otherwise E8.

**Pool (2,560 prompts, 160 updates × 16, like the W windows):**

| Slice | Prompts | Source | Reward |
|---|---|---|---|
| business_arithmetic | 640 | all levels and task types | `eurl_typed` (exact minor unit) |
| finance_amount_due | 512 | levels 1–3 (+ 64 level 4) | `eurl_typed` |
| formal_email | 384 | 2–4 constraints | `eurl_typed` (strict) |
| meeting_reschedule | 256 | only if the tool-episode service (C2) is ready; otherwise the 256 go to arithmetic and finance | end state |
| eu_qa honesty replay | 256 | E8 pool, search and closed-book (known guard) | +1 / 0 / −1 |
| math, IF, code, tools replay | 512 | unused rows | as in W3/E8 |

**Coverage.**
- **Languages:** spread over all 36 languages, weighted 2:1 towards the 24 EU official languages. Each language
  has at least 40 european-rl prompts.
- **Modes:** 60 % default, 40 % thinking. Thinking answers must close within the 7K window; the baselines left
  20–40 % without an answer at 6K.

**Difficulty.**
- Pass-rate profile on the parent (k = 8, `oellm-rlvr pass-rate-sample/score/select`): keep prompts with in-group
  variance (0 < passes < 8).
- Levels are rebalanced towards the window that still has signal, as the Nemotron curriculum does.

**Preconditions.**
1. L1 baselines on E8 and the merge for formal_email and business_arithmetic (running, 63509254/6). The rest of
   business_arithmetic follows after its 30-language translation.
2. The `eurl_typed` verifier (oellm-rlvr, done).
3. The pass-rate profile (~20 local-h).
4. For meeting_reschedule, a multi-turn eval harness and the RL tool service (C2).

**Cost.** ~300 local-h (W3/E8 scale) + ~20 profile + ~60 evals.

**Gate.**
- L3 on each european-rl environment: a held-out eval split, +5 points, positive lower bound of the paired 95 %
  bootstrap interval.
- No regression on the release checklist's NO-GO items against the parent, in particular:
  - eu_qa honesty (head band, hallucination rate, fabricated entities);
  - tools, code, reasoning, EUROPA.
