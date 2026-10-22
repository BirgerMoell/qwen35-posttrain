# Stage 5 — RL run program

Status 2026-10-23. This is the **run-by-run** plan for Stage 5: which RL runs we make, in what order, what each one
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
| Reasoning in the user's language | not yet measured | ≥ 90 % of traces in the prompt language, accuracy cost ≤ 2 points | EUROPA Reasoning trace-language metric |
| Direct-mode answers and strict formats | EUROPA numeric 0.056, 122 malformed | numeric ≥ 0.5, malformed ≤ 20 | EUROPA Eval |
| Tool use | decision 0.870, JSON 0.995; EUROPA tool-recovery 0.486 | decision ≥ 0.92, recovery ≥ 0.8 | held-out tool eval, EUROPA Eval, BFCL |
| Retention | EUROPA Eval 0.616 / Alignment 0.642 (before DPO) | no regression vs the Stage-4 checkpoint | EUROPA Eval/Alignment, identity, default-mode think-tag rate = 0 |

## 2. Budget envelope

Working budget 50K local-h (account total 400K, shared). Used by 2026-10-23: ~7.7K. Planned DPO: ~5–7K.
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

### RL-0 — Think probe (running: job 61868620)

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

Each runs 32 updates at 16–32 prompts × 8 samples, 6K max response, against the RL-0 settings as the baseline arm.

| Run | Question | Arms | Data | Cost | Decides |
|---|---|---|---|---|---|
| **A1 language** | Can RL make the model reason in the prompt language, and what does it cost in accuracy? | (a) baseline; (b) answer-language multiplier; (c) answer + trace-language multiplier | oellm-math EU24 + exam-MCQ, non-English only | 2 × ~150 local-h (arm a = RL-0) | whether W1–W3 use the trace-language multiplier, and for which languages (drop any language that loses > 3 points) |
| **A2 length** (only if RL-0 does not fix closure) | Which shaping fixes termination? | (a) DAPO soft overlong punishment; (b) (a) + loop penalty (n-gram repeat ratio ⇒ 0) | RL-0 data | 2 × ~150 local-h | the structure guards for all windows |
| **A3 both modes** | Does mixing default-mode prompts keep and improve the direct mode? | (a) thinking only; (b) 70 % thinking + 30 % default-mode strict-format and numeric prompts | math + strict-format set | ~150 local-h (arm a = RL-0) | the default-mode share in W1–W3; expected to fix EUROPA numeric |
| **A4 tools single-turn** | Does the ToolRL reward on the 46K GRPO split improve held-out tool decisions? | one arm | oellm-eu-tooluse-v1 `grpo` + no-call items | ~150 local-h | whether tools enter W2 or wait for T1 |

Ablation total: ~600–900 local-h.

### W1 — Warm-up window (Stage-4 DPO checkpoint)

| | |
|---|---|
| Question | Does the RL-0 + ablation recipe transfer to the DPO checkpoint, at the production batch size? |
| Data | math (oellm EU24 ≥ 40 % non-English, DeepMath 3–5), exam-MCQ (35 languages, letter-only answers), strict format, IF ≤ 2 constraints; 30 % easy / 60 % medium / 10 % hard by measured pass@8 |
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

oellm-eu-tooluse-v1 `grpo` (46K, 16 languages), Nemotron function-calling pivot, no-call items; ToolRL reward;
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

## 8. Order of work (next two weeks)

1. RL-0 finishes; read the closed-rate curve and the end eval; record in the playbook.
2. While DPO trains: build the A1/A3 data (non-English math + exam-MCQ slice; strict-format numeric set), add the
   language and loop guards to `oellm-rlvr` verifiers with adversarial tests, run A1 and A3 (and A2 if needed).
3. Build W1 pools and the profiling job; profile on the DPO checkpoint as soon as it passes its gates.
4. W1 → W2 → W3, each gated; T1 inside W2 if A4 passes.
5. In parallel with W2/W3: SkyRL qualification canary on Leonardo, mock-tool and Wikipedia-search environments.

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
