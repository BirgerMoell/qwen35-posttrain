# OpenEuroLLM 9B post-training roadmap: reasoning → tool use → DPO → RL

> The complete, reproducible record of what was actually run (data, configs, hashes, jobs, costs, results, pitfalls)
> is [`OELLM_POST_TRAINING_PLAYBOOK.md`](OELLM_POST_TRAINING_PLAYBOOK.md); this file is the plan.

Status 2026-10-21. Stage 1 (Instruct SFT) is done; Stage 2 (reasoning SFT) is building/queued.
Every stage starts from the previous **selected** checkpoint: a candidate is promoted only when its
paired evals pass the retention gates below; otherwise the next stage starts from the previous one
(the "selected vs candidate" rule from `oellm-rlvr`).

| Stage | Starts from | Purpose | Data (main) | Budget |
|---|---|---|---|---|
| 1 Instruct SFT ✅ | 256K anneal300b base | direct answers, 19+ languages | stage1-mix-v1 (3.1M rows) | ~395 GPU-h (done) |
| 2 Reasoning SFT ⏳ | Stage 1 | opt-in thinking mode, keep default mode | openeurollm-reasoning-instruct-v1 | ~450–500 GPU-h |
| 3 Tool-use SFT | Stage 2 selected | function calling in both modes, *when* to call | tool mix + 40–50 % replay | ~100–150 GPU-h |
| 4 DPO | Stage 3 selected | safety, format discipline, multilingual preference | threeway-run replication (1.44M pairs), corrected tokenizer/template | ~600–750 GPU-h |
| 5 RLVR | Stage 4 selected | verifiable math / IF / tool-call accuracy | oellm-rlvr task pools | campaign-gated |

## Compute budget (decided 2026-10-22)

The Leonardo account `aifac_f02_352` is shared; this project's share is **~50K local-h** (1 GPU-h = 8 local-h on
boost_usr_prod). Spent so far (`sacct`, own jobs since 2026-10-20): **~6.6K local-h** (Stage 1 3.2K, Stage 2 2.8K,
tool translation 0.45K, evals/smokes ~0.14K). Per-stage caps:

| Stage | Cap (GPU-h) | Cap (local-h) | Notes |
|---|---:|---:|---|
| 3 Tool-use SFT | ~100 | ~800 | running; ~470M tokens, one mid-run checkpoint |
| Gate evals per stage | ~15 | ~120 | EUROPA Eval + Alignment, quick reasoning eval |
| 4 DPO | ~250 | ~2,000 | threeway recipe on a subsample (~400K Dolci-translated pairs + HelpSteer3 + UltraFeedback + targeted pairs); the full 1.44M replication (~5K local-h) only with explicit approval |
| 5 RL | ~200 | ~1,600 | Tier A only (math, IF, strict format, single tool calls), two windows; Tiers B/C only if A clearly helps and budget remains |
| **Planned total** | | **~11K** | leaves headroom in the shared 50K |

Every job above ~20 GPU-h gets a local-h estimate before submission.

## Baseline facts that shape the plan

Stage-1 eval vs the earlier Neonkraft SFT (same base): EUROPA Eval 0.600 vs 0.762, Alignment 0.455 vs 0.773.
Causes: (a) strict-format failures — keyword translated, worked solutions instead of "only the number",
extra text after MCQ letters; (b) **zero refusal data** in Stage 1 (101 refusals in 3.15M rows).
Strengths: no think leakage, clean termination, 19-language fluency, long-context revision tracking 0.889 vs 0.694.
Stage 2 already adds precise-IF reasoning (Dolci Persona/Precise IF) and safety reasoning
(WildJailbreak, WildGuardMix, CoCoNot), so Stages 4–5 start from a better place.

## Stage 3 — Tool-use SFT

### Template v2.1 (prerequisite)
v2 deliberately rejects tools. v2.1 adds, without changing any existing rendering:
- `tools` rendered in the system turn in the Qwen/Hermes form (`<tools>…</tools>` + call instructions);
  assistant calls as `<tool_call>{"name":…, "arguments":…}</tool_call>` inside the supervised span;
  tool results as `<|im_start|>user\n<tool_response>…</tool_response>` (Qwen3 convention; vLLM `hermes` parser).
- Think switch unchanged: `enable_thinking=True` pre-fills `<think>\n` on the generation prompt.
- **Reasoning retention rule changes for agent loops**: keep `reasoning_content` on every assistant turn
  *after the last user message* (the turns of the current tool loop); drop it for turns before it. Stage 1/2 rows
  (single final answer) render identically, so the test suite must prove byte-identity for all existing data.
- Tests: tool rendering round-trip, mask covers only assistant text + tool calls (never tool responses),
  multi-call turns, think + tool-call ordering, and v2/v2.1 identity for tool-free rows.

### Data (candidates; all licenses checked on the HF cards 2026-10-21)
| Source | License | Role |
|---|---|---|
| [`birgermoell/oellm-eu-tooluse-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1) (55K SFT, 46K GRPO) | Apache-2.0 | core single/multi-call, already in Hermes form; GRPO split → Stage 5 |
| [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 | **when not to call**: direct / ask-for-info / cannot-answer; fixes the earlier tool model's over-calling. Mix moderately — the paper warns pure When2Call SFT makes models over-conservative |
| [`nvidia/Nemotron-SFT-Agentic-v2`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) / [`Nemotron-Agentic-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | CC-BY-4.0 / Apache / MIT per subset | large multi-turn, multi-step and customer-service trajectories (sample, don't take 1.2M) |
| [`Agent-Ark/Toucan-1.5M`](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Apache-2.0 | real MCP-server executions (≈495 servers, 2K+ tools); sample for tool diversity |
| [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE), [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Apache-2.0 | API diversity (already partly inside oellm-eu-tooluse-v1 — dedup) |
| [`Nanbeige/ToolMind`](https://huggingface.co/datasets/Nanbeige/ToolMind) / ToolMind-Web-QA | Apache-2.0 | multi-step search trajectories with reasoning (your fcanalysis staging has 4.5K audited) |
| [`openbmb/UltraData-SFT-Agent-2609`](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609) | Apache-2.0 | recent (Sep 2026) 500K agent SFT incl. virtual tools/DB — sample and audit |
| `Salesforce/xlam-function-calling-60k` | CC-BY-4.0, gated | optional; **`APIGen-MT-5k` is CC-BY-NC-4.0 → exclude** from open releases |

**Multilingual gap**: every tool source is English (ToolACE adds Chinese). Plan: translate the *user turns and
final answers* (not tool names/JSON) of a ~10–20K slice into the 12 Dolci languages + Nordic set, keeping
arguments language-correct (city names, dates). Evaluate with MASSIVE-Agents-style multilingual function calling.

### Mix and training
- ~150–250M tokens: tool trajectories ~55 % (with ≥15 % no-call/irrelevance/ask-for-info cases, both modes),
  reasoning replay from Stage 2 ~25 %, direct replay ~20 % (identity v1+v2 included).
- Think mode share in tool rows: ~50 % (traces before calls where the source has them; otherwise direct calls).
- LR ~3e-6, 1 epoch, 32K packing; same audits plus: every `<tool_call>` parses as JSON and names a declared tool;
  arguments validate against the tool schema; tool responses are never supervised.

### Evals / gates
BFCL v4 (incl. irrelevance + multi-turn), When2Call (call vs no-call confusion matrix, hallucination rate),
τ²-bench if feasible, multilingual tool-call set (translated BFCL/MASSIVE-Agents subset).
Retention: EUROPA Eval within noise of Stage 2, both modes; think-tag rate in default mode = 0;
no tool call emitted when no tools are declared.

## Stage 4 — DPO

**Reference recipe**: [`birgermoell/oellm-9b-threeway-dpo-full-v1`](https://huggingface.co/birgermoell/oellm-9b-threeway-dpo-full-v1)
(from the Neonkraft SFT): TRL DPO, sigmoid loss, β 0.1, LR 5e-7 cosine, 3 % warmup, global batch 128, max length 2,048
(truncated pairs dropped), 1 epoch = 11,263 steps on 1.44M pairs, ~1,387 MI250X GCD-h. Result: EUROPA Eval 76.2 % → 84.0 %,
EUROPA Alignment 0.893, held-out preference accuracy 0.878.

| Source (threeway run) | Train pairs | Share | License |
|---|---:|---:|---|
| [openeurollm/Dolci-Instruct-DPO-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-DPO-translated) | 1,352,763 | 93.8 % | Apache-2.0 |
| [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 59,889 | 4.2 % | MIT |
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) `preference` | 28,895 | 2.0 % | CC-BY-4.0 |

**Tokenizer incompatibility — reuse the data and recipe, not the weights.** The threeway model's tokenizer
appends ChatML/think tokens at IDs 262144–262151 (`<|im_start|>`=262144, `<think>`=262151). Our lineage maps them onto
native slots (`<|im_start|>`=3, `<|im_end|>`=4, `<tool_call>`=11, `<tools>`=13, `<tool_response>`=15, `</think>`=17,
`<think>`=18). Its checkpoint therefore cannot serve as our reference model or initialization, and its tokenized data
must not be reused: re-render all pairs from text with our tokenizer and template.

**Never use the threeway repo's `chat_template.jinja` or tokenizer files** — shipping them was the actual error in that
release (it also emits empty `<think>` blocks). Every stage uses only our tokenizer and `templates/oellm_qwen3_two_stage_v2*.jinja`;
a stage's data build must fail if the template or `tokenizer.json` sha256 differs from the pinned one.

**Our Stage-4 plan (decided 2026-10-21): replicate the threeway run, fix only the tokenizer/template.**
- Same data as the threeway run, at full size and with all translations: Dolci-Instruct-DPO-translated (~1.35M),
  UltraFeedback-cleaned (~60K), HelpSteer3 preference (~29K); same trainer filters; same validation split rule.
- Same recipe: TRL DPO, sigmoid loss, β 0.1, LR 5e-7 cosine, 3 % warmup, global batch 128, max length 2,048
  (truncated pairs dropped), 1 epoch, seed 42.
- Only changes: (1) pairs rendered from text with **our tokenizer and template v2.1** (`<think>`=18, `</think>`=17,
  default mode, no empty think blocks) — never the threeway repo's tokenizer/template; (2) policy init and reference =
  the Stage-3 selected checkpoint; (3) `scripts/dpo_train.py` gets the same guards as SFT (explicit tokenizer +
  `CHAT_TEMPLATE_FILE`, template/tokenizer hash check, abort on supervised `<think>`).
- Cost: ~1.44M pairs ≈ 11.3K steps; on Leonardo A100s roughly 600–750 GPU-h (the LUMI run was ~1,387 MI250X GCD-h).
- Optional follow-up (separate, small DPO ablation, not part of the replication): targeted format/refusal/identity
  pairs, jonasaise multilingual pairs, When2Call preference pairs, thinking-mode pairs.
- **Format-compliance pairs (decided 2026-10-22, part of the Stage-4 mix, not optional).** Stage 3 scores 0.616 on
  EUROPA Eval vs 0.762 (Neonkraft SFT) and 0.840 (threeway DPO) almost entirely on strict-format items: numeric 0.06
  vs 0.58, tool-recovery letter 0.49 vs 0.92, 122 malformed answers vs 38 / 9. Cause in our instruct data: of 9,685
  strict-format user turns ("answer only with the number/letter/name") in 3.15M rows, 99 % of targets ignore the
  constraint (mostly OpenEuroLLM-Excellent-SFT zebra puzzles: "The person … is **Alice**. Step-by-step explanation…").
  Build multilingual pairs on held-out strict-format prompts (number, letter, single word, JSON-only, N lines, no
  explanation) in the 36 languages: *chosen* = the bare compliant answer (verified correct), *rejected* = the Stage-3
  model's own verbose or wrong answer (on-policy). Keep EUROPA items out of the prompt pool (no eval leakage); check
  that refusal and explanation quality do not drop (answers must stay full when no constraint is given).
- Next instruct-data version: rewrite or drop the 9.4K non-compliant strict-format targets and add a small
  multilingual strict-format SFT set, so a future SFT does not relearn the habit.
- Gates: EUROPA Alignment ≥ 0.85, EUROPA Eval ≥ Stage 3 (target: malformed answers ≤ 40), over-refusal check
  (benign_non_refusal), response length not inflated vs Stage 3, no length collapse on unconstrained prompts.

## Stage 5 — RLVR (`oellm-rlvr`)

**Detailed plan: [`OELLM_STAGE5_RL_PLAN.md`](OELLM_STAGE5_RL_PLAN.md)** (data pools, difficulty windows, rewards, gates).
PrismMath is used here as an RL problem pool, not in the reasoning SFT data.

- Backend: TMAX/Open-Instruct DAPO for math/IF/reasoning (qualified on OELLM 9B on LUMI); SkyRL + Harbor for
  agentic/tool tasks (backend qualified; OELLM checkpoint not yet). CUDA profile exists for Leonardo.
- Chat-template contract: the model's v2.1 template, thinking mode on for reasoning domains; reward parsers must
  read only the text after `</think>` and treat a missing `</think>` as failure (loop guard).
- Domains & verifiers, in order (each with measured-difficulty bins, 8 samples/prompt, saturated and 0/8 excluded):
  1. multilingual math (`birgermoell/oellm-math-rlvr`, existing EU math pools),
  2. precise IF with programmatic constraint checkers (IFEval-style, multilingual),
  3. tool calls: `oellm-eu-tooluse-v1` GRPO split — reward = parsed JSON, correct tool name, argument
     match (ToolRL-style fine-grained format + name + parameter credit), plus no-call correctness from When2Call,
  4. code with hidden tests (`oellm-code-rlvr`).
- Known lesson from prior runs: gains so far are small (multilingual math 36.3 % → 38.9 %, p≈0.06) and the
  nominal mix was not the realized mix — verify realized pool ratios, use paired held-out evals and McNemar,
  and keep the selected-checkpoint rule.
- Safety: re-run EUROPA Alignment after RL; if it regresses, a small final DPO safety patch (not a full stage).

## Release hygiene (all stages)
Pinned source revisions, per-row source/license, deterministic selection, rejection reports, pre-tokenized data
equal to the audited tokens, private model uploads until eval + license review, one template file per stage
(v2 → v2.1) with tests proving existing data renders identically.
