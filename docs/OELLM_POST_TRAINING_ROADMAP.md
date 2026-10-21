# OpenEuroLLM 9B post-training roadmap: reasoning → tool use → DPO → RL

Status 2026-10-21. Stage 1 (Instruct SFT) is done; Stage 2 (reasoning SFT) is building/queued.
Every stage starts from the previous **selected** checkpoint: a candidate is promoted only when its
paired evals pass the retention gates below; otherwise the next stage starts from the previous one
(the "selected vs candidate" rule from `oellm-rlvr`).

| Stage | Starts from | Purpose | Data (main) | Budget |
|---|---|---|---|---|
| 1 Instruct SFT ✅ | 256K anneal300b base | direct answers, 19+ languages | stage1-mix-v1 (3.1M rows) | ~395 GPU-h (done) |
| 2 Reasoning SFT ⏳ | Stage 1 | opt-in thinking mode, keep default mode | openeurollm-reasoning-instruct-v1 | ~450–500 GPU-h |
| 3 Tool-use SFT | Stage 2 selected | function calling in both modes, *when* to call | tool mix + 40–50 % replay | ~100–150 GPU-h |
| 4 DPO | Stage 3 selected | safety, format discipline, multilingual preference | threeway sources (subsampled) + multilingual + targeted pairs | ~250–350 GPU-h |
| 5 RLVR | Stage 4 selected | verifiable math / IF / tool-call accuracy | oellm-rlvr task pools | campaign-gated |

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

**Our Stage-4 plan**
- Reference and initialization = the Stage-3 selected checkpoint (same tokenizer).
- Pairs (text-only, re-rendered with v2.1 in default mode): the three threeway sources, but **subsample the
  Dolci-DPO-translated block** (its 93.8 % share is many translations of the same prompts) to ~400–500K pairs with one
  language per prompt family (as in Stage 1), keeping HelpSteer3 and UltraFeedback whole; plus

- **More data**: [`jonasaise/dpo-*-interpreted`](https://huggingface.co/jonasaise) (sv, da, no, is, fi, de, fr, es, it, pl;
  MIT; UltraFeedback-derived, Kimi-K3 adapted — dedup against UltraFeedback prompts), `allenai/Dolci-Think-DPO`-style
  pairs for thinking mode, When2Call preference split, plus **targeted synthetic pairs from our own failures**:
  strict format (`15` ≻ `18 - 7 + 4 = 15`; `INSUFFICIENT` ≻ translated keyword; `B` ≻ `B. explanation`),
  refusal (refuse harmful ≻ comply; help benign-but-scary ≻ over-refuse), identity (OpenEuroLLM ≻ OpenAI claim).
  Generate rejected samples on-policy from the Stage-3 checkpoint where possible.
- Both modes: thinking-mode pairs compare final answers with the trace kept identical or regenerated; never prefer
  empty think blocks.
- Method: the threeway recipe (sigmoid DPO, β 0.1, LR 5e-7, batch 128) with max length raised to 4,096 so
  thinking-mode pairs fit; ~550–650K pairs ≈ 4.5–5K steps ≈ 250–350 A100 GPU-h on Leonardo (vs ~700 for the full 1.44M).
  SimPO (`scripts/simpo_train.py`) as an ablation only.
- Gates: EUROPA Alignment ≥ 0.85, EUROPA Eval ≥ Stage 3, over-refusal check (benign_non_refusal), response length
  not inflated vs Stage 3.

## Stage 5 — RLVR (`oellm-rlvr`)

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
