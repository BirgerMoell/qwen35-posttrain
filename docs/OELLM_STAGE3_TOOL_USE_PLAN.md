# Stage 3 — Tool-use SFT: plan

Status 2026-10-21. Starts from the Stage-2 (reasoning) selected checkpoint. Goal: reliable function calling in
both modes, correct *decisions* about when to call (call / answer directly / ask for missing info / say it cannot),
multilingual user turns, and no regression in instruct or reasoning quality.

## 1. Template v2.1 — done

`templates/oellm_qwen3_two_stage_v2_1.jinja`, tests in `tests/test_chat_template_v2_1.py`.

- Qwen3/Hermes format, so vLLM's `hermes` tool parser works unchanged: tool signatures in `<tools>` in the system
  turn; assistant calls `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>` (parallel calls one per block);
  consecutive tool results grouped into one `user` turn as `<tool_response>` blocks.
- Think switch as v2. Reasoning is rendered for **every assistant turn after the last user message** (all steps of
  the current tool loop) and dropped before it.
- Supervised: assistant reasoning, content, tool calls, `<|im_end|>`. Never: the `<think>` opener, tool lists,
  tool responses.
- Verified: tool-free rows render with identical token IDs and masks to v2 (2,843 real rows from
  openeurollm-reasoning-instruct-v1); all tool tags are single tokens in our tokenizer
  (`<tool_call>`=11, `</tool_call>`=12, `<tools>`=13, `</tools>`=14, `<tool_response>`=15, `</tool_response>`=16).
- Still to do: a vLLM round-trip test on the Stage-2 checkpoint (template → generation → `hermes` parser →
  structured call) once vLLM is available on Leonardo/LUMI.

## 2. Dataset: `birgermoell/openeurollm-tool-use-v1` (public)

### Canonical schema (HF / OpenAI standard, consumed directly by v2.1)
| Column | Content |
|---|---|
| `id`, `source`, `source_dataset` (repo@revision), `license`, `language` | provenance, per row |
| `category` | `single_call`, `parallel_call`, `multi_step`, `multi_turn`, `no_call_direct`, `ask_for_info`, `cannot_answer`, `call_then_refuse` |
| `mode` | `reasoning` (every assistant turn after the last user turn has a trace) or `direct` (no traces anywhere) |
| `tools` | JSON list of `{"type": "function", "function": {name, description, parameters}}` |
| `messages` | `system`/`user`/`assistant`/`tool`; assistant may carry `tool_calls` (arguments as objects) and `reasoning_content` |

### Sources and adapters (licenses from the HF cards, 2026-10-21)
| Source | License | Native format → adapter | Use |
|---|---|---|---|
| [birgermoell/oellm-eu-tooluse-v1](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1) (55K) | Apache-2.0 | pre-rendered Hermes text → parse `<tools>` and `<tool_call>` back to structure | core single/parallel calls |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 | `tools` (JSON strings) + messages; SFT file + preference file | no-call / ask / cannot categories; preference file → Stage 4 |
| [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) `tool_calling` (15 GB) | CC-BY-4.0 (per subset) | OpenAI `tool_calls` + `reasoning_content` | multi-step loops **with traces** — main thinking-mode source |
| [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) (22 GB) | Apache-2.0 | OpenAI tools JSON + messages | real MCP executions, tool diversity; sample by `subset_name` |
| [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) (+ Web-QA) | Apache-2.0 | `conversations` with `tool_calls`, `role: tool` | multi-step search; the fcanalysis staging adds 4.5K audited web trajectories |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | tools inlined in system text, pythonic `[fn(a=1)]` calls → parse | API diversity, dedup against eu-tooluse |
| [openbmb/UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609) (54 GB) | Apache-2.0 | messages + tools | optional sample (virtual tools, DB/finance workflows) |
| Excluded | | [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) is CC-BY-NC-4.0; `xlam-function-calling-60k` is gated — not in an open release | |

### Normalization and filters (one builder, same style as the Stage-1/2 builders)
1. Adapter per source → canonical schema; tool definitions normalized to JSON Schema (`dict`→`object`, `str`→`string`).
2. **Call validity**: every call names a declared tool, arguments are a JSON object that validates against the tool's
   schema (required keys present, types correct); otherwise drop.
3. **Trajectory validity**: every call is followed by a tool message (except the final assistant turn), the
   conversation ends with an assistant turn without calls, no tool message without a preceding call.
4. **Mode rule**: `reasoning` only if every assistant turn after the last user turn has a nonempty trace; otherwise all
   traces are removed (`direct`). Inline `<think>` normalized into `reasoning_content` (Stage-2 `normalize`).
5. Identity hygiene (foreign assistant names unprompted → drop), `<image>` placeholders → drop, ≤32K tokens
   (reject, never truncate), length-capped tool responses (truncate *tool output* with an explicit `…[truncated]`
   marker only if > 4K tokens, never assistant text).
6. Render with v2.1 + our tokenizer and audit: supervised tokens contain no `<think>`, no `<tool_response>`, no tool
   list; each supervised `<tool_call>` block parses back to the stored call (round-trip).
7. Dedup by normalized user prompt + tool-name set across sources; decontaminate against BFCL v4, When2Call test,
   τ²-bench and MASSIVE-Agents prompts.

### Mixture targets (~200M tokens of tool data)
| Category | Share | Notes |
|---|---|---|
| single + parallel call | 30 % | eu-tooluse, ToolACE, Toucan single-turn |
| multi-step / multi-turn loops | 35 % | Nemotron-Agentic (with traces), Toucan multi-turn, ToolMind |
| **no call needed** (tools present, answer directly) | 15 % | When2Call `direct` + synthetic: attach irrelevant tools to Stage-2 direct rows |
| ask for missing info / cannot answer | 10 % | When2Call; moderate — the paper warns pure When2Call SFT over-corrects |
| call then refuse / unsafe tool requests | 5 % | build from WildGuardMix-style prompts + tools (refuse to call) |
| multilingual (translated user turns + final answers) | 5–10 % overlay | see below |
Thinking-mode share ≈ 50 % of tool rows (Nemotron-Agentic traces; for sources without traces, a subset gets traces
generated by a teacher in a later version — v1 keeps them direct).

### Multilingual slice
All candidate sources are English (ToolACE adds Chinese). Translate **user turns and final assistant text only** —
never tool names, schemas, JSON keys or tool outputs — for ~10–20K rows into the 12 Dolci languages + da/no/is/pt,
then re-validate (arguments that are user-facing strings, e.g. city names, stay consistent with the query). Use the
same translator family as Dolci-translated (Gemma) and record `translation_model` per row.

### Hosting
Public HF dataset `birgermoell/openeurollm-tool-use-v1`: Parquet shards by `source`, `manifest.json` with counts,
rejection reasons and pinned revisions, dataset card in the style of openeurollm-reasoning-instruct-v1 (per-row
license column; NC sources excluded). Pre-tokenized training mix built in the same pass (`data/stage3-mix-v1`).

## 3. Training
- Init: Stage-2 selected checkpoint; template v2.1; tokenizer unchanged.
- Mix: tool data ~55 % + Stage-2 replay ~45 % (half reasoning, half direct; includes identity and safety rows).
  ~350–450M tokens total → ~350–450 steps at 1M tokens/step, ≈ 70–100 A100 GPU-h.
- LR 3e-6 cosine, 1 epoch, 32K BFD packing, `SFT_FORBID_LABEL_IDS=18,15` (no supervised `<think>` or
  `<tool_response>`).

## 4. Evaluation and promotion gates
- **BFCL v4** (simple, multiple, parallel, multi-turn, irrelevance) — run locally from
  [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard).
- **When2Call** test set: call / direct / ask / cannot confusion matrix and tool-hallucination rate.
- Multilingual function calling: translated BFCL subset + MASSIVE-Agents-style items in our target languages.
- Format: % outputs whose `<tool_call>` blocks parse with vLLM's `hermes` parser; no calls when no tools are declared.
- Retention vs Stage 2: EUROPA Eval and Alignment (reasoning off and on), generation check in both modes
  (0 think tags in default mode, closed traces in thinking mode), identity check.
- Promote only if BFCL/When2Call improve and retention is within noise; otherwise keep Stage 2 as the selected
  checkpoint for DPO.

## 5. Order of work
1. ✅ Template v2.1 + tests + real-data identity check.
2. Adapters + validator for eu-tooluse, When2Call, Nemotron-Agentic, Toucan, ToolMind, ToolACE (unit tests per adapter
   on 20 real rows each).
3. Stage on Leonardo scratch (pinned revisions), build canonical dataset + rejection report; human spot-check 50 rows
   per category.
4. Multilingual translation slice (GPU job; Gemma translator), re-validate.
5. Publish `openeurollm-tool-use-v1`; build `stage3-mix-v1`.
6. Train (after Stage-2 evals select the start checkpoint), then evals and gates above.
