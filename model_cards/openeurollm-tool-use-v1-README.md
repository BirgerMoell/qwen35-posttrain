---
license: other
license_name: mixed-upstream-see-card
language: [en, cs, da, de, el, es, fi, fr, is, it, nl, "no", pl, pt, ro, sv, uk]
task_categories: [text-generation]
tags: [tool-use, function-calling, agentic, sft, openeurollm, multilingual, conversational]
size_categories: [100K<n<1M]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# OpenEuroLLM Tool-Use v1

Tool-use / function-calling conversations for supervised fine-tuning, normalized from seven open sources into one
schema and validated end to end. Built for Stage 3 of the OpenEuroLLM 9B post-training (after instruct and reasoning
SFT), with an explicit focus on **when not to call a tool**.

- **158,703 conversations** (148,285 English + 10,418 translated into 16 European languages), ~328M tokens.
- Categories: single call 50,213 · multi-turn 20,778 · multi-step 18,832 · parallel calls 12,014 ·
  **ask for missing information 19,542 · cannot answer with the given tools 13,214 · no call needed 11,193** ·
  no tools declared 2,499 — about 30 % of rows teach the model *not* to call.
- Modes: `reasoning` rows (~38K) carry a trace in `reasoning_content` on every assistant step of the current tool
  loop; `direct` rows carry none.

## Schema (HF / OpenAI chat format)

| Column | Content |
|---|---|
| `id`, `source`, `source_dataset` (repo@revision), `license`, `language` | provenance, per row |
| `category`, `mode` | labels above |
| `tools` | JSON string: list of `{"type": "function", "function": {name, description, parameters}}` (JSON Schema) |
| `messages` | `system` / `user` / `assistant` / `tool`; assistant turns may have `tool_calls` (`function.arguments` is a JSON string) and `reasoning_content` |
| `translation_model` | set for translated rows |

Load `tools` with `json.loads` and pass both to a chat template, e.g. `tokenizer.apply_chat_template(messages, tools=tools)`.
The rows are rendered and audited with the OpenEuroLLM template v2.1 (Qwen3/Hermes tool format: `<tools>` in the
system turn, `<tool_call>` JSON from the assistant, `<tool_response>` blocks in a user turn).

## Sources

| Source | Rows | License |
|---|---:|---|
| [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) `SFT` (real MCP executions) | ~62.6K | Apache-2.0 |
| [birgermoell/oellm-eu-tooluse-v1](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1) | ~19.6K | Apache-2.0 |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) SFT | ~14.8K | CC-BY-4.0 |
| [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) (tool calling, search, interactive agent) | ~20.6K | CC-BY-4.0 |
| [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) (graphsyn; When2Call-, glaive-, ToolACE-derived queries) | ~23.5K | Apache-2.0 (query sources: CC-BY-4.0 / Apache-2.0) |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | ~7.3K | Apache-2.0 |
| Translations (Gemma-4-31B via Ollama, `gemma4:31b` sha256:6316f062…) | 10,418 | upstream row license; translated text is model output |

Not included: ToolMind's APIGen-MT (CC-BY-NC), xLAM (gated) and τ-train (would contaminate τ-bench) subsets; BUTTON
(unclear license); Salesforce APIGen-MT-5k (CC-BY-NC).

## Construction and filters

1. Per-source adapters to the schema above (Hermes text, pythonic calls, separate `tool_call`/`tool_response` messages,
   inline `<think>`); loose types (`"str, optional"`, `"dict"`, `List[float]`) normalized to JSON Schema.
2. **Validation**: every call names a declared tool; arguments are a JSON object with all required keys and correct
   types; every call gets exactly one tool result; a row ends with an assistant turn (an answer, or calls whose results
   are not part of the row). Trailing user turns ("thanks!") are trimmed.
3. Rejected: literal special-tag text in content (`<tool_call>`, `<think>`, …), answers that *describe* a call instead of
   making it, image placeholders, other assistants' identities named without the prompt mentioning them, rows over
   32,768 tokens (never truncated).
4. Mode: `reasoning` only if every assistant step after the last user turn has a trace; otherwise all traces removed.
5. Rendered with the OpenEuroLLM tokenizer + template v2.1 and token-audited: no supervised `<think>`, tool list or tool
   response; every call appears exactly once in the supervised tokens.
6. Deduplicated across sources (first user prompt + tool set, priority order above); decontaminated against BFCL and
   the When2Call test set (13-gram overlap).
7. Category labels are heuristic for sources without labels (When2Call, ToolACE).

**Translation**: user, system and final-answer text only — never tool names, schemas, arguments or tool outputs. Each
source row is translated into one language. Strings and numbers that appear in call arguments must survive verbatim;
rows also pass a language-ID and length-ratio check (95 % pass rate). Languages: cs, da, de, el, es, fi, fr, is, it, nl,
no, pl, pt, ro, sv, uk. Tool outputs remain in English. Native-speaker review pending.

## Limitations

Synthetic and model-generated trajectories (various teacher models); simulated tool outputs; English-dominant;
heuristic category labels; no human review. Each row stays under its upstream terms (see `license`); model-generated
content is additionally subject to the generating models' output terms.
