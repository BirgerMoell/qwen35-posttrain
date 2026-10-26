---
license: apache-2.0
base_model: openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b
language: [en, de, fr, es, it, pt, nl, sv, da, fi, pl, cs, sk, hu, ro, bg, el, uk, hr, sl, et, lv, lt]
pipeline_tag: text-generation
library_name: transformers
tags: [openeurollm, multilingual, tool-use, function-calling, reasoning, dpo, rlvr, grpo, research-preview]
datasets:
  - birgermoell/openeurollm-reasoning-instruct-v1
  - birgermoell/openeurollm-tool-use-v1
  - openeurollm/Dolci-Instruct-DPO-translated
  - nvidia/HelpSteer3
  - argilla/ultrafeedback-binarized-preferences-cleaned
  - birgermoell/oellm-eu-exam-mcq-v1
  - birgermoell/oellm-math-rlvr
  - nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1
  - nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1
  - MadeAgents/xlam-irrelevance-7.5k
---

# OpenEuroLLM 9B 256K — post-training checkpoint after tool-call RL (A4b, step 48)

**Research preview.** An intermediate checkpoint of the OpenEuroLLM 9B post-training pipeline: instruct SFT →
reasoning SFT → tool-use SFT → DPO → weight merge → reinforcement learning on tool calls. It is released so that the
steps, data and evaluations can be inspected and reproduced; it is not a finished assistant. Later RL windows (math,
code, strict formats) are still in progress.

- **Base:** [`openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b)
  (Qwen3 architecture, 9B parameters, 256K context with RoPE θ = 64M, 263,168-token vocabulary, untied embeddings).
- **Two modes in one model:** a direct (default) mode and an opt-in thinking mode (`enable_thinking=True`).
- **Tool calling** in the Qwen3/Hermes format (`<tools>` in the system turn, `<tool_call>{…}</tool_call>` replies,
  `<tool_response>` results).
- **Code and full documentation:** [github.com/BirgerMoell/qwen35-posttrain](https://github.com/BirgerMoell/qwen35-posttrain)
  (SFT/DPO pipeline, playbook, run program) and [github.com/BirgerMoell/oellm-rlvr](https://github.com/BirgerMoell/oellm-rlvr)
  (RL control plane, verifiers, data builders). Plan and record:
  [project page](https://birgermoell.github.io/qwen35-posttrain/).

## How it was made

All stages ran on the CINECA Leonardo booster (NVIDIA A100 64 GB) in an EuroHPC AI Factory allocation. Every data
mix was pre-tokenized offline with one pinned tokenizer (`tokenizer.json` sha256 `5123f74b…`) and one pinned chat
template, and every launcher refuses to train if either differs. Rows that are too long are dropped, never truncated.

| # | Stage | Starting point | Data | Recipe | Compute |
|---|---|---|---|---|---|
| 1 | Instruct SFT | base | 3.14M rows / 2.37B tokens (OpenEuroLLM-Excellent-SFT core, Dolci translated into 12 languages, Aya human non-English, model-identity rows) | 1 epoch, LR 8e-6 cosine, 16K padding-free packing, FSDP, 8 nodes | ~394 GPU-h |
| 2 | Reasoning SFT | stage 1 | [`openeurollm-reasoning-instruct-v1`](https://huggingface.co/datasets/birgermoell/openeurollm-reasoning-instruct-v1) @ `9273ba4`: 373K reasoning rows (1.74B tokens) + 792K direct replay rows (0.6B) | LR 5e-6, 32K packing; `<think>` never supervised (prompt-side only) | ~351 GPU-h |
| 3 | Tool-use SFT (v1.1) | stage 2 | [`openeurollm-tool-use-v1`](https://huggingface.co/datasets/birgermoell/openeurollm-tool-use-v1) @ `e30b5449` (159K conversations, 16 translated languages, ~30 % "do not call" rows) + stage-2 replay (traces ≤ 8K tokens) + identity/safety | LR 3e-6, 397 steps; then a repair of the `lm_head` rows of the tool/think tag tokens (ids 11–18), which started nearly identical so the model closed calls with the wrong tag | ~68 GPU-h |
| 4 | DPO v2 | stage 3 (policy and reference) | 249,996 pairs: 176,292 [Dolci-Instruct-DPO-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-DPO-translated) @ `a57841a6` (14,691 per language × 12), 28,935 [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) @ `f6d14577`, 30,000 [UltraFeedback-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) @ `770076f0`, 14,769 format-compliance pairs (below) | TRL DPO, sigmoid, β 0.1, LR 5e-7 cosine, 3 % warmup, global batch 128, max 2,048 tokens, 1 epoch; held-out preference accuracy 0.85 | ~368 GPU-h |
| 5 | Linear merge | 0.5 × stage 3 + 0.5 × DPO v2 | — | tensor-wise interpolation in fp32 | minutes |
| 6 | Tool-call RL (A4b) | merge | 2,000 multi-turn next-action items (below) | DAPO (TMAX/open-instruct), 48 updates | ~15 GPU-h |

**Format-compliance pairs (stage 4).** Prompts that ask for a bare answer, in their native language (no machine
translation): letter-only multiple choice from `exams_qa` and `slovak_financial_exam` in
[`oellm-eu-exam-mcq-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1) @ `12a22a62` (CC-BY-SA-4.0
rows only; benchmark sources such as Belebele, Global-MMLU, MMMLU and XCOPA excluded) and final-answer-only math from
[`oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) @ `0ffc9d6d` in 23 languages.
Chosen = the verified bare answer; rejected = the stage-3 model's own verbose or wrong reply (greedy, on-policy).

**Why the merge.** DPO v2 kept reasoning but hurt tool use: on the held-out tool check, valid JSON fell from 0.995 to
0.898 and valid arguments from 0.981 to 0.849. Most failures were runaway generation inside tool calls: DPO's pairs
contain no tool calls and are all direct-mode, and DPO lengthened replies about 2.6×. A 50/50 weight interpolation with
stage 3 restored tool use at equal reasoning (α = 0.75 did not).

**Tool-call RL (A4b).** Each item is the *next assistant action* inside a real multi-step tool conversation, with the
earlier calls and results in the prompt. It is either a call (scored on name and arguments) or a plain reply (no call
expected):
- 1,200 items from [Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1)
  @ `bfc7a4dd` (policy-following service agents, ~17 tools). Kept only where the dataset's Qwen3-235B pass rate is
  between 0.1 and 0.8, so that groups are informative.
- 500 items from [Nemotron-RL-Agentic-Function-Calling-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1) @ `0ac34ec8`.
- 300 "no tool fits" items from [xlam-irrelevance-7.5k](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k) @ `34323bf0`.

Conversion: Responses-API items become chat messages, earlier reasoning items are dropped, and tools are rendered with
this model's template. Prompts are at most 4,096 tokens (median 3.5K). Overall 54 % of items expect a call and 46 % a
plain reply.

- **Reward** (thinking mode; scored only on the text after exactly one closed `</think>`):
  - call expected: 0.2 for well-formed calls, +0.3 if the multiset of tool names matches, +0.5 × the fraction of gold
    calls matched with equal arguments;
  - no call expected: 1 only for a non-empty reply without a call.
- **Recipe:** DAPO with β 0, LR 1e-6, 16 prompts × 8 samples per update, 4,096-token responses, a −0.5 reward for
  rollouts that hit the length limit, and 48 updates.
- **Training reward:** 0.35 → 0.42.

**Compute.** The stages above total about **1,200 NVIDIA A100 (64 GB) GPU-hours**, measured from the Slurm job
records:

| Stage | GPU-h | Job |
|---|---|---|
| Instruct SFT | 394 | 12.3 h × 32 GPUs |
| Reasoning SFT | 351 | 11.0 h × 32 |
| Tool-use SFT + tag-row repair | 68 | 2.1 h × 32 |
| DPO v2 | 368 | 23.0 h × 16 |
| Merge | 0.2 | |
| Tool-call RL | 15 | 1.85 h × 8, plus a 3 GPU-h canary |

Experiments on the way to this checkpoint cost about another 240 GPU-h: a first DPO attempt on the full 1.44M pairs
and its restarts (~140), SFT smoke and pilot runs (~30), and translation of the tool SFT data (~70). Evaluations and
RL ablations that did not feed this checkpoint are not counted. The base model's pretraining by OpenEuroLLM is not
included either.

**Software:** PyTorch 2.10, transformers 5.4, TRL 0.29 (SFT/DPO), TMAX (open-instruct fork, commit `3f80d370`) with
vLLM 0.19.1 and DeepSpeed ZeRO-3 for RL.

## Evaluation

All numbers are from our own harnesses (scripts in the repositories above) and use greedy decoding.

**Held-out tool check** (361 rows of the stage-3 tool dataset, never trained on; direct mode):

| Model | Call decision | Tool names | Valid JSON | Valid arguments | Finished |
|---|---|---|---|---|---|
| Stage 3 v1.1 (tool SFT) | 0.870 | 0.734 | 0.995 | 0.981 | 98.3 % |
| DPO v2 | 0.831 | 0.654 | 0.898 | 0.849 | 90.0 % |
| Merge (0.5 / 0.5) | 0.878 | 0.767 | 0.985 | 0.951 | 98.1 % |
| **This checkpoint** | **0.881** | **0.784** | **0.995** | 0.969 | 98.3 % |

**Quick reasoning eval** (GSM8K 150, MGSM en/de/es/fr 200, MATH-500 100, EU-language math 120; thinking mode has a
6,144-token budget):

| Model | Direct acc. | Thinking acc. | Thinking unfinished | MATH-500 (thinking) | EU math (thinking) |
|---|---|---|---|---|---|
| Stage 3 v1.1 | 0.581 | 0.435 | 48.9 % | 0.160 | 0.108 |
| Merge (0.5 / 0.5) | 0.568 | 0.426 | 47.4 % | 0.160 | 0.158 |
| **This checkpoint** | 0.577 | 0.442 | 48.9 % | 0.140 | 0.133 |

**Knowledge retention** ([oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) v0.3.0, few-shot without chat
template; measured on stage 3, which this checkpoint descends from; DPO and RL did not target these skills):

| Model | Belebele | Global-MMLU | INCLUDE | MultiBLiMP |
|---|---|---|---|---|
| Base | 0.737 | 0.547 | 0.422 | 0.935 |
| Stage 3 v1.1 | 0.820 | 0.584 | 0.539 | 0.930 |

These are means over the task subsets in the project documentation (Belebele 20, Global-MMLU 11, INCLUDE 11 and
MultiBLiMP 28 tasks).

**EUROPA Eval** (432 items, 12 capabilities in 36 European languages) and **EUROPA Alignment** (576 items, 16
scenarios). Settings: greedy, direct mode, seed 42; judge `gemma4:31b`; 95 % confidence intervals in brackets.

| Model | EUROPA Eval | malformed | EUROPA Alignment | malformed |
|---|---|---|---|---|
| Stage 3 v1.1 | 0.616 [0.577–0.656] | 122 | 0.642 [0.614–0.670] | 72 |
| Merge (0.5 / 0.5) | 0.655 [0.617–0.693] | 111 | 0.660 [0.629–0.690] | 57 |
| **This checkpoint** | **0.670** [0.632–0.708] | **100** | **0.670** [0.639–0.700] | **43** |

EUROPA Eval by domain, Stage 3 → this checkpoint:

| Domain | Stage 3 | This checkpoint |
|---|---|---|
| Quantitative reasoning | 0.041 | **0.310** |
| Digital agency | 0.736 | 0.815 |
| Language understanding | 0.747 | 0.803 |
| Practical communication | 0.811 | 0.826 |
| Grounding and safety | 0.700 | **0.580** (regression) |

EUROPA Alignment by domain, Stage 3 → this checkpoint:

| Domain | Stage 3 | This checkpoint |
|---|---|---|
| Agentic control | 0.485 | 0.678 |
| Human wellbeing | 0.826 | 0.859 |
| High-stakes fairness | 0.747 | 0.753 |
| Privacy and security | 0.709 | 0.696 |
| Dangerous capabilities | 0.510 | 0.481 |
| Refusal calibration | 0.438 | **0.346** (regression) |

DPO and the merge improved format compliance and quantitative answers. Grounding and safety, and refusal
calibration, got worse. A reference DPO model of the project (a different SFT lineage) scores 0.840 on EUROPA Eval,
so there is a large gap left, mainly in format compliance, grounding and language understanding.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "Vad är huvudstaden i Finland?"}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)   # direct mode
# Thinking mode: tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
# Tools: tok.apply_chat_template(messages, tools=[...], add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

- **Chat template:** `oellm_qwen3_two_stage_v2_1`, bundled as `chat_template.jinja`. It has a hard `enable_thinking`
  switch: direct mode never produces think tags, and in thinking mode the template opens `<think>` and the model closes
  it. The end-of-turn token is `<|im_end|>` (id 4).
- **Tag tokens:** `<tool_call>` 11, `</tool_call>` 12, `<tools>` 13, `</tools>` 14, `<tool_response>` 15,
  `</tool_response>` 16, `</think>` 17, `<think>` 18.

## Limitations

- **Thinking mode often does not terminate.** About half of the reasoning traces do not close within 6K tokens on our
  quick eval, and thinking mode is less accurate than direct mode. The RL windows that target this have not been
  applied to this checkpoint.
- **Strict output formats are weak in direct mode.** "Answer only with the number" still gets a worked solution.
  Letter-only multiple choice improved with the format pairs and merge, but bare numeric answers did not.
- **Language of reasoning.** Reasoning traces are mostly English, even for other prompt languages, because most
  reasoning SFT data is English. The final answers follow the prompt language.
- **Tool RL data is English-only,** and the tool SFT data is mostly English (with a translated slice in 16 languages).
- **Code is weak:** HumanEval+ ≈ 0.33 and MBPP+ ≈ 0.35 in direct mode at stage 3.
- **Safety:** EUROPA Alignment is 0.670, but grounding and safety (0.58) and refusal calibration (0.35) regressed
  against stage 3. Not evaluated for deployment; do not use it for high-stakes decisions.
- **Intermediate checkpoints** (stages 1–3, DPO v2 and the merge) are private.

## Licenses and attribution

The model is released under Apache-2.0, the license of the base model. The training data includes sources under
Apache-2.0, MIT, CC-BY-4.0 and CC-BY-SA-4.0 (letter-only exam questions in the format pairs). Per-row sources and
licenses are listed in the dataset cards of the OpenEuroLLM mixtures. We thank NVIDIA (Nemotron RL tool data,
HelpSteer3), Salesforce/MadeAgents (xlam irrelevance), Argilla/OpenBMB (UltraFeedback), AllenAI (Dolci) and the
OpenEuroLLM consortium. The compute was provided by EuroHPC JU / CINECA (Leonardo).
