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

<!-- glance:start -->
## At a glance

> **This model** is step 6 of the pipeline below (tool-call RL). In every figure it is the **red star** or the
> **red box**; earlier checkpoints of the same lineage are blue, other models are grey diamonds. All scores are
> on a 0–1 scale, higher is better.

![Per-domain scores of this model and its earlier checkpoints](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48/resolve/main/assets/overview.png)

| Domain | Measured by | This model | Tool-use SFT (step 3) | Change |
|---|---|---|---|---|
| Chat capabilities in 36 European languages | EUROPA Eval | **0.670** | 0.616 | ▲ +0.054 |
| Safety and behaviour | EUROPA Alignment | **0.670** | 0.642 | ▲ +0.028 |
| Math, step by step (thinking mode) | quick reasoning eval | **0.442** | 0.435 | ≈ same |
| Math, direct answers | quick reasoning eval | **0.577** | 0.581 | ≈ same |
| Thinking traces that finish | share of traces closed within 6K tokens | **0.511** | 0.511 | ≈ same |
| Tool calling: right call or no call | held-out tool check | **0.881** | 0.870 | ▲ +0.011 |
| Tool calling: correct tool names | held-out tool check | **0.784** | 0.734 | ▲ +0.050 |
| Tool calling: valid arguments | held-out tool check | **0.969** | 0.981 | ▼ -0.012 |
| Code | HumanEval+ (direct mode) | *running* | 0.335 |  |
| Reading comprehension | Belebele | *running* | 0.820 |  |
| Knowledge | Global-MMLU | *running* | 0.584 |  |
| Regional knowledge | INCLUDE | *running* | 0.539 |  |
| Grammar | MultiBLiMP | *running* | 0.930 |  |

**In short.**
- **Strongest:** tool calling (the best call decisions and tool names in the lineage, valid JSON 99.5 %);
  reading comprehension and grammar in European languages (Belebele 0.82, MultiBLiMP 0.93 at step 3); safe and
  well-formed behaviour overall (EUROPA Alignment 0.67, the best in the lineage).
- **Improved by DPO, merge and RL:** EUROPA Eval 0.62 → 0.67, mostly from quantitative answers (0.04 → 0.31),
  agentic control and fewer malformed answers (122 → 100).
- **Weakest:** thinking mode often does not finish (about half the traces hit the 6K-token limit), so thinking
  mode is less accurate than direct mode; code is weak (HumanEval+ ≈ 0.34 at step 3); grounding & safety (0.70 →
  0.58), refusal calibration (0.44 → 0.35) and direct-mode EU-language math (0.45 → 0.31) regressed after DPO.
- **Compared with other models:** above Apertus-8B-Instruct on EUROPA Eval (0.57) but below the project's
  instruct SFT (0.76) and DPO (0.84) models, which come from a different SFT lineage.
<!-- glance:end -->

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

Every number below was produced with the scripts in the repositories linked above, with greedy decoding.
The checkpoints are the stages of this model's own lineage:

| Checkpoint | What it is |
|---|---|
| Base | the pretrained OpenEuroLLM 9B 256K base (no chat ability) |
| 1 Instruct SFT | instruction tuning (direct mode only) |
| 2 Reasoning SFT | adds the opt-in thinking mode |
| 3 Tool-use SFT | adds tool calling (v1.1, after the tag-row repair) |
| 4 DPO v2 | preference tuning on 250K pairs |
| 5 Merge | 0.5 × step 3 + 0.5 × step 4 (restores tool use) |
| **6 Tool-call RL (this model)** | 48 updates of multi-turn tool-call RL from the merge |

**What each evaluation measures:**

| Evaluation | What it measures | Size | Languages | Notes |
|---|---|---|---|---|
| EUROPA Eval | practical capabilities: digital agency, grounding & safety, language understanding, practical communication, quantitative reasoning | 432 items | 36 European | direct mode; judged by gemma4:31b; [BirgerMoell/europa-eval](https://github.com/BirgerMoell/europa-eval) |
| EUROPA Alignment | behaviour: agentic control, dangerous capabilities, fairness, wellbeing, privacy, refusal calibration | 576 items | 36 European | direct mode; judged by gemma4:31b; [BirgerMoell/europa-alignment](https://github.com/BirgerMoell/europa-alignment) |
| Quick reasoning eval | math word problems and competition math, final answer checked | 570 items (GSM8K 150, MGSM 200, MATH-500 100, EU-language math 120) | MGSM en/de/es/fr; EU math in 15 languages | both modes; thinking budget 6,144 tokens |
| Held-out tool check | call or no call, tool names, JSON and argument validity, finishing | 361 conversations | 17 | direct mode; never trained on |
| EvalPlus | code correctness on extended hidden tests (pass@1, plus) | HumanEval+ 164, MBPP+ 378 | en | code runs in a contained sandbox |
| oellm-eval v0.3.0 | knowledge and language competence, few-shot without chat template | Belebele 20, Global-MMLU 11, INCLUDE 11, MultiBLiMP 28 tasks | many European | project harness [OpenEuroLLM/oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) |

![Every evaluation on every checkpoint](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48/resolve/main/assets/scorecard.png)

'–' means the evaluation was not run at that stage, and *running* means it is in progress for this
model. The card will be updated when it finishes.

### EUROPA Eval and Alignment

![EUROPA scores along the pipeline](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48/resolve/main/assets/europa_progress.png)

| Model | EUROPA Eval | malformed answers | EUROPA Alignment | malformed answers |
|---|---|---|---|---|
| 1 Instruct SFT | 0.600 | 120 | 0.455 | 72 |
| 2 Reasoning SFT | 0.609 | 132 | 0.605 | 72 |
| 3 Tool-use SFT | 0.616 | 122 | 0.642 | 72 |
| 5 Merge | 0.655 | 111 | 0.660 | 57 |
| **6 Tool-call RL (this model)** | **0.670** | 100 | **0.670** | 43 |
| *Reference: OpenEuroLLM instruct SFT (Neonkraft)* | 0.762 | 38 | 0.773 | 41 |
| *Reference: OpenEuroLLM threeway DPO* | 0.840 | 9 | – | – |
| *Reference: Apertus-8B-Instruct-2509* | 0.573 | 116 | – | – |

![EUROPA domains](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48/resolve/main/assets/europa_domains.png)

**EUROPA Eval by domain**

| Domain | 1 Instruct SFT | 2 Reasoning SFT | 3 Tool-use SFT | 5 Merge | **6 Tool-call RL (this model)** |
|---|---|---|---|---|---|
| digital agency | 0.727 | 0.708 | 0.736 | 0.782 | **0.815** |
| grounding safety | 0.560 | 0.710 | 0.700 | 0.573 | **0.580** |
| language understanding | 0.761 | 0.731 | 0.747 | 0.782 | **0.803** |
| practical communication | 0.754 | 0.800 | 0.811 | 0.821 | **0.826** |
| quantitative reasoning | 0.153 | 0.030 | 0.041 | 0.312 | **0.310** |

**EUROPA Alignment by domain**

| Domain | 1 Instruct SFT | 2 Reasoning SFT | 3 Tool-use SFT | 5 Merge | **6 Tool-call RL (this model)** |
|---|---|---|---|---|---|
| agentic control | 0.425 | 0.500 | 0.485 | 0.617 | **0.678** |
| dangerous capabilities | 0.095 | 0.447 | 0.510 | 0.479 | **0.481** |
| high stakes fairness | 0.761 | 0.738 | 0.747 | 0.765 | **0.753** |
| human wellbeing | 0.490 | 0.751 | 0.826 | 0.866 | **0.859** |
| privacy security | 0.549 | 0.667 | 0.709 | 0.686 | **0.696** |
| refusal calibration | 0.344 | 0.402 | 0.438 | 0.347 | **0.346** |

### Reasoning (quick eval)

| Model | direct: all | thinking: all | thinking traces finished | GSM8K (d / t) | MGSM (d / t) | MATH-500 (d / t) | EU-language math (d / t) |
|---|---|---|---|---|---|---|---|
| 2 Reasoning SFT | 0.588 | 0.402 | 46% | 0.71 / 0.64 | 0.71 / 0.58 | 0.34 / 0.09 | 0.43 / 0.06 |
| 3 Tool-use SFT | 0.581 | 0.435 | 51% | 0.70 / 0.67 | 0.68 / 0.59 | 0.36 / 0.16 | 0.45 / 0.11 |
| 4 DPO v2 | 0.558 | 0.440 | 53% | 0.83 / 0.62 | 0.71 / 0.62 | 0.31 / 0.16 | 0.17 / 0.14 |
| 5 Merge | 0.568 | 0.426 | 53% | 0.80 / 0.61 | 0.70 / 0.58 | 0.28 / 0.16 | 0.29 / 0.16 |
| **6 Tool-call RL (this model)** | **0.577** | **0.442** | 51% | 0.77 / 0.67 | 0.74 / 0.60 | 0.28 / 0.14 | 0.31 / 0.13 |

d = direct mode, t = thinking mode. Thinking mode is weaker than direct mode on these checkpoints because about
half of the traces do not finish within the budget.

### Tool calling (held-out check, 361 rows)

| Model | call decision | tool names | valid JSON | valid arguments | finished |
|---|---|---|---|---|---|
| 3 Tool-use SFT v1 (before the tag-row repair) | 0.496 | 0.496 | 0.000 | 0.000 | 98.1% |
| 3 Tool-use SFT | 0.870 | 0.734 | 0.995 | 0.981 | 98.3% |
| 4 DPO v2 | 0.831 | 0.654 | 0.898 | 0.849 | 90.0% |
| 5 Merge | 0.878 | 0.767 | 0.985 | 0.951 | 98.1% |
| **6 Tool-call RL (this model)** | **0.881** | **0.784** | **0.995** | **0.969** | 98.3% |

### Code (EvalPlus, pass@1 on the extended tests)

| Model | HumanEval+ direct | MBPP+ direct | HumanEval+ thinking | MBPP+ thinking |
|---|---|---|---|---|
| 3 Tool-use SFT | 0.335 | 0.357 | 0.189 | 0.243 |
| **6 Tool-call RL (this model)** | *running* | *running* | – | – |

### Knowledge and language (oellm-eval v0.3.0)

| Model | Belebele | Global-MMLU | INCLUDE | MultiBLiMP |
|---|---|---|---|---|
| Base | 0.737 | 0.547 | 0.422 | 0.935 |
| 3 Tool-use SFT | 0.820 | 0.584 | 0.539 | 0.930 |
| **6 Tool-call RL (this model)** | *running* | *running* | *running* | *running* |
| *Reference: OpenEuroLLM instruct SFT (Neonkraft)* | 0.808 | 0.573 | 0.546 | 0.931 |
| *Reference: Apertus-8B-Instruct-2509* | 0.792 | 0.560 | 0.578 | 0.924 |
| *Reference: Qwen3-8B* | 0.865 | 0.663 | 0.587 | 0.866 |

Means over the tasks every compared model completed. These benchmarks run few-shot without a chat template,
so they measure knowledge and language, not chat behaviour.

**Measurement notes.**
- The EUROPA and quick-eval numbers for step 3 were measured on its first revision (`0dc97efa`). Revision v1.1,
  which this lineage uses, differs only in the 8 output rows of the tool/think tag tokens.
- The EUROPA judge is gemma4:31b, run locally with a pinned digest. Scores have 95 % confidence intervals of about
  ±0.03–0.04. The numbers behind every figure are in [`assets/evals.json`](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48/blob/main/assets/evals.json).

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
