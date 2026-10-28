---
license: apache-2.0
base_model: openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b
language: [en, de, fr, es, it, pt, nl, sv, da, fi, pl, cs, sk, hu, ro, bg, el, uk, hr, sl, et, lv, lt]
pipeline_tag: text-generation
library_name: transformers
tags: [openeurollm, multilingual, reasoning, tool-use, function-calling, dpo, rlvr, grpo, dapo, research-preview]
datasets:
  - birgermoell/openeurollm-reasoning-instruct-v1
  - birgermoell/openeurollm-tool-use-v1
  - openeurollm/Dolci-Instruct-DPO-translated
  - nvidia/HelpSteer3
  - argilla/ultrafeedback-binarized-preferences-cleaned
  - birgermoell/oellm-eu-exam-mcq-v1
  - birgermoell/oellm-math-rlvr
  - birgermoell/oellm-code-rlvr
  - openai/gsm8k
  - nvidia/Nemotron-RL-instruction_following
  - nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1
  - nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1
  - MadeAgents/xlam-irrelevance-7.5k
---

# OpenEuroLLM 9B 256K — post-training checkpoint after two RL windows (W2, step 160)

**Research preview.** The current best checkpoint of the OpenEuroLLM 9B post-training pipeline: instruct SFT →
reasoning SFT → tool-use SFT → DPO → weight merge → reinforcement learning window W1a → RL window W2. It replaces the
previous public release, [`oellm-9b-256k-stage5-a4b-tool-rl-step48`](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48),
as the recommended checkpoint. It is released so that the steps, data and evaluations can be inspected and reproduced;
it is not a finished assistant.

- **Base:** [`openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b)
  (Qwen3 architecture, 9B parameters, 256K context with RoPE θ = 64M, 263,168-token vocabulary, untied embeddings).
- **Two modes in one model:** a direct (default) mode and an opt-in thinking mode (`enable_thinking=True`). After RL,
  thinking mode is the more accurate one for math and finishes 86 % of its traces within 6K tokens.
- **Tool calling** in the Qwen3/Hermes format (`<tools>` in the system turn, `<tool_call>{…}</tool_call>` replies,
  `<tool_response>` results).
- **Code and full documentation:** [github.com/BirgerMoell/qwen35-posttrain](https://github.com/BirgerMoell/qwen35-posttrain)
  (SFT/DPO pipeline, playbook, RL run program) and [github.com/BirgerMoell/oellm-rlvr](https://github.com/BirgerMoell/oellm-rlvr)
  (RL control plane, verifiers, data builders). Plan and record:
  [project page](https://birgermoell.github.io/qwen35-posttrain/).

<!-- glance:start -->
## At a glance

> **This model** is step 7 of the pipeline below (the RL window W2). In every figure it is the **red star** or the
> **red box**; earlier checkpoints of the same lineage are blue, the previous public release (A4b) is an orange
> square, other models are grey diamonds. All scores are on a 0–1 scale, higher is better.

![Per-domain scores of this model and its earlier checkpoints](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160/resolve/main/assets/overview.png)

| Domain | Measured by | This model | Previous release (A4b) | Change vs A4b | Tool-use SFT (step 3) |
|---|---|---|---|---|---|
| Chat capabilities in 36 European languages | EUROPA Eval | **0.727** | 0.670 | ▲ +0.057 | 0.616 |
| Safety and behaviour | EUROPA Alignment | **0.717** | 0.670 | ▲ +0.047 | 0.642 |
| Math, step by step (thinking mode) | quick reasoning eval | **0.630** | 0.442 | ▲ +0.188 | 0.435 |
| Math, direct answers | quick reasoning eval | **0.575** | 0.577 | ≈ same | 0.581 |
| Thinking traces that finish | share of traces closed within 6K tokens | **0.860** | 0.511 | ▲ +0.349 | 0.511 |
| Tool calling: right call or no call | held-out tool check | **0.886** | 0.881 | ≈ same | 0.870 |
| Tool calling: correct tool names | held-out tool check | **0.789** | 0.784 | ≈ same | 0.734 |
| Tool calling: valid arguments | held-out tool check | **0.972** | 0.969 | ≈ same | 0.981 |
| Code | HumanEval+ (direct mode) | **0.390** | 0.354 | ▲ +0.036 | 0.335 |
| Reading comprehension | Belebele | *running* | 0.824 |  | 0.821 |
| Knowledge | Global-MMLU | *running* | 0.587 |  | 0.584 |
| Regional knowledge | INCLUDE | *running* | 0.542 |  | 0.539 |
| Grammar | MultiBLiMP | *running* | 0.932 |  | 0.930 |
| European facts (with a penalty for guessing) | EU factual QA accuracy | **0.280** | 0.265 | ▲ +0.015 | – |
| Business documents (invoices, tables, contracts) | EU document tasks | **0.376** | 0.360 | ▲ +0.016 | – |

**In short.**
- **Strongest:** reasoning, now mostly in thinking mode (math accuracy 0.63 in thinking vs 0.58 direct; 86 % of
  traces finish, up from about half before RL); tool calling (the best scores in the lineage: call decision 0.89,
  tool names 0.79, valid JSON 100 %, valid arguments 0.97); chat capabilities and safe behaviour across 36
  European languages (EUROPA Eval 0.73, Alignment 0.72: the best Alignment score in this lineage).
- **Improved over the previous release (A4b):** thinking-mode math 0.44 → 0.63, EU-language math in thinking
  mode 0.13 → 0.50, EUROPA Eval 0.67 → 0.73 (digital agency, language understanding, quantitative reasoning),
  EUROPA Alignment 0.67 → 0.72 (agentic control), code (HumanEval+ 0.35 → 0.39 direct, 0.17 → 0.30 thinking).
- **Knows more European facts than Qwen3-8B but rarely says "I don't know":** on our EU factual QA it answers
  0.28 correctly (Qwen3-8B 0.16 in the same mode) but abstains only 5 % of the time, so most wrong answers are
  confident guesses.
- **Weakest:** business documents with multi-step arithmetic (0.38 vs 0.74 for Qwen3-8B), partly because the
  model translates the required `ANSWER:` keyword; refusal calibration (0.35) and grounding & safety (0.60), which
  regressed after DPO and have not recovered; strict "answer only with a number" prompts in direct mode.
- **Compared with other models:** above Apertus-8B-Instruct on EUROPA Eval (0.57), below the project's instruct
  SFT (0.76) and DPO (0.84) models, which come from a different SFT lineage.
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
| 4 | DPO v2 | stage 3 (policy and reference) | 249,996 pairs: 176,292 [Dolci-Instruct-DPO-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-DPO-translated) @ `a57841a6` (14,691 per language × 12), 28,935 [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) @ `f6d14577`, 30,000 [UltraFeedback-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) @ `770076f0`, 14,769 format-compliance pairs | TRL DPO, sigmoid, β 0.1, LR 5e-7 cosine, 3 % warmup, global batch 128, max 2,048 tokens, 1 epoch; held-out preference accuracy 0.85 | ~368 GPU-h |
| 5 | Linear merge | 0.5 × stage 3 + 0.5 × DPO v2 | — | tensor-wise interpolation in fp32 (restores the tool use that DPO weakened) | minutes |
| 6 | RL window W1a | merge | 2,048 prompts (below) | DAPO (TMAX / open-instruct), 128 updates | ~39 GPU-h |
| 7 | **RL window W2 (this model)** | W1a, step 128 | 2,560 prompts (below) | DAPO, 160 updates, length penalty | ~30 GPU-h |

**Format-compliance pairs (stage 4).** Prompts that ask for a bare answer, in their native language: letter-only
multiple choice from `exams_qa` and `slovak_financial_exam` in [`oellm-eu-exam-mcq-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1)
@ `12a22a62` (CC-BY-SA-4.0 rows only; benchmark sources excluded) and final-answer-only math from
[`oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) @ `0ffc9d6d` in 23 languages.
Chosen = the verified bare answer; rejected = the stage-3 model's own verbose or wrong reply.

**Why the merge.** DPO v2 kept reasoning but hurt tool use (valid JSON 0.995 → 0.898, valid arguments 0.981 → 0.849),
mostly through runaway generation inside tool calls: DPO's pairs contain no tool calls and are all direct-mode. A 50/50
weight interpolation with stage 3 restored tool use at equal reasoning.

### RL setup (stages 6 and 7)

Both windows use the same loop: for each prompt the model samples 8 answers, a program scores each one, and the update
makes better-than-group-average answers more likely (DAPO: group-relative advantages, no KL term, clip-higher,
token-level loss, dynamic sampling that refills groups where all 8 answers scored the same). LR 1e-6, 16 prompts × 8
samples per update, 2 nodes (4 DeepSpeed ZeRO-3 learners, 4 vLLM engines), asynchronous generation 2 updates ahead
with in-flight weight updates, and an fp32 output layer in both the trainer and vLLM. Thinking-mode rows are scored
only on the text after exactly one closed `</think>`, so an answer inside an unfinished trace earns nothing, and a
rollout that hits the length limit gets −0.5.

**W1a (stage 6), 2,048 prompts, 6K-token responses, 128 updates, 4 h 26 min:**
- 560 EU-language math rows from [`oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) @ `0ffc9d6d`
  (12 languages, thinking mode). Reward: correct final answer, written in the prompt's language (the reasoning itself
  may be in any language).
- 540 English math rows (`oellm-math-rlvr` English and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) @ `740312ad` train, thinking mode).
- 300 "final answer only" math rows and 300 "answer only with the letter" exam questions (`exams_qa`, CC-BY-SA-4.0),
  both in direct mode: reward 1 only for a bare correct answer.
- 348 programming problems from [`oellm-code-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-code-rlvr) @ `e1cae771`:
  reward 1 only if every hidden test passes, run in a contained, network-free execution server.

**W2 (stage 7, this model), 2,560 prompts, 7K-token responses, 160 updates, 3 h 41 min:**
- 896 thinking-mode math rows (682 EU-language, 214 English), 806 of them chosen by profiling W1a: only prompts it
  solved 1–7 times out of 8, hardest first, so that most groups carry a learning signal.
- 192 "final answer only" math rows with a shaped reward (1 for a bare correct answer; partial credit, falling with
  length, for a correct answer after working) and 192 letter-only exam questions (direct mode).
- 384 instruction-following prompts with verifiable constraints (word counts, keywords, formats, case, JSON, …) from
  [Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) @ `bcdec57a`
  (ODC-BY): 256 in direct mode, 128 in thinking mode; reward = fraction of constraints met.
- 384 programming problems (`oellm-code-rlvr`, unused by W1a).
- 512 multi-turn tool-call items (the next assistant action inside a real tool conversation) from
  [Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1),
  [Nemotron-RL-Agentic-Function-Calling-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1)
  and [xlam-irrelevance-7.5k](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k): 0.2 for well-formed
  calls, +0.3 for the right tool names, +0.5 × the share of calls with the right arguments; 1 for a plain reply when no
  call is expected.
- A concave length penalty on top of the −0.5 for truncated rollouts.

What happened in training: in W1a the share of rollouts cut off at the token limit fell from 41 % to almost none and
the mean answer length from about 3,400 to 1,250 tokens, while the reward rose. In W2 truncation stayed below 2 %,
answers were 600–950 tokens, and the reward rose from about 0.38 to 0.45. About two thirds of the groups sampled in W2
had all 8 answers scored the same (mostly all wrong) and were refilled by dynamic sampling.

**Compute.** The stages above total about **1,250 NVIDIA A100 (64 GB) GPU-hours**, from the Slurm job records:

| Stage | GPU-h | Job |
|---|---|---|
| Instruct SFT | 394 | 12.3 h × 32 GPUs |
| Reasoning SFT | 351 | 11.0 h × 32 |
| Tool-use SFT + tag-row repair | 68 | 2.1 h × 32 |
| DPO v2 | 368 | 23.0 h × 16 |
| Merge | 0.2 | |
| RL window W1a | 39 | 4.4 h × 8, plus a 3.5 GPU-h canary |
| RL window W2 | 30 | 3.7 h × 8 |

Experiments on the way (a first DPO attempt on the full 1.44M pairs, SFT pilots, data translation, about 15 RL
ablations, the A4b tool-RL branch and a specialist-and-merge test) and all evaluations are not counted. Neither is the
base model's pretraining by OpenEuroLLM.

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
| 6 RL window W1a | 128 updates of RL from the merge: EU-language and English math, code with tests, strict formats |
| A4b tool RL (previous release) | previous public release: 48 updates of tool-call RL from the merge (a sibling branch, not an ancestor) |
| **7 RL window W2 (this model)** | 160 updates of RL from W1a: math, strict formats, instruction following, code, tool calls; length penalty |

**What each evaluation measures:**

| Evaluation | What it measures | Size | Languages | Notes |
|---|---|---|---|---|
| EUROPA Eval | practical capabilities: digital agency, grounding & safety, language understanding, practical communication, quantitative reasoning | 432 items | 36 European | direct mode; judged by gemma4:31b; [BirgerMoell/europa-eval](https://github.com/BirgerMoell/europa-eval) |
| EUROPA Alignment | behaviour: agentic control, dangerous capabilities, fairness, wellbeing, privacy, refusal calibration | 576 items | 36 European | direct mode; judged by gemma4:31b; [BirgerMoell/europa-alignment](https://github.com/BirgerMoell/europa-alignment) |
| Quick reasoning eval | math word problems and competition math, final answer checked | 570 items (GSM8K 150, MGSM 200, MATH-500 100, EU-language math 120) | MGSM en/de/es/fr; EU math in 15 languages | both modes; thinking budget 6,144 tokens |
| Held-out tool check | call or no call, tool names, JSON and argument validity, finishing | 361 conversations | 17 | direct mode; never trained on |
| EvalPlus | code correctness on extended hidden tests (pass@1, plus) | HumanEval+ 164, MBPP+ 378 | en | code runs in a contained sandbox |
| oellm-eval v0.3.0 | knowledge and language competence, few-shot without chat template | Belebele 20, Global-MMLU 11, INCLUDE 11, MultiBLiMP 28 tasks | many European | project harness [OpenEuroLLM/oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) |
| EU task suites | factual QA about European people, places, works and organisations (+1 correct, 0 "UNKNOWN", −1 wrong) and business-document tasks (invoices, staff tables, contracts, room bookings, letter → JSON) | 600 + 500 items | 10 EU | default mode, strict `ANSWER:` line; [birgermoell/oellm-eu-tasks](https://huggingface.co/datasets/birgermoell/oellm-eu-tasks) |

![Every evaluation on every checkpoint](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160/resolve/main/assets/scorecard.png)

'–' means the evaluation was not run at that stage.

### EUROPA Eval and Alignment

![EUROPA scores along the pipeline](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160/resolve/main/assets/europa_progress.png)

| Model | EUROPA Eval | malformed answers | EUROPA Alignment | malformed answers |
|---|---|---|---|---|
| 1 Instruct SFT | 0.600 | 120 | 0.455 | 72 |
| 2 Reasoning SFT | 0.609 | 132 | 0.605 | 72 |
| 3 Tool-use SFT | 0.616 | 122 | 0.642 | 72 |
| 5 Merge | 0.655 | 111 | 0.660 | 57 |
| 6 RL window W1a | 0.735 | 47 | 0.711 | 4 |
| A4b tool RL (previous release) | 0.670 | 100 | 0.670 | 43 |
| **7 RL window W2 (this model)** | **0.727** | 53 | **0.717** | 2 |
| *Reference: OpenEuroLLM instruct SFT (Neonkraft)* | 0.762 | 38 | 0.773 | 41 |
| *Reference: OpenEuroLLM threeway DPO* | 0.840 | 9 | – | – |
| *Reference: Apertus-8B-Instruct-2509* | 0.573 | 116 | – | – |

![EUROPA domains](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160/resolve/main/assets/europa_domains.png)

**EUROPA Eval by domain**

| Domain | 1 Instruct SFT | 2 Reasoning SFT | 3 Tool-use SFT | 5 Merge | 6 RL window W1a | A4b tool RL (previous release) | **7 RL window W2 (this model)** |
|---|---|---|---|---|---|---|---|
| digital agency | 0.727 | 0.708 | 0.736 | 0.782 | 0.898 | 0.815 | **0.898** |
| grounding safety | 0.560 | 0.710 | 0.700 | 0.573 | 0.620 | 0.580 | **0.600** |
| language understanding | 0.761 | 0.731 | 0.747 | 0.782 | 0.876 | 0.803 | **0.896** |
| practical communication | 0.754 | 0.800 | 0.811 | 0.821 | 0.826 | 0.826 | **0.832** |
| quantitative reasoning | 0.153 | 0.030 | 0.041 | 0.312 | 0.403 | 0.310 | **0.357** |

**EUROPA Alignment by domain**

| Domain | 1 Instruct SFT | 2 Reasoning SFT | 3 Tool-use SFT | 5 Merge | 6 RL window W1a | A4b tool RL (previous release) | **7 RL window W2 (this model)** |
|---|---|---|---|---|---|---|---|
| agentic control | 0.425 | 0.500 | 0.485 | 0.617 | 0.834 | 0.678 | **0.865** |
| dangerous capabilities | 0.095 | 0.447 | 0.510 | 0.479 | 0.466 | 0.481 | **0.493** |
| high stakes fairness | 0.761 | 0.738 | 0.747 | 0.765 | 0.806 | 0.753 | **0.763** |
| human wellbeing | 0.490 | 0.751 | 0.826 | 0.866 | 0.890 | 0.859 | **0.886** |
| privacy security | 0.549 | 0.667 | 0.709 | 0.686 | 0.691 | 0.696 | **0.710** |
| refusal calibration | 0.344 | 0.402 | 0.438 | 0.347 | 0.351 | 0.346 | **0.348** |

### Reasoning (quick eval)

| Model | direct: all | thinking: all | thinking traces finished | GSM8K (d / t) | MGSM (d / t) | MATH-500 (d / t) | EU-language math (d / t) |
|---|---|---|---|---|---|---|---|
| 2 Reasoning SFT | 0.588 | 0.402 | 46% | 0.71 / 0.64 | 0.71 / 0.58 | 0.34 / 0.09 | 0.43 / 0.06 |
| 3 Tool-use SFT | 0.581 | 0.435 | 51% | 0.70 / 0.67 | 0.68 / 0.59 | 0.36 / 0.16 | 0.45 / 0.11 |
| 4 DPO v2 | 0.558 | 0.440 | 53% | 0.83 / 0.62 | 0.71 / 0.62 | 0.31 / 0.16 | 0.17 / 0.14 |
| 5 Merge | 0.568 | 0.426 | 53% | 0.80 / 0.61 | 0.70 / 0.58 | 0.28 / 0.16 | 0.29 / 0.16 |
| 6 RL window W1a | 0.574 | 0.619 | 85% | 0.77 / 0.79 | 0.74 / 0.78 | 0.28 / 0.27 | 0.28 / 0.43 |
| A4b tool RL (previous release) | 0.577 | 0.442 | 51% | 0.77 / 0.67 | 0.74 / 0.60 | 0.28 / 0.14 | 0.31 / 0.13 |
| **7 RL window W2 (this model)** | **0.575** | **0.630** | 86% | 0.76 / 0.79 | 0.74 / 0.74 | 0.29 / 0.31 | 0.30 / 0.50 |

d = direct mode, t = thinking mode. Before RL about half of the thinking traces did not finish within the 6K-token budget, so thinking mode was weaker than direct mode; the RL windows (W1a, W2) taught the model to close its traces, and thinking mode is now the stronger one.

### Tool calling (held-out check, 361 rows)

| Model | call decision | tool names | valid JSON | valid arguments | finished |
|---|---|---|---|---|---|
| 3 Tool-use SFT v1 (before the tag-row repair) | 0.496 | 0.496 | 0.000 | 0.000 | 98.1% |
| 3 Tool-use SFT | 0.870 | 0.734 | 0.995 | 0.981 | 98.3% |
| 4 DPO v2 | 0.831 | 0.654 | 0.898 | 0.849 | 90.0% |
| 5 Merge | 0.878 | 0.767 | 0.985 | 0.951 | 98.1% |
| 6 RL window W1a | 0.878 | 0.778 | 0.991 | 0.958 | 99.2% |
| A4b tool RL (previous release) | 0.881 | 0.784 | 0.995 | 0.969 | 98.3% |
| **7 RL window W2 (this model)** | **0.886** | **0.789** | **1.000** | **0.972** | 98.1% |

### Code (EvalPlus, pass@1 on the extended tests)

| Model | HumanEval+ direct | MBPP+ direct | HumanEval+ thinking | MBPP+ thinking |
|---|---|---|---|---|
| 3 Tool-use SFT | 0.335 | 0.357 | 0.189 | 0.243 |
| 6 RL window W1a | 0.372 | 0.362 | 0.348 | 0.328 |
| A4b tool RL (previous release) | 0.354 | 0.362 | 0.165 | 0.254 |
| **7 RL window W2 (this model)** | **0.390** | **0.365** | **0.299** | **0.333** |

Thinking-mode code is read only after a closed trace (budget 8,192 tokens). This model closes 94 % of HumanEval+ and 93 % of MBPP+ traces.

### Knowledge and language (oellm-eval v0.3.0)

| Model | Belebele | Global-MMLU | INCLUDE | MultiBLiMP |
|---|---|---|---|---|
| Base | 0.737 | 0.547 | 0.422 | 0.935 |
| 3 Tool-use SFT | 0.821 | 0.584 | 0.539 | 0.930 |
| 6 RL window W1a | 0.826 | 0.589 | 0.539 | 0.932 |
| A4b tool RL (previous release) | 0.824 | 0.587 | 0.542 | 0.932 |
| **7 RL window W2 (this model)** | *running* | *running* | *running* | *running* |
| *Reference: OpenEuroLLM instruct SFT (Neonkraft)* | 0.808 | 0.573 | 0.546 | 0.931 |
| *Reference: Apertus-8B-Instruct-2509* | 0.792 | 0.560 | 0.578 | 0.924 |
| *Reference: Qwen3-8B* | 0.865 | 0.663 | 0.587 | 0.866 |

Means over the tasks every compared model completed. These benchmarks run few-shot without a chat template,
so they measure knowledge and language, not chat behaviour.

### EU task suites (default mode)

| Model | QA accuracy | QA abstain | QA index (correct − wrong, ×100) | Document score |
|---|---|---|---|---|
| 6 RL window W1a | 0.273 | 3 % | −42.0 | 0.378 |
| A4b tool RL (previous release) | 0.265 | 3 % | −43.7 | 0.360 |
| **7 RL window W2 (this model)** | **0.280** | 5 % | **−39.3** | **0.376** |
| *Reference: Qwen3-8B (default / thinking)* | 0.157 / 0.205 | 15 % / 24 % | −53.3 / −35.5 | 0.742 / 0.947 |

The QA index rewards abstaining over guessing (0 = always "UNKNOWN"). Document tasks need the exact answer on a final
`ANSWER:` line; this model sometimes translates that keyword ("Respuesta:"), which costs about 4–9 points.

**Measurement notes.**
- The EUROPA and quick-eval numbers for step 3 were measured on its first revision (`0dc97efa`). Revision v1.1,
  which this lineage uses, differs only in the 8 output rows of the tool/think tag tokens.
- The EUROPA judge is gemma4:31b, run locally with a pinned digest. Scores have 95 % confidence intervals of about
  ±0.03–0.04. The numbers behind every figure are in [`assets/evals.json`](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160/blob/main/assets/evals.json).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "birgermoell/oellm-9b-256k-stage5-w2-rl-step160"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "Ett tåg går 14:20 och kommer fram 16:05. Hur lång är resan?"}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)  # thinking mode
# Direct mode: tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# Tools: tok.apply_chat_template(messages, tools=[...], add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

- **Chat template:** `oellm_qwen3_two_stage_v2_1`, bundled as `chat_template.jinja`. It has a hard `enable_thinking`
  switch: direct mode never produces think tags, and in thinking mode the template opens `<think>` and the model closes
  it. The end-of-turn token is `<|im_end|>` (id 4). For math and multi-step problems, use thinking mode with a budget of
  at least 6K new tokens.
- **Tag tokens:** `<tool_call>` 11, `</tool_call>` 12, `<tools>` 13, `</tools>` 14, `<tool_response>` 15,
  `</tool_response>` 16, `</think>` 17, `<think>` 18.

## Limitations

- **Overconfident on facts.** On our EU factual QA the model abstains in only 5 % of cases, and 93 % of its non-correct
  answers are confident guesses. Do not rely on it for names, dates or places without checking.
- **Weak on business documents** with multi-step arithmetic (score 0.38; Qwen3-8B 0.74 in direct mode), and it sometimes
  translates a required output keyword.
- **Safety:** EUROPA Alignment is 0.72, but grounding & safety (0.60) and refusal calibration (0.35) are below the
  tool-use SFT and did not recover in RL. Not evaluated for deployment; do not use it for high-stakes decisions.
- **Strict output formats in direct mode** ("answer only with the number") are still often answered with working.
- **Language of reasoning.** Reasoning traces are mostly English, also for other prompt languages (RL rewarded only
  the final answer's language); final answers follow the prompt language.
- **Thinking mode still fails to finish on hard problems** (38 % of MATH-500 traces hit the 6K-token limit).
- **Code thinking mode** dropped from W1a (HumanEval+ 0.35 → 0.30) while direct mode improved (0.37 → 0.39).
- **Intermediate checkpoints** (stages 1–6 and DPO v2) are private; the previous public release (A4b) is a sibling branch.

## Licenses and attribution

The model is released under Apache-2.0, the license of the base model. The training data includes sources under
Apache-2.0, MIT, CC-BY-4.0, CC-BY-SA-4.0 (letter-only exam questions) and ODC-BY (instruction-following prompts).
Per-row sources and licenses are listed in the dataset cards of the OpenEuroLLM mixtures. We thank NVIDIA (Nemotron RL
data, HelpSteer3), Salesforce/MadeAgents (xlam irrelevance), Argilla/OpenBMB (UltraFeedback), AllenAI (Dolci, TMAX /
open-instruct), OpenAI (GSM8K) and the OpenEuroLLM consortium. The compute was provided by EuroHPC JU / CINECA (Leonardo).
