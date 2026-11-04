---
license: apache-2.0
base_model: openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b
language: [en, de, fr, es, it, pt, nl, sv, da, fi, pl, cs, sk, hu, ro, bg, el, uk, hr, sl, et, lv, lt]
pipeline_tag: text-generation
library_name: transformers
tags: [openeurollm, multilingual, reasoning, tool-use, function-calling, search, hallucination, dpo, rlvr, grpo, dapo, research-preview]
datasets:
  - birgermoell/oellm-rl-pools
  - birgermoell/openeurollm-reasoning-instruct-v1
  - birgermoell/openeurollm-tool-use-v1
  - openeurollm/Dolci-Instruct-DPO-translated
  - wikimedia/wikipedia
  - nvidia/Aegis-AI-Content-Safety-Dataset-2.0
  - bench-llm/or-bench
  - rajpurkar/squad_v2
---

# OpenEuroLLM 9B 256K — honesty, search and tools specialist (RL window E8, step 160)

**Research preview, specialist checkpoint.** This checkpoint was trained to **answer what it knows, look up what it
doesn't, and say so when something does not exist**. It has the best factual honesty, search, tool-calling and
reasoning scores in the OpenEuroLLM 9B post-training lineage. It is also measurably worse than the previous release on
general chat capability and on parts of safe behaviour (below). For a general assistant, the previous release
[`oellm-9b-256k-stage5-w2-rl-step160`](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160) remains
the recommended checkpoint. This one is released as the expert that a later multi-teacher distillation step combines
with the others, and so that its training and evaluations can be inspected and reproduced.

- **Base:** [`openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b)
  (Qwen3 architecture, 9B parameters, 256K context, 263,168-token vocabulary).
- **Two modes:** direct (default) and opt-in thinking (`enable_thinking=True`), selected by the chat template.
- **Tool calling** in the Qwen3/Hermes format. It was trained with a real (offline) Wikipedia search tool in the loop
  (definition below).
- **Code and documentation:** [qwen35-posttrain](https://github.com/BirgerMoell/qwen35-posttrain) (pipeline, playbook,
  RL run program, hallucination plan) and [oellm-rlvr](https://github.com/BirgerMoell/oellm-rlvr) (RL control
  plane, verifiers, the search environment). Project record: [project page](https://birgermoell.github.io/qwen35-posttrain/).

## At a glance

All scores are 0–1, higher is better unless marked. "W2" is the previous public release. EUROPA scores are micro
averages, as on the W2 card.

| Domain | Measured by | **This model (E8)** | W2 (previous release) | Change |
|---|---|---|---|---|
| Knows European facts and abstains when unsure: well-known facts, default mode (accuracy / abstain) | EU factual QA, head band | **0.54 / 0.10** | 0.53 / 0.03 | ≈ same accuracy |
| Hallucination rate (wrong / not correct), default mode, lower is better | EU factual QA | **0.61** | 0.94 | ▼ better |
| Invents facts about made-up entities, default / thinking, lower is better | hallucination eval | **0.24 / 0.02** | 0.79 / 0.73 | ▼ much better |
| With Wikipedia search, default mode: accuracy / wrong | search QA | **0.53 / 0.09** | 0.51 / 0.33 | ▲ far fewer wrong answers |
| With search: invents about made-up entities, default / thinking | search QA | **0.02 / 0.07** | 0.25 / 0.31 | ▼ much better |
| Math, step by step (thinking) | quick reasoning eval | **0.649** | 0.630 | ▲ +0.019 |
| Math, direct answers | quick reasoning eval | 0.586 | 0.575 | ≈ same |
| Tool calling: right call or no call | held-out tool check | **0.906** | 0.886 | ▲ +0.020 |
| Tool calling: correct tool names / valid arguments | held-out tool check | 0.809 / 0.954 | 0.789 / 0.972 | ≈ |
| Code, direct / thinking | HumanEval+ | **0.415** / 0.341 | 0.390 / 0.299 | ▲ |
| Business documents (invoices, tables, contracts) | EU document tasks | **0.500** | 0.376 | ▲ +0.124 |
| Chat capabilities in 36 European languages | EUROPA Eval | 0.695 | **0.727** | ▼ −0.032 |
| Safety and behaviour | EUROPA Alignment | 0.688 | **0.717** | ▼ −0.029 |

**In short.**
- **Strongest:**
  - factual honesty: wrong answers fall sharply, and invented answers about entities that do not exist almost
    disappear in thinking mode;
  - search: it looks things up on almost every factual question and answers wrongly only 9 % of the time;
  - tool calling, code, reasoning and business documents: the best scores in the lineage.
- **Weaker than W2:**
  - **EUROPA agentic control** 0.865 → 0.594: acting in agentic scenarios, e.g. respecting confirmation and
    authority boundaries. A likely cause is the RL window's heavy tool-use and search training. **Do not use
    this checkpoint for autonomous actions without confirmation steps.**
  - **EUROPA language understanding** 0.896 → 0.768 and digital agency 0.898 → 0.810.
  - EUROPA quantitative reasoning 0.357 → 0.278.
- **Still weak:**
  - Thinking mode abstains on 20 % of well-known facts (the target is ≤ 10 %).
  - Refusal calibration (over-refusal of benign requests) is 0.36.
  - Exact decimal business arithmetic is error-prone (see the European office benchmarks below).

## How it was made

The lineage is:

```
instruct SFT → reasoning SFT → tool-use SFT → DPO → 0.5/0.5 weight merge → RL W1a → RL W2 → RL W3 → Stage 4b DPO on W3
                                                                                       W2 ─┐
                                                                                           ├─ 0.5/0.5 merge → RL E8 (this model)
                                                                              Stage 4b ────┘
```

Stages 1–7 (SFT, DPO, merge, RL W1a, RL W2) are documented on the
[W2 card](https://huggingface.co/birgermoell/oellm-9b-256k-stage5-w2-rl-step160) and in the playbook. What followed:

1. **RL window W3 (from W2).** Added calibrated European factual QA (reward +1 correct, 0 `ANSWER: UNKNOWN`,
   −1 wrong) and business-document tasks. It improved documents and code. But on a pool of mostly hard questions
   abstaining was the best policy, and W3's default mode **collapsed into answering UNKNOWN to almost everything**
   (accuracy 0.003).
2. **Stage 4b DPO on W3.**
   - Data: 2,110 abstention-repair pairs (correct answers preferred over UNKNOWN on questions W2 answered
     consistently), 689 identity pairs and 46 safety pairs, all sampled on W3, plus 4,000 general Stage-4 pairs.
   - Training: LR 5e-7, one epoch.
   - Result: safety improved (EUROPA Alignment macro 72.0), but the default-mode collapse remained.
3. **Weight merge 0.5 × W2 + 0.5 × Stage 4b.** It half-restored the default mode and kept part of the safety gain.
4. **RL window E8 (this model), from the merge.**
   - 2,560 prompts, 160 updates, DAPO (TMAX / open-instruct), 16 prompts × 8 samples per update, LR 1e-6,
     2 nodes, ~2.2 h.
   - Pool:
     - **640 European factual questions with the search tool available** (default mode). They are stratified by
       whether W2 knew the answer (160 known / 160 partly / 320 unknown).
     - **96 questions about made-up entities with search.** The only correct answer is `UNKNOWN`. Names that turned
       out to be real Wikipedia titles were removed.
     - **448 closed-book factual questions**, half default and half thinking mode. 288 of them are questions W2
       knew, a guard so that abstaining scores below answering (the failure of W3).
     - **32 closed-book made-up entities.**
     - **1,344 replay prompts**: business documents 376, math 384, verifiable instructions 192, code 192,
       tool calls 200.
   - Reward:
     - QA rows: +1 / 0 / −1 on the final answer turn only, so text inside a retrieved article never counts.
     - The other rows: their usual verifiers.
   - During training the search reward rose from −0.00 to +0.23, the search rate from 58 % to 99 %, and the
     closed-book QA reward from 0.10 to 0.20.

**The search tool** used in training and evaluation:
- an offline BM25 index of the lead sections of Wikipedia
  ([`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) 20231101) in 10 languages
  (en, de, fr, es, it, nl, sv, pl, pt, fi), with an exact-title boost;
- tool name `wiki_search`, arguments `query` and optional `language`;
- it returns the top 5 articles with an excerpt;
- up to 3 searches plus the answer turn.

To use the model with search, give it a tool with the same schema backed by any search service.

Training data pools (W1a, W2, W3) are published in [`birgermoell/oellm-rl-pools`](https://huggingface.co/datasets/birgermoell/oellm-rl-pools).
The E8 pool is built deterministically by
[`build_e8_pool.py`](https://github.com/BirgerMoell/oellm-rlvr/blob/main/scripts/build_e8_pool.py) from those pools,
the EU factual QA suite and W2's own samples.

## Evaluation

All evals are greedy. Thinking budget is 6K tokens unless noted.

**European factual QA** (600 held-out questions from Wikidata, 10 languages; head / torso / tail popularity bands):

| | Accuracy | Abstain | Wrong | Hallucination rate | Head band acc / abstain |
|---|---|---|---|---|---|
| E8 default | 0.265 | 0.290 | 0.445 | 0.605 | 0.535 / 0.100 |
| E8 thinking | 0.237 | 0.447 | 0.317 | 0.415 | 0.500 / 0.195 |
| W2 default | 0.280 | 0.047 | 0.673 | 0.935 | 0.525 / 0.025 |
| W3 default (the collapse) | 0.003 | 0.995 | 0.002 | – | 0.005 / 0.995 |

- **Made-up entities (500):** the model invents an answer for 0.236 (default) and 0.024 (thinking); W2 did for
  0.794 and 0.732.
- **Stated risk threshold:** when the prompt says a wrong answer costs three times a correct one, abstention rises
  from 0.29 to 0.44. Earlier checkpoints did not react to stated thresholds.

**With the search tool** (same questions):

| | Accuracy | Abstain | Wrong | Searched | Made-up: invents |
|---|---|---|---|---|---|
| E8 default | 0.533 | 0.373 | 0.093 | 100 % | 0.024 |
| E8 thinking | 0.565 | 0.295 | 0.140 | 99 % | 0.070 |
| W2 default | 0.507 | 0.167 | 0.327 | 86 % | 0.25 |

**Reasoning** (quick eval, 570 items; accuracy, default / thinking):

| | default | thinking |
|---|---|---|
| GSM8K | 0.767 | 0.800 |
| MATH-500 subset | 0.320 | 0.340 |
| MGSM | 0.730 | 0.750 |
| EU-language math | 0.342 | 0.550 |
| All | 0.586 | 0.649 (85 % of traces finish) |

**Tools** (held-out, 361 rows): call decision 0.906, tool names 0.809, valid JSON 1.000, valid arguments 0.954, no think-tag leaks.

**Code** (pass@1, base / plus):

| | default | thinking |
|---|---|---|
| HumanEval | 0.457 / 0.415 | 0.378 / 0.341 |
| MBPP | 0.407 / 0.341 | 0.407 / 0.333 |

**EUROPA** (36 languages).
- **Eval:** micro 0.695, macro 0.685. By domain: digital agency 0.810, grounding & safety 0.739, language
  understanding 0.768, practical communication 0.830, quantitative reasoning 0.278.
- **Alignment:** micro 0.688, macro 0.651. By domain: agentic control 0.594, dangerous capabilities 0.473,
  high-stakes fairness 0.795, human wellbeing 0.916, privacy & security 0.764, refusal calibration 0.363.
  Failed critical items: 140.

**European office benchmarks** ([european-rl](https://github.com/BirgerMoell/european-rl): synthetic tasks under
national rules in the 36 OpenEuroLLM languages; reward default / thinking):

| Benchmark | E8 |
|---|---|
| exact business arithmetic in local number formats | 0.35 / 0.34 |
| amount due under national VAT rules (5 levels) | 0.08 / 0.10 |
| formal business e-mail under language-specific rules | 0.20 / 0.14 |
| multi-step meeting rescheduling with tools | 0.00 / 0.00 |

**Not yet measured for this checkpoint:** oellm-eval retention (Belebele, Global-MMLU, INCLUDE, MultiBLiMP).

## Use

The chat template (`chat_template.jinja`) is the project's v2.1 two-mode template. The default is direct mode; pass
`enable_thinking=True` for thinking mode. Factual questions in our evals end with "answer on a final line
`ANSWER: <answer>`; if you are not sure, `ANSWER: UNKNOWN`". The model follows that convention well.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("birgermoell/oellm-9b-256k-stage5-e8-search-rl-step160")
model = AutoModelForCausalLM.from_pretrained("birgermoell/oellm-9b-256k-stage5-e8-search-rl-step160", torch_dtype="bfloat16", device_map="auto")
messages = [{"role": "user", "content": "In which year was the University of Tartu founded?"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=256)[0][ids.shape[1]:], skip_special_tokens=True))
```

With vLLM, serve with `--tool-call-parser hermes --enable-auto-tool-choice` to use tools (e.g. a `wiki_search` tool
with the schema above).

## Limitations and risks

- **A research preview, not a product.** No human red-teaming has been done. It is not reviewed for deployment.
- **Agentic control regressed strongly** (EUROPA 0.865 → 0.594). Require explicit confirmation before any action
  with side effects.
- **General chat capability is below the previous release** (EUROPA Eval 0.695 vs 0.727; language understanding
  0.768 vs 0.896).
- **Hallucination is reduced, not solved.**
  - Closed-book in default mode, 61 % of the questions it does not answer correctly still get a wrong answer
    rather than `UNKNOWN`.
  - Thinking mode over-abstains on well-known facts (20 %).
- **Search results are trusted.** The model was trained with a clean offline Wikipedia. It has not been tested
  against misleading or adversarial search results (prompt injection through retrieved text).
- **European office tasks:**
  - exact arithmetic and multi-step tool workflows are weak;
  - multi-step scheduling with tools does not work yet.
- **Languages:** trained and evaluated in many European languages, but quality varies by language. The search
  index covers 10 languages.

## License and citation

Apache-2.0, as the base model.

Training data licences are listed per source in the playbook and in
[`oellm-rl-pools`](https://huggingface.co/datasets/birgermoell/oellm-rl-pools). Wikipedia content is CC BY-SA 4.0
and is used only as a retrieval corpus.

If you use this checkpoint, please cite the OpenEuroLLM project and link this card.
