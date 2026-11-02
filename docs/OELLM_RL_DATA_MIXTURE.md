# RL data mixture by training window

This page lists the prompt pools of the combined RL windows of OpenEuroLLM 9B (Stage 5) trained so far.
- W2 and W3 each start from the previous window's final checkpoint.
- Every row is scored by a programmatic verifier; there is no learned reward model.
- Counts are prompts in the pool. Each training update samples 16 prompts × 8 completions, one pass over the pool.

**The exact pools are on Hugging Face:** [`birgermoell/oellm-rl-pools`](https://huggingface.co/datasets/birgermoell/oellm-rl-pools)
(configs `w1a`, `w2`, `w3`; one row per prompt with its verifier label). Every source dataset is listed with its
revision and license under [Where the data is](#where-the-data-is). The run-by-run record is in the
[RL run program](OELLM_RL_RUN_PROGRAM.md).

| Window | Prompts | Updates | Status | Focus |
|---|---|---|---|---|
| W1a | 2,048 | 128 | done | math (EU languages + English), default-mode answers, code |
| W2 | 2,560 | 160 | done, released | adds multi-turn tools and instruction following |
| W3 | 2,560 | 160 | training (started 2026-11-02) | adds EU factual QA with abstention and EU document tasks; 1,688 thinking / 872 default-mode prompts |

## Mixture

Prompts per slice, with the share of the window in brackets. Modes:
- **thinking**: the reply reasons in a closed trace before the answer.
- **default**: the reply answers directly.

| Verifier / slice | Mode | W1a | W2 | W3 |
|---|---|---:|---:|---:|
| `multilingual_math` | thinking | 560 (27 %) | 682 (27 %) | 255 (10 %) |
| `math` | thinking | 540 (26 %) | 214 (8 %) | 257 (10 %) |
| `math_direct` | default | 300 (15 %) | – | – |
| `math_direct_shaped` | default | – | 192 (8 %) | – |
| `mcq_letter_direct` | default | 300 (15 %) | 192 (8 %) | – |
| `code_stdio` | thinking | 348 (17 %) | 384 (15 %) | 256 (10 %) |
| `tool_call` | both | – | 512 (20 %) | 384 (15 %) |
| `ifeval_direct` | default | – | 256 (10 %) | 192 (8 %) |
| `ifeval_think` | thinking | – | 128 (5 %) | 64 (3 %) |
| `eu_qa` | both | – | – | 512 (20 %) |
| `eu_doc` | both | – | – | 640 (25 %) |
| **Total** | | **2,048** | **2,560** | **2,560** |

## What each slice is

| Slice | Content and source | Reward |
|---|---|---|
| `multilingual_math` | Math word problems in EU languages (oellm-math-rlvr, levels 3–5). W2 used prompts profiled to be solved 1–7 times out of 8. | 1 if correct and the final answer is in the prompt language |
| `math` | English math (oellm-math-rlvr, GSM8K-style) | 1 / 0, boxed answer, symbolic equivalence |
| `math_direct`, `math_direct_shaped` | Math asked for a bare final answer, with no reasoning shown | W1a: exact bare answer. W2: 1.0 for a bare correct answer, 0.2–0.6 for a correct answer with working (shorter scores higher) |
| `mcq_letter_direct` | Multilingual exam multiple choice, answered with the letter only | 1 / 0 |
| `code_stdio` | Programming problems with stdin/stdout tests (oellm-code-rlvr, 12 families) | 1 if all tests pass (sandboxed execution) |
| `tool_call` | Multi-turn tool use (function calling): English, plus a machine-translated slice in EU languages (user turns only; argument strings protected) | Correct call / no-call decision, function names and JSON arguments |
| `ifeval_direct`, `ifeval_think` | Instruction following with checkable constraints (nvidia/Nemotron-RL-instruction_following, ODC-BY; constraints validated against the IFEval/IFBench checkers) | Fraction of constraints satisfied |
| `eu_qa` | Short factual questions about European people, places, works and organisations, from Wikidata (CC0). 10 languages, in head / torso / tail popularity bands.  | +1 correct, 0 for `UNKNOWN`, −1 wrong (abstaining scores better than guessing wrong) |
| `eu_doc` | Generated business documents in 10 languages, with locale-formatted numbers and dates: invoices with VAT, staff tables, contracts, room bookings, and letters to JSON. Same public dataset. | 1 / 0 for an exact answer; JSON tasks: fraction of fields correct |

Every slice also gets −0.5 when a reply does not finish within the response budget.

## Where the data is

| Data | Hugging Face (revision) | License |
|---|---|---|
| **The W1a / W2 / W3 pools as trained** | [birgermoell/oellm-rl-pools](https://huggingface.co/datasets/birgermoell/oellm-rl-pools) | per slice (below) |
| `multilingual_math`, `math`, `math_direct`, `math_direct_shaped` | [birgermoell/oellm-math-rlvr](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) @ `0ffc9d6d` | Apache-2.0 |
| `math` (GSM8K rows) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` @ `740312ad` | MIT |
| `mcq_letter_direct` | [birgermoell/oellm-eu-exam-mcq-v1](https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1) (EXAMS rows only) | CC-BY-SA-4.0 |
| `code_stdio` | [birgermoell/oellm-code-rlvr](https://huggingface.co/datasets/birgermoell/oellm-code-rlvr) @ `e1cae771` | Apache-2.0 |
| `tool_call` | [nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1) @ `0ac34ec8`, [nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) @ `bfc7a4dd`, [MadeAgents/xlam-irrelevance-7.5k](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k) @ `34323bf0`. The translated slice is only in the pools dataset. | CC-BY-4.0 |
| `ifeval_direct`, `ifeval_think` | [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) @ `bcdec57a` | ODC-BY |
| `eu_qa`, `eu_doc` | [birgermoell/oellm-eu-tasks](https://huggingface.co/datasets/birgermoell/oellm-eu-tasks) @ `50092422` (train split; the test split is the eval) | CC-BY-4.0 |

## How much of W3 carries learning signal

Before W3, we sampled 8 completions per prompt from W3's starting checkpoint (W2) and scored them with the
training verifiers (oellm-rlvr pass-rate filter). GRPO only learns from prompts whose 8 rewards differ: a prompt
that is always solved or never solved contributes nothing. **62 % of W3's prompts carry signal.**

| Slice | Prompts | With signal | Share | Main loss |
|---|---:|---:|---:|---|
| `eu_doc` | 640 | 366 | 57 % | 270 never solved |
| `eu_qa` | 512 | 290 | 57 % | 216 never solved |
| `tool_call` | 384 | 253 | 66 % | 80 always solved |
| `multilingual_math` | 255 | 183 | 72 % | 71 never solved |
| `math` | 257 | 177 | 69 % | 69 always solved |
| `code_stdio` | 256 | 173 | 68 % | 76 never solved |
| `ifeval_direct` | 192 | 111 | 58 % | 42 incomplete (scoring) |
| `ifeval_think` | 64 | 43 | 67 % | 9 incomplete (scoring) |

The "incomplete" instruction-following rows hit a missing tokenizer resource during scoring, since fixed; they say
nothing about difficulty.

## Notes

- **Algorithm:** DAPO, a GRPO variant: group-relative advantages, no KL term, asymmetric clipping, and groups
  without reward variance filtered out.
- **Unscored slices in W1a and W2:** a bug in how verifier patches were registered left some slices without a reward.
  - W1a: `multilingual_math`.
  - W2: `ifeval_*`, `math_direct_shaped`, `mcq_letter_direct` and `multilingual_math`.

  Those rows only received the unfinished-reply penalty, so they added no task signal. W3 is the first window with
  every verifier active.
- **Planned for W4:**
  - pass-rate filtering on the current checkpoint;
  - fewer always-solved math and tool rows, and easier EU tasks;
  - fabricated-entity questions, where the correct answer is `UNKNOWN`, to reduce hallucination.

*Status as of 2026-11-02. Counts from the pool manifests of each window.*
