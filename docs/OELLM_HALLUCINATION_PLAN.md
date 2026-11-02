# OpenEuroLLM 9B: hallucination plan

Goal: cut factual hallucination (confident wrong answers) in all OELLM languages without losing knowledge or turning
into a model that refuses everything. We do it in post-training, inside the existing SFT → DPO → RL program
(`OELLM_RL_RUN_PROGRAM.md` §10), and measure it every round.

Code:
- oellm-rlvr `src/oellm_rlvr/eu_tasks.py` (eu_qa suite, ternary reward).
- oellm-rlvr `src/oellm_rlvr/hallucination.py` (fabricated entities, stated thresholds, self-consistency).
- oellm-rlvr `scripts/build_hallucination_evals.py`.
- qwen35-posttrain `scripts/eval_eu_tasks.py` (reports per condition).

## 1. What we measure

**eu_qa** is short factual questions built from Wikidata (CC0) about European people, places, works and
organisations.
- It covers 10 languages: en, de, fr, es, it, nl, sv, pl, pt, fi.
- Questions fall into three popularity bands: head, torso and tail.
- Every prompt says "if you are not sure, write `ANSWER: UNKNOWN`; a wrong answer counts against you".

Each reply is graded **correct / abstain / wrong** (a missing `ANSWER:` line counts as wrong). We report:

| Metric | Definition | Good direction |
|---|---|---|
| accuracy | correct / n | keep |
| wrong | wrong / n | down |
| abstain | abstain / n | up on tail, ~0 on head |
| hallucination rate | wrong / (n − correct): of the questions it doesn't get right, how often it makes something up | down |
| knowledge index | 100·(correct − wrong)/n (AA-Omniscience; 0 = always abstain) | up |

**hallu-v1** (built from the eu_qa eval, 600 questions, held out from every RL pool by subject):

| Condition | Rows | Question it answers |
|---|---|---|
| `standard` | 600 | baseline, the prompt as in eu_qa |
| `t50` / `t75` / `t90` | 600 each | Does abstention follow the stated stakes? The prompt says "answer only if you are > t confident; a wrong answer costs t/(1−t) points" (1, 3, 9 points; Kalai et al. 2025). A calibrated model abstains more as t rises. |
| `fabricated` | 50 per language | Questions about people, films, books, companies and rivers that do not exist (names built from syllables; any exact Wikidata label/alias match dropped). The only right answer is UNKNOWN. This is HalluLens NonExistentRefusal in our languages. |

**Over-refusal guards:**
- abstention on the head band, where the model mostly knows the answer;
- EUROPA refusal calibration;
- the general knowledge benchmarks in oellm-eval.

**External anchors** (when a release is planned):
- Multilingual SimpleQA Verified (de, fr, es, it, nl, pl, cs, da);
- swedish-facts-v1 (arXiv 2510.21360);
- SimpleQA Verified (en);
- AbstentionBench (over- and under-abstention).

## 2. Where we are (11-02)

**eu_qa eval, greedy** (`eval/eu-tasks/*/report.json`):

| Model | Mode | Acc | Abstain | Wrong | Halluc. rate | Index | Tail abstain |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | default | 0.16 | 0.15 | 0.69 | 0.82 | −53 | 0.20 |
| Qwen3-8B | thinking | 0.20 | 0.23 | 0.56 | 0.70 | −36 | 0.38 |
| A4b (SFT+tool RL) | default | 0.27 | 0.03 | 0.70 | 0.95 | −44 | 0.04 |
| W1a | default | 0.27 | 0.03 | 0.69 | 0.95 | −42 | 0.03 |
| W2 | default | 0.28 | 0.05 | 0.67 | 0.94 | −39 | 0.06 |
| W2 | thinking | 0.26 | 0.04 | 0.70 | 0.95 | −44 | 0.05 |

- Our model **knows more than Qwen3-8B** (0.28 vs 0.16; head band 0.53 vs 0.31).
- But it **almost never abstains**. When it doesn't know, it guesses in 94 % of cases; Qwen3-8B guesses in 70–82 %.
- RL so far (W1a, W2) moved neither number. W1a/W2 had no eu_qa rows and a binary reward elsewhere (§10.5 of the run
  program).

**Self-consistency headroom (experiment E1, done).** W2 sampled k=8 at temperature 1 on 512 eu_qa training prompts
(4,096 samples, from the W3 pass-rate profile). How often a sample's answer recurs among the eight predicts whether
it is right: **AUROC 0.87**. Accuracy by agreement:

| Agreement | 1/8 | 2/8 | 3/8 | 4/8 | 5/8 | 6/8 | 7/8 | 8/8 |
|---|---|---|---|---|---|---|---|---|
| n | 2128 | 310 | 126 | 100 | 85 | 72 | 56 | 56 |
| accuracy | 0.04 | 0.13 | 0.26 | 0.48 | 0.77 | 0.83 | 1.00 | 0.71 |

Policy "answer only if at least τ of the samples agree, else UNKNOWN":

| τ | Correct | Wrong | Abstain | Halluc. rate | Index |
|---|---|---|---|---|---|
| as sampled | 0.102 | 0.823 | 0.074 | 0.92 | −72.1 |
| 0.375 | 0.074 | 0.171 | 0.755 | 0.19 | −9.8 |
| 0.5 | 0.066 | 0.092 | 0.842 | 0.10 | −2.7 |
| 0.625 | 0.054 | 0.037 | 0.909 | 0.04 | +1.7 |

**Conclusion:** the model already carries the signal for what it doesn't know but does not act on it. This is the
cheap case in the literature: a training signal that rewards abstaining when uncertain can recover most of this
headroom without new knowledge. Caveat: this pool is tail-heavy, so the head band will lose less accuracy than these
averages suggest.

## 3. Why it happens (literature)

- **Binary grading rewards guessing.** Under right/wrong scoring a guess has positive expected value, so pre-training,
  SFT and binary RLVR all teach the model to guess (Kalai et al., "Why Language Models Hallucinate", arXiv
  2509.04664).
- **RL fine-tuning lowers abstention further:**
  - The "hallucination tax" of RFT: refusals fell by more than 80 % (arXiv 2505.13988).
  - AbstentionBench: reasoning fine-tuning cut abstention by about 24 % (arXiv 2506.09038).
  - OpenAI's o3 and o4-mini hallucinate more on PersonQA than o1.
- **SFT on facts the model doesn't know raises hallucination.** Each fitted "unknown" fact costs accuracy, and the
  model learns to produce confident answers regardless (Gekhman et al., arXiv 2405.05904).

## 4. Methods, ranked by evidence per cost for our pipeline

| # | Method | Evidence | Cost here | Where |
|---|---|---|---|---|
| 1 | **Ternary reward + unanswerable prompts in RL** (+1 / 0 / −1) | TruthRL (arXiv 2509.25760): hallucination 39.5 % → 19.4 % vs binary RL at −3.7 pp accuracy; the simple ternary reward beat more elaborate variants. "Hallucination tax": ~10 % unanswerable data restores refusal. | Reward exists (eu_qa); fabricated generator exists; pool share only | W3 (eu_qa), W4+ |
| 2 | **Abstention DPO from self-consistency** (on-policy pairs) | Self-consistency confidence as the factuality judge works without labels (FactTune, arXiv 2311.08401: −40–58 % factual errors at 7B). Our E1 shows the signal is strong (AUROC 0.87). | Pairs from k=8 samples we already draw; one DPO run | Stage 4b (on W3) |
| 3 | **Calibration reward (RLCR)**, Brier score on stated confidence | arXiv 2507.16806: ECE 0.37 → 0.03 at equal accuracy; holds out of domain, where binary RLVR degrades it | Reward change + answer format | W4 arm |
| 4 | **Knowledge-aware SFT data**: probe each fact-seeking SFT item per language; drop unknowns or relabel them as abstentions | Gekhman et al. arXiv 2405.05904; R-Tuning arXiv 2311.09677 (alone it over-refuses) | k samples per item; next SFT round | Next SFT |
| 5 | **Long-form factuality DPO** (atomic-claim verification) | Mask-DPO arXiv 2503.02846 (ANAH 49 → 78; biography FActScore 30 → 39); FactTune | gemma4 judge + Wikipedia per language; moderate | Stage 4b+ |
| 6 | **Grounded faithfulness** for eu_doc / RAG (Context-DPO, SSFO) | arXiv 2412.15280, 2508.17225 | Low; pairs with and without the context | later |
| 7 | **Online long-form factuality RL** (VeriScore reward) | arXiv 2508.05618: −23 pp hallucination, no loss in helpfulness | High judge compute | only if needed |

**Known failure modes and guards:**
- **Abstention collapse.** TruthRL saw it with a rule-based verifier. Guard with head-band abstention ≤ 0.10 and
  "known" prompts in every batch: prompts with agreement ≥ 6/8 and correct, where abstaining costs reward against
  answering.
- **Answering less.** The model can learn to give vaguer or fewer claims (long-form methods). Gate on helpfulness and
  length.
- **Thinking mode.** It hallucinates at the same rate. Train and gate both modes.
- **Over-refusal on safety/benign prompts.** Gate on EUROPA refusal calibration.

## 5. Experiments

| ID | What | Status | Cost (local-h) |
|---|---|---|---|
| E0 | eu_qa baselines across checkpoints (§2) | done | 0 (existing) |
| E1 | Self-consistency headroom on W2 (§2) | done | 0 (existing samples) |
| E2 | hallu-v1 on W2, Qwen3-8B, Apertus-8B: do stated thresholds move abstention; fabricated-entity abstention | done (§6) | 8 |
| E3 | hallu-v1 + eu_qa three-way on W3, against W2: did the ternary eu_qa reward in RL already move abstention? | after W3 | ~8 |
| E4-pilot | The E4 recipe on **W2** as a throwaway pilot: about 3K abstention pairs (W2 samples) + 3K general Stage-4 pairs, 1 epoch DPO on 2 nodes. Chain: data 63225457 → DPO 63225458 → export 63225459 → hallu-v1 63225461 + EUROPA 63225462 | running (11-02) | ~115 |
| E4 | Abstention-DPO pairs from W3 samples (below), as part of Stage 4b with identity and safety | after W3 and E4-pilot | ~25 sampling + the 4b DPO |
| E5 | W4-H RL arm: ternary eu_qa + 15 % fabricated + "known" prompts; optional RLCR confidence variant; vs W4 without | after E3/E4 | estimate before launch (~400 for a 1-node 12 h arm) |
| E6 | Knowledge probe of the fact-seeking part of the SFT mix (per language) | next SFT round | ~50 |
| E7 | **Search baseline**: W2 with an offline Wikipedia `wiki_search` tool on hallu-v1 (standard + fabricated), both modes. Does it search when unsure, and does it answer from the results? | index 63227897 → eval 63229550 | ~40 |
| E8 | **Search RL window** (T-search): eu_qa + fabricated prompts with the search tool in rollouts, ternary reward, from W3 (or W3 + 4b) | after E7 | estimate before launch (~400 for 1 node × 12 h) |

**E4 pair construction** (on-policy, from the checkpoint being trained; eu_qa train pool and fabricated prompts
only, disjoint from hallu-v1 by subject):
- **Agreement ≤ 2/8, majority answer wrong:** chosen = a sampled `UNKNOWN` reply (or the model's own reply with the
  answer line replaced), rejected = a sampled wrong guess.
- **Fabricated entity:** chosen = UNKNOWN, rejected = any invented answer.
- **Agreement ≥ 6/8, majority answer correct:** chosen = the correct reply, rejected = UNKNOWN. This is the
  over-refusal guard, about 1/3 of the pairs.
- Default and thinking mode in the proportions of the RL pool.

**Gate for any checkpoint that changes the hallucination recipe** (vs its parent; all on hallu-v1 and the standard
gates):
- `standard` hallucination rate down ≥ 0.15 absolute, with `standard` accuracy down ≤ 0.02 and head-band abstention
  ≤ 0.10;
- `fabricated` abstention ≥ 0.7;
- abstention increases monotonically t50 → t75 → t90;
- no regression on EUROPA refusal calibration, the safety gate or the oellm-eval knowledge benchmarks.

### Search (E7, E8)

Retrieval is the only fix for long-tail facts: tail accuracy is 7–9 %, and pre-training or RL will not teach a 9B
model those facts (Kandpal et al. arXiv 2211.08411; PopQA arXiv 2212.10511). Search complements abstention rather
than replacing it:
- Deciding *when* to search needs the same calibration as abstaining.
- A search-using model hallucinates in new ways: it misreads results, or answers when the search finds nothing.
- The model often runs offline (Ollama, MLX), where it still has to abstain.

**Setup.** Leonardo's compute nodes are offline, so the search is local, in the style of Search-R1 (arXiv
2503.09516):
- **Index:** BM25 over the lead sections of Wikipedia (wikimedia/wikipedia 20231101 @ `b04c8d1c`, CC BY-SA), one
  index per language (en, de, fr, es, it, nl, sv, pl, pt, fi).
  - Built by oellm-rlvr `scripts/leonardo_wiki_search_index.sbatch`.
  - Served by `scripts/wiki_search.py serve` from its own venv.
- **Tool in RL rollouts:** `wiki_search(query, language)` through open-instruct's tool support
  (`OELLM_PATCH_WIKI_SEARCH_TOOL=1`, hermes parser).
- **Eval:** `scripts/eval_search_qa.py`.

**E8 design (if E7 shows the model can use the tool):**
- **Pool:**
  - about 60 % eu_qa train questions not used before;
  - about 15 % fabricated entities, where the search finds nothing and the right answer is UNKNOWN;
  - about 25 % W3 slices, so other skills hold.
- **Reward:** +1 / 0 / −1 on the final answer, at most 3 searches, tool output masked from the loss.
- **What to watch:** search rate on head-band questions (it should stay low when the model knows), answers that
  contradict the retrieved text, and accuracy without the tool (it must not drop).

## 6. Results log

- **11-02 E3 (W3 vs W2): abstention collapse.** eu_qa default mode: accuracy 0.28 → 0.003, abstain 0.05 → 0.995;
  thinking: accuracy 0.26 → 0.18, abstain 0.04 → 0.67, head-band abstain 0.01 → 0.42. The ternary reward alone, on a
  pool where most prompts are never solved, teaches "always UNKNOWN". The guard in §4 (known prompts where abstaining
  loses reward) is necessary, not optional. Fixes for W4:
  - balance eu_qa by pass rate on the current checkpoint: prompts solved ≥ 4/8 alongside the unknown ones;
  - give an abstention on a known prompt a negative reward (e.g. −0.5);
  - gate on head-band abstention.

- **11-02 E0, E1:** see §2.
- **11-02 E2** (hallu-v1, greedy; `eval/eu-tasks/hallu-v1-*`; jobs 63218034–36). Columns are the rate of invented
  answers to fabricated entities, standard eu_qa results, and abstention under each stated threshold:

  | Model | Mode | Fabricated: invents an answer | Standard acc | Standard halluc. rate | Abstain t50 / t75 / t90 |
  |---|---|---|---|---|---|
  | W2 | default | **0.79** | 0.28 | 0.94 | 0.11 / 0.09 / 0.07 |
  | W2 | thinking | **0.73** | 0.28 | 0.94 | 0.04 / 0.04 / 0.03 |
  | Qwen3-8B | default | 0.43 | 0.16 | 0.82 | 0.09 / 0.09 / 0.09 |
  | Qwen3-8B | thinking | 0.14 | 0.20 | 0.70 | 0.26 / 0.25 / 0.26 |
  | Apertus-8B-Instruct | default | 0.55 | 0.26 | 0.77 | 0.13 / 0.14 / 0.13 |

  Apertus has no thinking mode; its thinking-mode rows are invalid and not shown.

  - **W2 invents details for most nonexistent people, films and companies:** 4 in 5, against 1 in 7 for Qwen3-8B
    thinking.
  - **No model's abstention follows the stated stakes.** Abstention does not rise from t50 to t90, and the
    threshold prompt often lowers it relative to `standard`. The prompt alone cannot fix this. It needs training
    (methods 1–2), and the threshold-monotonicity gate is a target none of the tested models meets yet.
  - Thinking helps Qwen3 a lot on fabricated entities (0.43 → 0.14) but helps W2 little (0.79 → 0.73). Our closed-
    trace thinking does not yet check whether an entity exists.
