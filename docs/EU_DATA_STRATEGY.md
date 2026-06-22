# European-Language Data Strategy for Qwen 9B Post-Training

Created: 2026-06-22

This is the data plan for turning the current Qwen 9B EU-delta run into a
world-class European-language post-training recipe. The current Dolci/EuroBlocks
pipeline is a good bootstrap. It is not enough by itself: a top model needs
native European-language instruction signal, multilingual verifiable reasoning,
on-policy preference data, tool/agent trajectories, and long-context tasks built
from real European documents.

## Bottom line

Do not add more translated SFT as the main move. Add or make these data products:

1. `oellm-eu-native-instruct-v1`
2. `oellm-eu-reasoning-v1`
3. `oellm-eu-preference-v1`
4. `oellm-eu-tool-agent-v1`
5. `oellm-eu-longctx-instruct-v1`
6. `oellm-eu-safety-civic-v1`
7. `oellm-eu-eval-holdouts-v1`

The most important missing ingredient is native-authored EU-language data turned
into supervised, preference, and verifiable tasks. Translated data is useful for
coverage, but it should be filtered, downweighted, and corrected with
anti-translationese preference pairs.

## Existing backbone to keep

Keep these as the backbone because they are already aligned with the current
repo and OpenEuroLLM infrastructure:

| Stage | Keep using | Role |
| --- | --- | --- |
| SFT | `openeurollm/Dolci-Instruct-SFT-translated`, `openeurollm/dolci-instruct-sft-tokenized` | broad multilingual instruction bootstrap |
| Reasoning SFT | `openeurollm/Dolci-Think-*`, `open-thoughts/OpenThoughts-114k`, `open-thoughts/OpenThoughts2-1M`, `nvidia/OpenMathInstruct-2` | reasoning traces and math/code/science coverage |
| DPO/SimPO | `openeurollm/Dolci-Instruct-DPO-translated`, Dolci Think-DPO | preference bootstrap |
| RLVR/GRPO | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, `allenai/Dolci-RL-Zero-Mix-7B` | verifiable math/code/IF training |
| Long context | `birgermoell/oellm-longctx-tokenized-streamed-all-v2`, FinePDFs-Edu tiers | native long-document source |
| Filtering | `openeurollm/propella-annotations`, decontamination tools | quality selection and contamination control |
| Eval | `openeurollm/ArenaHard-EU-v0`, OneRuler/RULER, lm-eval multilingual tasks | stage gates |

The current 85/15 EuroBlocks/Tulu SFT run is still a good first proof that the
stack trains. It should not be the final mix.

## Data products to add or make

### 1. `oellm-eu-native-instruct-v1`

Goal: 500k-1M high-quality SFT conversations grounded in native European text.

Primary sources:

- FinePDFs-Edu and the project long-context corpus for 35 OpenEuroLLM target
  languages.
- HPLT for Irish, Albanian, and Luxembourgish, plus any other low-resource gaps.
- `data.europa.eu`, EUR-Lex, EU publications, DGT/JRC resources, EU Bookshop,
  national open-data portals, parliament/government pages, and university/public
  science PDFs where license metadata permits reuse.
- Wikimedia/Wikisource and other permissive cultural/encyclopedic sources for
  language style and public-domain literature.

Task types to synthesize:

- grounded QA with evidence spans
- document and section summarization
- explain-this-policy/instruction in plain language
- compare two official documents
- rewrite into local register: formal, neutral public-service, concise
- extract fields from forms, tables, reports, and public notices
- locale-specific writing: emails, applications, complaints, meeting summaries

Quality gates:

- source URL, license, language, and dedupe ID required
- answer must be grounded in source snippets when source-grounded
- language ID must match both prompt and answer
- reject obvious machine-translationese unless deliberately used as a negative
- Propella/high-educational-value filters where available
- PII scan and exact/minhash dedupe before training

Target minimums:

| Language group | Minimum SFT examples |
| --- | ---: |
| high-resource EU languages | 30k-80k each |
| mid-resource EU languages | 15k-40k each |
| low-resource and regional languages | 5k-20k each |

Prioritize language debt first: `ga`, `mt`, `cy`, `lb`, `sq`, `eu`, `gl`, `is`,
`mk`, `sl`, `lt`, `lv`, `et`, `hr`, `sk`.

### 2. `oellm-eu-reasoning-v1`

Goal: 200k-400k multilingual reasoning examples where the problem, reasoning
trace style, and final answer work naturally in the target language.

Sources:

- OpenThoughts and Dolci Think for seed tasks.
- OpenMathInstruct and Numina-style math for verifiable answers.
- Permissively licensed national school exams, olympiad-style problems, public
  university exercises, scientific explainers, and logic puzzles.
- Code tasks from permissive competitive-programming or unit-test datasets.

Required construction:

- translate/localize seed problems, then judge for native fluency and correctness
- generate original problems from native sources, not just translations
- keep verifiers for math, exact-answer, multiple-choice, code, and symbolic tasks
- include "answer in user's language" and "think in user's language" variants
- add short-reasoning and long-reasoning versions, because not every EU-language
  query should produce a giant chain of thought

Do not rely on English CoT replay alone. That protects reasoning, but it will not
teach Finnish, Greek, Polish, Swedish, or Maltese reasoning style.

### 3. `oellm-eu-preference-v1`

Goal: 150k-300k preference pairs, including on-policy negatives from the current
Qwen checkpoints.

Pair families:

- native answer preferred over translationese answer
- grounded answer preferred over plausible hallucination
- correct local register preferred over too-American/too-English register
- answer in requested language preferred over English drift
- concise public-service answer preferred over verbose generic answer
- correct tool/no-tool behavior preferred over tool hallucination
- safer medical/legal/civic answer preferred over overconfident advice

Construction:

1. Sample prompts from `oellm-eu-native-instruct-v1`.
2. Generate 2-4 candidate responses from base, SFT checkpoint, and a stronger
   teacher.
3. Judge with Propella/LLM judges, automatic grounding checks, and targeted human
   spot checks by language.
4. Build DPO/SimPO pairs and keep the full response pool for future on-policy
   refreshes.

Immediate repo note: `configs/dpo_qwen35_9b.yaml` currently points at English
Dolci DPO. For a real EU run, replace that with a mix including
`openeurollm/Dolci-Instruct-DPO-translated` and on-policy native pairs.

### 4. `oellm-eu-tool-agent-v1`

Goal: 100k-250k tool-use and agentic trajectories with multilingual user
instructions, executable validation, and European workflows.

Seed sources:

- Dolci tool-use data
- Berkeley Function Calling Leaderboard as eval format inspiration
- APIGen/xLAM-style function-calling data
- ToolACE-style generation and verification pipeline ideas

Make the OpenEuroLLM-specific version instead of only importing English tool data.

Domains:

- calendar/email/document/spreadsheet operations
- public holidays, weather, maps, train/flight-style itinerary APIs
- data.europa.eu search, Eurostat-style tables, EUR-Lex lookup, VIES/VAT-style
  validation, company registry style tasks
- code assistant loops: edit, run tests, inspect error, patch again
- long-context tool use over retrieved documents

Trajectory types:

- single call, parallel calls, multi-turn missing-parameter clarification
- no-tool-needed and unavailable-tool cases
- tool result summarization in the user's language
- invalid API response recovery
- multi-step planning with JSON/function-call strictness

Function names and schemas can stay English; natural-language instructions and
tool outputs should vary by language. Verify every trajectory by executing the
tool stub or by strict JSON/schema checks.

### 5. `oellm-eu-longctx-instruct-v1`

Goal: 50k-150k long-context SFT examples from real EU-language documents.

Use the existing long-context corpus as source material, not only for CLM. Build:

- 16k, 32k, 64k, 128k, and 256k examples
- evidence QA with answer spans in early, middle, and late context
- multi-document comparison
- aggregation over tables/records
- no-answer and insufficient-evidence negatives
- long summarization with section-level constraints
- tool-use over long retrieved context

The current 4k SFT run is operationally useful but cannot preserve a 262k-context
base by itself. Add a separate low-learning-rate long-context SFT stage or mix
long examples into final SFT with explicit length buckets.

### 6. `oellm-eu-safety-civic-v1`

Goal: 50k-100k multilingual examples for safe, useful European deployment.

Coverage:

- GDPR/privacy and PII handling in natural user scenarios
- EU AI Act style transparency and refusal boundaries
- medical, legal, financial, and immigration-adjacent advice with proper caveats
- hate/harassment and minority-language robustness
- civic misinformation, elections, public services, sanctions, consumer rights

This should be mostly preference data and short SFT exemplars, not a giant safety
overfit stage.

### 7. `oellm-eu-eval-holdouts-v1`

Goal: private held-out eval/dev sets before large synthetic generation.

ArenaHard-EU is useful but small. Add 200-500 prompts per target language across:

- general chat/instruction
- native style and register
- grounded QA
- math/reasoning
- tool/function calling
- long-context retrieval and summarization
- safety/civic tasks

Never train on these. Denylist source documents, synthetic key namespaces, and
prompt templates from data generation.

## Recommended stage mixes

### Qwen 9B instruct-delta run

For the near-term instruct checkpoint:

| Component | Weight |
| --- | ---: |
| filtered Dolci/EuroBlocks multilingual SFT | 35% |
| `oellm-eu-native-instruct-v1` | 20% |
| `oellm-eu-reasoning-v1` | 15% |
| English replay from Tulu/Dolci/OpenThoughts | 10% |
| `oellm-eu-tool-agent-v1` | 8% |
| `oellm-eu-longctx-instruct-v1` | 7% |
| `oellm-eu-safety-civic-v1` | 5% |

Use temperature/tempered language sampling so English, German, French, and
Spanish do not drown out lower-resource languages.

### Qwen 9B base-full run

Use stages:

1. Reasoning SFT: 50% Dolci Think/OpenThoughts, 30% `oellm-eu-reasoning-v1`,
   20% code/math replay.
2. General SFT: same as instruct-delta mix, but slightly more generic Dolci and
   English replay.
3. DPO/SimPO: 45% translated Dolci DPO, 35% `oellm-eu-preference-v1`, 10%
   reasoning preference, 10% safety/tool preference.
4. RLVR/GRPO: 40% math, 25% instruction-following constraints, 20% code/tool,
   15% multilingual exact-answer and locale-specific verifiable tasks.

## Five-week execution plan

| Week | Data work | Training/eval work |
| --- | --- | --- |
| 1 | freeze eval holdouts, build language coverage matrix, run Propella/MDQ-style source triage, create native-instruct v0 for 10-15 key languages | baseline Qwen 9B and current SFT/DPO run |
| 2 | generate native-instruct v0, preference v0, reasoning v0; start longctx-instruct from existing corpus | 2B/4B ablations for mix ratios, LR, length mix |
| 3 | on-policy generations from best 9B SFT; build anti-translationese and grounded preference pairs | 9B SFT v1, DPO/SimPO v1, RLVR smoke |
| 4 | expand weak languages and tool-agent trajectories; refresh eval with error-driven prompts | 9B final candidates, GRPO if ROCm/vLLM stable |
| 5 | decontam, license manifest, model card data section, final held-out eval | final 9B eval, optional 27B/35B-A3B scale if 9B gates pass |

The 300k GPU-hour budget is not the binding constraint for the 9B run. The binding
constraint is data generation, verification, and eval turnaround. Spend compute
on ablations, judging, on-policy sampling, and one larger scale-up only after the
9B data recipe clearly wins.

## Evaluation gates

Do not ship based on aggregate multilingual averages. Gate per language and per
capability:

- ArenaHard-EU win rate improves overall and for each reported language.
- mMMLU or equivalent does not regress for high- and mid-resource languages.
- Low-resource smoke prompts improve in `ga`, `mt`, `cy`, `lb`, `sq`, `eu`, `gl`.
- GSM8K/MATH/LCB do not drop beyond the current thresholds.
- Tool-call strict JSON validity improves and no-tool hallucination decreases.
- Long-context retrieval at 32k/64k/128k does not regress.
- Human/native review sample shows less translationese than the base and the
  first SFT checkpoint.

## Source triage rules

Use as primary:

- native official/public/educational/scientific texts with metadata
- permissively licensed PDF/document corpora
- high-quality filtered web corpora with source manifests
- verifiable reasoning/task data with answer checks
- on-policy preference pairs

Use carefully:

- OPUS, DGT, EuroParl, JRC-Acquis, and translation memories. They are valuable
  for terminology, alignment, and contrastive preference data, but they can make
  the model sound translated if overused as SFT.
- subtitles and conversational corpora. Useful for chat style, risky for quality
  and safety if not filtered.
- raw Common Crawl derivatives. Use only with quality filters, license review,
  dedupe, and language ID.

Reject:

- examples without source/license metadata
- synthetic answers that cannot be grounded or verified
- translated examples with obvious source-language word order
- eval-contaminated prompts or source documents
- PII-heavy public records unless transformed into safe, synthetic patterns

## Immediate next changes

1. Add a data registry file for the seven `oellm-eu-*` artifacts, with language
   quotas and source manifests.
2. Change DPO from English-only Dolci to translated Dolci plus on-policy native
   preference data.
3. Add a long-context SFT config at 16k/32k first, with tiny weight and strict
   eval gates, before trying 128k+ SFT.
4. Build the eval holdout first, then generate data. This prevents accidental
   self-contamination.
5. Run small 2B/4B ablations on data mix quality before spending the 9B/27B
   budget.

