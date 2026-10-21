# OpenEuroLLM 36-language Instruct and Reasoning data plan

Status: research and selection plan, 2026-10-20. These are **candidate inputs**, not a
claim that a production mixture has been cleared, materialized, or trained.
The weights remain the 256K anneal300b base named in
`OELLM_EXCELLENT_256K_TWO_STAGE_LEONARDO.md`. Stage 1 trains direct-answer
Instruct; Stage 2 starts from the accepted Stage-1 checkpoint and trains
genuine, nonempty reasoning with direct-answer replay.

## Language contract

Use the [OpenEuroLLM catalogue's 36 macrolanguages](https://github.com/OpenEuroLLM/training-data-catalogue/blob/main/languages)
as the allowlist:

| Group | ISO-639-1 keys (36 total) |
| --- | --- |
| EU official (24) | `bg cs da de el en es et fi fr ga hr hu it lt lv mt nl pl pt ro sk sl sv` |
| Co-official (3) | `ca eu gl` |
| Candidate-member (7) | `bs ka mk sq sr tr uk` |
| Closely associated (2) | `is no` |

The catalogue also specifies script and standard variants: Serbian Cyrillic
and Latin, Norwegian Bokmål and Nynorsk, and Estonian/Latvian/Albanian variants.
Normalize two-letter source codes to the catalogue's three-letter + script
codes in the final manifest; do not collapse distinct writing systems during
quality review. The tokenizer can encode languages outside this set, but these
36 are the intended model targets. Exclude non-target examples such as
Japanese, Welsh, Luxembourgish, and Russian from *this* run, while retaining
their source IDs in the rejection report.

## Stage 1 — direct-answer Instruct SFT

| Candidate | Role and observed coverage | Admission gate |
| --- | --- | --- |
| [OpenEuroLLM Excellent-SFT](https://huggingface.co/datasets/openeurollm/OpenEuroLLM-Excellent-SFT), pinned `c4ee55804f256c5f8cc4a6ef3fdecbe4a4ea95e0` | Core general instruction pool. Leonardo's full structural scan found 2,649,634 direct rows of 6,187,716 total; language coverage is **not** established by that scan. | Keep only text-only direct rows with complete assistant answers; render and audit assistant masks and length; review source-level license and quality; deduplicate by canonical prompt and source ancestry. |
| [Dolci-Instruct-SFT-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated) | Translated instruction candidate in `cs de el es fi fr it nl pl ro sv uk`. Important for Finnish and Swedish. | The current card has no substantive README and the viewer includes incomplete conversations. Require a valid final assistant turn, native-language spot checks, source-license review, and cross-language/source-ID deduplication against English Dolci inside Excellent. |
| [EU-Instruct-Synthetic](https://huggingface.co/datasets/openeurollm/EU-Instruct-Synthetic) | 1,497,276 single-turn rows in `cs de el es fr it nl pl pt ro uk`; adds Portuguese. | Sample and score constraint compliance, factuality, idiomaticity, and pattern repetition per language before inclusion. Some previewed Czech appears unnatural; do not use wholesale. Deduplicate against translated Dolci and holdouts. |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | 202,364 train prompt-completion pairs, human annotated; 20 of our target languages: `da de el en eu fi fr ga hu it lt nl pl pt es sq sr sv tr uk`. Especially useful new candidates for `da eu ga hu lt sq sr tr`. | Use train only; measure actual per-language counts and quality. Its card flags uneven contributions and possible annotator bias. Deduplicate against Aya embedded in Aya Collection and existing Excellent/Dolci ancestry. Apache-2.0 per card. |
| [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection) | 513M templated/translated prompt-completion instances; its published language table intersects **35/36** of our target languages (all except `bs`). Includes `ka`, `mt`, `mk`, `lv`, `is`, and `gl`, but most low-resource cells are machine-translated rather than native-authored. | **Gap-filling candidate, not a 513M-row training prescription.** Choose specific source subsets and prompt families; preserve source and language/script metadata; audit original-source terms, translation idiomaticity, answer validity and leakage. Aya Dataset is included inside it, so never ingest both un-deduplicated. Apache-2.0 aggregate card. |
| [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Small human-rated multilingual conversation supplement: 13,854 ready trees/135,174 messages, including 1,283 **Catalan messages** and smaller `eu/hu/da` counts. This provides multi-turn style, not balanced 36-language volume. | Reconstruct paths from `parent_id` and select accepted/high-ranked assistant turns; exclude spam/deleted and check language per turn. `sl` and `gl` appear only in the *all-trees* statistics, not the ready export; do not count those as approved training coverage. Dedup against OASST derivatives in Excellent. Apache-2.0 per card. |
| [LumiOpen/AutoIF-FI](https://huggingface.co/datasets/LumiOpen/AutoIF-FI) | 30K Finnish multi-turn constraint-following examples with generated verifiers and response checks; the linked project register identifies it as a candidate. | Promising **after** exact license/source review: the current HF card labels the license `other`, not Apache-2.0. Keep its 150-example test split out of training; sample verifier validity and native fluency. Not included in the first pilot. |
| [Kakugo low-resource collection](https://huggingface.co/collections/ptrdvn/kakugo-datasets) | Target-language members `kakugo-gle` (~37.8K Irish), `kakugo-glg` (~40.1K Galician), and `kakugo-mlt` (~40.5K Maltese) mix direct-answer and explicit thinking-mode conversations. Individual cards show Apache-2.0. | Separate direct and reasoning rows by actual rendered labels and mode prompts; audit native fluency, answer quality, FineWeb-2/Infinity-Instruct source terms, and prompt-family overlap. Viewer examples include English-language reasoning under Irish prompts; never admit such rows as native reasoning without checks. Do not indiscriminately place all rows in Stage 1. |
| [Tbilisi AI Lab Georgian SFT mix](https://huggingface.co/datasets/tbilisi-ai-lab/kona-sft-mix-2.6M) | 2.61M mixed Georgian/English instruction rows with per-row `source`, including roughly 200K Georgian-domain examples. ODC-BY card. | **Conditional**: extract Georgian-language, direct, non-tool subsets only; source-level licensing, privacy, native QA and decontamination required. The aggregate includes reused Tülu/Orca/SmolTalk and reasoning data, so overlap is substantial. Not included in the first pilot. |
| [EuroBlocks-SFT-Synthetic-1124](https://huggingface.co/datasets/utter-project/EuroBlocks-SFT-Synthetic-1124) | 340K-row, 31-language potential gap-filler with quality and dedup fields. | **Conditional only.** Its card assigns different upstream licenses by subset, including Llama terms. Filter to source-level acceptable subsets, check actual target-language labels, and manually review low-resource output. Do not treat the aggregate as uniformly Apache-2.0. |
| [oellm-eu-defect-repair-sft-v1](https://huggingface.co/datasets/birgermoell/oellm-eu-defect-repair-sft-v1) | Narrow language-repair candidate, not a general instruction backbone. | Source-provenance/attribution, quality, overlap, and per-language benefit checks; cap its supervised-token share. |
| [OpenEuroLLM Model Identity v1.0.0](https://huggingface.co/datasets/birgermoell/openeurollm-model-identity), pinned `9dddd45d932c8bdde12414ea0201ab2d5c6d52ec` | Small, dedicated direct-answer identity curriculum: 1,000 training and 120 validation conversations in `en de es fr it pt pl nl cs ro el sv`. The user-linked repository matches the OpenEuroLLM mirror's published split checksums. CC-BY-4.0, English-authored and machine-translated. | Stage-1 train only; never ingest the 120 validation rows. Verify exact chat-template labels, hold out exact prompt collisions, and native-review translated answers. A published French preview contains `OpenEuroLLL`, so quarantine known text errors rather than trusting the aggregate validation report blindly. Preserve attribution; evaluate identity accuracy and unwanted identity boilerplate on unrelated tasks. |
| New 36-language native/translated direct-answer set | Needed for languages with no validated direct SFT source in the current shortlist. | Start with licensed, diverse seed instructions; use both native prompts and translated prompts/answers; language/script checks, answer correctness, native-speaker review, prompt-family split, and source-level redistribution review. Do not equate machine translation with quality. |

The original OpenEuroLLM translated/synthetic shortlist explicitly covered 13
non-English targets (`cs de el es fi fr it nl pl pt ro sv uk`) plus English.
Aya Dataset adds eight distinct targets (`da eu ga hu lt sq sr tr`); OASST2's
*ready* export adds Catalan (`ca`), though its 1,283 are **messages**, not
1,283 vetted conversations. Kakugo and the Georgian mix add conditional
`mt` and `ka` candidates. Thus 25/36 targets now have an identified direct
candidate before counting Aya Collection. Aya Collection lists 35/36 targets,
but this is **nominal source coverage**, not quality-cleared SFT coverage; `bs`
is absent from its published table. Excellent's actual language distribution
must still be measured before setting any budgets.

The linked team [post-training source register](https://docs.google.com/spreadsheets/d/1hFFDRk_JdjbQCCv3zsemKulSnMyk_fBQIfKPSzSaP9M/edit?gid=1339797209)
is a useful provenance index, **not** a ready-made mixture: it records the
earlier Tülu/EuroBlocks run, a Finnish AutoIF lead, and research ablations in
which adding WildChat/OASST2 gave no clear gain. Treat those as evidence to
test incremental value rather than automatically increasing their weight.
The register also labels the user's EU eval holdouts and ArenaHard-EU as
evaluation-only.

Do **not** use [Sag69/bosnian-dataset](https://huggingface.co/datasets/Sag69/bosnian-dataset)
as a shortcut to the missing `bs` cell: its card lists substantial modern
literary works under an aggregate MIT label. That is not adequate evidence
that the underlying works can be reused for model training/release.

| Target cells | Direct Instruct shortlist | Reasoning shortlist |
| --- | --- | --- |
| `en` (1) | Excellent core | English Dolci/Excellent anchor |
| `cs de el es fi fr it nl pl ro sv uk` (12) | Translated Dolci; EU-Instruct overlaps in ten; Aya in some | Translated Dolci-Think, plus selected Excellent rows |
| `pt` (1) | EU-Instruct and Aya | Only tiny pilot currently identified |
| `da eu ga hu lt sq sr tr` (8) | Human-annotated Aya candidate; Aya Collection also lists them | Tiny pilot only; needs substantial verified data |
| `ca` (1) | OASST2 ready messages and Aya Collection candidate | Tiny pilot only; needs substantial verified data |
| `bg et gl hr is lv mk no sk sl` (10) | Aya Collection candidate; quality/source-subset audit pending | Tiny pilot only; needs substantial verified data |
| `bs` (1) | No Aya Collection row; needs separate sourced or produced data | Tiny pilot only; needs substantial verified data |
| `ka` (1) | Aya Collection translated candidate; Georgian SFT mix conditional | Gap; absent even from the tiny pilot |
| `mt` (1) | Aya Collection translated candidate; Kakugo direct subset conditional | Kakugo reasoning-mode subset conditional |

“Coverage” here means an identifiable candidate source, not a quality pass or
adequate training volume. In particular, a few dozen translated math problems
are not enough for a general reasoning model in a language.

If the Stage-1 release must support function calling, use a separately
rendered non-reasoning tool-use substage and tool benchmark. The current
text-only template deliberately rejects tool trajectories.

## Stage 2 — genuine Reasoning SFT

| Candidate | Role and observed coverage | Admission gate |
| --- | --- | --- |
| Excellent-SFT reasoning rows | Large mixed-source pool: 3,526,040 structurally valid reasoning rows in the full Leonardo scan. Its Nemotron `de/es/fr/it/ja` slices contribute about 3.10M reasoning rows; `ja` is outside our allowlist. | Filter all target-language rows by nonempty closed `<think>` and nonempty answer; verify answer correctness where possible, full render, length, language of prompt/trace/answer, source terms, and duplicate ancestry. Do not let long traces silently dominate supervised tokens. |
| [Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B) | English anchor across math, code, precise instruction following, and general reasoning; 2.25M rows nominally. | Source-aware selection, valid traces, verifier checks for math/code, length cap, ODC-BY attribution/terms, and dedup against Excellent/OpenThoughts3. Avoid simply adding all 2.25M rows. |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | English math seed: about 220K problems across `default`/`extended`, with 2–4 DeepSeek-R1 traces per problem and correctness metadata; `default` has about 94K problems. Apache-2.0 per card. | Use only a chosen verified trace per train problem (or explicit capped variants), filter incomplete/incorrect generations, inspect judge-only cases, and keep source-family/eval holdouts. Good candidate to translate **problems and accepted traces** for sparse target languages, followed by native-language and mathematical re-verification; not general reasoning or 36-language coverage on its own. |
| [Kakugo collection](https://huggingface.co/collections/ptrdvn/kakugo-datasets) reasoning-mode subsets | Potential small Irish (`ga`), Galician (`gl`), and Maltese (`mt`) seeds with explicit `<think>` traces and mode prompts. | Partition by source mode and nonempty balanced tags; verify that **both** the trace and final answer are actually in the target language and factually correct. The Irish viewer shows some English traces, so raw `<think>` presence is insufficient. Exclude translated direct-only rows from reasoning. Check upstream teacher/source terms and dedup before any non-pilot use. |
| [Dolci-Think-SFT-translated](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-translated) | Current 12-language translated candidate: `cs de el es fi fr it nl pl ro sv uk`; card reports 1,263,942 rows with `filter_pass=true` across these languages. | Require `filter_pass=true` **and** our own format, token-length, math/code, language-consistency and native-speaker checks. Rejection is often truncation. The translated card says Apache-2.0 while the English Dolci source says ODC-BY; resolve terms before public release. |
| [OpenEuroLLM multilingual reasoning-traces pilot](https://huggingface.co/datasets/openeurollm/reasoning-traces-multilingual) | 3,425 accepted translations from only 99 math problems. It has 37 non-English language codes, including three outside our target set (`cy lb ru`), and **omits Georgian `ka`**. | Use primarily for translation-pipeline QA or controlled pilot evaluation, not as a production-scale curriculum. Prevent train/test leakage by grouping all translations with their original source problem. Its own card says native-speaker and solver validation remain incomplete. |
| Verified new multilingual math/code traces | Fill the 23-language gap beyond English + translated Dolci, especially `ka`. The user's [math](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) and [code](https://huggingface.co/datasets/birgermoell/oellm-code-rlvr) RLVR sets can supply **problems and verifiers**, not automatically correct reasoning traces. | Generate traces only for training-split problems, execute/verify final answers, check intermediate invariants when feasible, preserve problem-family splits, translate or generate in target languages, and review native quality. Compare target-language traces against English-trace/localized-answer alternatives. |
| [medqa-reasoning-traces](https://huggingface.co/datasets/birgermoell/medqa-reasoning-traces) | English clinical-specialist candidate, not general or multilingual coverage. | Default exclude from general reasoning run. If a medical branch is desired, use **train split only**, `correct=true`, `finish_reason=stop`, and review the research-use question license and generated-output terms; never leak validation/test questions. |

The named large-scale reasoning sources explicitly cover English plus 12
non-English targets. The 23 remaining target languages require validated new
reasoning data; the 3.4K-row pilot is evidence a translation path is possible,
not evidence that those languages have sufficient training coverage.

For evaluation rather than training, reserve
[ArenaHard-EU-v0](https://huggingface.co/datasets/openeurollm/ArenaHard-EU-v0)
and its prompt families as holdouts. Its card lists all **35 non-English**
OpenEuroLLM target languages, including Bosnian; do not turn a benchmark into
training examples. Verify the actual downloadable row counts/configurations
before claiming per-language benchmark coverage—the current card's claimed
500+ prompts/language and the viewer's 360 total rows do not agree.

## Selection and mixing rules shared by both stages

1. Freeze a manifest per source: repo/revision, subset, upstream generator,
   source license, language/script, row IDs, prompt-family/source-problem IDs,
   data quality decision, rendered tokens, and supervised tokens. Keep a
   rejection report. Do not publish a row-count-only mixture.
2. Split by canonical prompt/problem family *before* translating, sampling,
   or constructing variants; decontaminate against fixed 36-language eval and
   long-context holdouts. This also prevents English originals appearing in
   train while translations of the same problem appear in test.
3. Filter both stages through the exact saved tokenizer/template. Stage 1 has
   **zero** assistant `<think>` labels; Stage 2 has nonempty, balanced think
   spans plus a substantial direct-answer replay slice. Neither stage may
   synthesize empty tags or silently truncate a final answer.
4. Sample by **supervised tokens** with per-source and per-language floors and
   caps, rather than equal row counts. Set actual shares only after quality
   audits and a tiny packed-throughput pilot. Retain a strong English anchor
   but do not let it erase underserved target languages.
5. For Stage 2, train a clear mode policy: default direct answer, requested
   reasoning mode emits nonempty `<think>...</think>` and a final answer.
   Test the mode switch separately in every target language. Compare whether
   localized traces improve correctness versus English traces with localized
   answers; do not assume translation wins.
6. Evaluate answer correctness, instruction following, fluency, reasoning
   language consistency, refusal/safety, 4K/32K/128K/256K retrieval, and
   unwanted/empty-think rates by language. Native review is required for
   low-resource and script-variant groups. No claim of 36-language quality
   follows merely from having at least one training example per language.

## Full-volume Stage-1 intent and current staging

### Reference-style fast path for an internal Stage-1 run

The published `OpenEuroLLM-Prelude-SFT-Excellent-Instructions` run used the
curated `OpenEuroLLM-Excellent-SFT` corpus for two epochs at 32,768 tokens
with assistant-only loss. We can reuse that **data lineage** on the staged
256K anneal300b base without waiting for every optional data addition or a
full new source-by-source ranking. For this two-stage experiment, however,
Stage 1 must select the corpus's **direct-answer** text rows (2,649,634
structural candidates), rather than train on its 3,526,040 reasoning rows
and pretend the result is a non-reasoning Instruct checkpoint. Keep the
reasoning rows for Stage 2. The staged translated and synthetic EU data can
be a measured multilingual augmentation or a follow-up ablation, not a reason
to hold up the reference-style baseline.

The previous reference model's published `chat_template.jinja` wraps a final
assistant answer in `<think>...</think>` even when its `reasoning_content` is
empty. Its `enable_thinking=false` generation prompt can also emit an empty
block. **Do not use that template for this run.** Use the checked
`templates/oellm_qwen3_no_empty_assistant_mask.jinja` with the packaged
ChatML tokenizer; reject any direct row whose rendered assistant labels
contain `<think>` or `</think>`, and fail the job if any empty or unbalanced
think span appears. The CPU/TRL preflights and eight-step four-A100 canary
already passed on real data, but they are not a full-corpus label audit.

The minimum remaining preparation for a **real internal baseline** is
mechanical: stream the pinned corpus into a deterministic train/validation
split of direct, text-only, fully rendered rows; reject overlength/truncated
answers; record source counts, hashes and supervised-token totals; then run
a short checkpointed slice with fixed evaluation before allocating the full
budget. The model-identity candidate can be added through a measured small
replay/tail dose, not by repeating 782 distinct prompts dozens of times.
Review source-level terms in parallel under project policy: public release
needs a separate decision, and any source terms that restrict internal
training still apply. This need not turn optional data-ranking work into a
prerequisite for the reference-style baseline. Do not call
the run ready to submit until the actual full-size Stage-1 JSONL/Parquet and
manifest exist; today only the small engineering pilot is materialized.

The tiny 8-step run is an engineering canary only. Its pinned CPU dataset
build finished on Leonardo as job `61543719`: 695 Instruct train rows, 102
internal validation rows, and 29 separate nonempty-reasoning candidates
passed the exact render/mask/4K checks. The small Reasoning yield (67 sampled
rows exceeded 4K) is another reason to curate that stage at a different
sequence-length budget. These counts are **not** the proposed training volume.
CPU preflight job `61544751` then passed both stage-specific render audits
and TRL's actual SFTTrainer/collator checks. The eight-step, four-A100
Instruct-only GPU canary, job `61544957`, **completed successfully** in
6m39s (internal eval loss 1.323, not a quality comparison) and saved a
36.4-GB FP32 `model.safetensors` export. That export must be type-checked
and converted to BF16 before any promotion. The result remains an
engineering check, not a production checkpoint or a data-quality verdict.
The intended Instruct SFT
is a **large, multi-million-example candidate mix** built from the eligible
Excellent direct-answer pool (2,649,634 *before* source/quality filtering),
human-authored Aya (202,362 rows in the pinned train Parquet before language
filtering), and substantial selected portions
of translated Dolci and EU-Instruct-Synthetic. The full staged structural
profile found **5,774,712 direct rows** out of 5,937,525 translated Dolci rows
(162,813 malformed/unsupported) and **1,497,276 direct rows** in EU synthetic.
The four sources together therefore contain roughly **10.1 million structural
direct-answer candidate rows**, *not* 10.1 million admitted examples or
independent prompts. The admitted count is unknown until source, quality,
rendered-token, and duplicate audits. Do not
inflate the total by simply summing source-card row counts: English Dolci is
already in Excellent and translated variants share prompt ancestry. The
staged translated Dolci Parquet preserves the upstream `id` (for example an
OpenThoughts3 source ID), so split and deduplicate its translations by that
source-family ID across languages and against English Dolci, not only by
same-language prompt text.

The two additions are pinned at `63202c34b23a4f155658a9e157e988902fb62279`
(Dolci translated) and `c13be5d71144feb1007708aae2a8c2d823f8e59a`
(EU synthetic). Leonardo staging job `61543133` and full structural-profile
job `61543252` completed successfully. The profile and reproducible
per-language review samples are at
`/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/reports/instruct-expansion-structural-profile.json`.
They are not automatically admitted to a GPU run. This expands the research
beyond the first eight-step pilot without claiming unverified language
quality or source rights.

The first deterministic review samples show why admission cannot be based on
structural validity alone: a Czech EU-synthetic prompt requests one JSON
object, but the answer wraps it in a Markdown code fence; a sampled Swedish
translated Dolci math prompt receives a long step-by-step final answer despite
not requesting a derivation. These are examples to classify at scale (format
compliance and unsolicited worked solutions), not evidence that an entire
language subset is bad. Preserve these samples and run native-language and
constraint checks before deciding token weights.

Before the production run, make an exact deduplicated, rendered, token-counted
manifest with train/validation splits and source-language quotas. Keep a
large English anchor for broad capabilities, but reserve meaningful
supervised-token shares for every quality-cleared target language; do not
use equal row counts as a proxy. Record exclusions and compare performance
as data volume increases. The 36-language target does not justify admitting
poor low-resource translations; weak cells need new validated data rather
than repeated low-quality rows. Keep long-context and tool-use curricula
separate from this first direct-answer text stage.

### Identity addition and dose control

The pinned identity release has only 1,000 training conversations. Appending it
once to a multi-million-row mix would give it negligible exposure; repeating it
dozens of times to force a fixed percentage would overfit a narrow set of
synthetic phrasings. First run the CPU-only staging/render audit in
`scripts/stage_oellm_model_identity.py`, then use its **train output only** in
a tiny identity-dose ablation with general-instruction replay. Test one, two,
and at most three effective exposures per identity row before considering new,
independently authored examples. Choose the production dose by held-out
identity accuracy *and* unchanged general-task performance, not by a guessed
row percentage. The identity validation split remains separate from the
internal general Instruct validation and external release tests. The Stage-2
Reasoning run need not copy identity examples as fabricated thinking traces;
if identity regresses, replay the same direct-answer rows in its direct slice.

Leonardo CPU staging job `61549601` audited every source row with the saved
tokenizer and chat template. The current candidate outputs contain **782
train** rows (56,666 supervised tokens) and **119 validation** rows (9,012
supervised tokens), with no exact canonical-prompt overlap between them. Six
train rows and one validation row containing the observed project-name typo
were quarantined; another 212 train rows repeated an existing user prompt
with a different answer phrasing and were deduplicated deterministically.
These are audited *candidates*, not evidence of native-language quality or
membership in the still-unbuilt production mixture. The manifest and outputs
are under `/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/data/model-identity-v1`.
A no-GPU TRL SFTTrainer/collator check on a real multi-turn identity row passed
as Leonardo job `61549699`: all 62 supervised tokens were assistant text,
without an injected `<think>` span. No GPU training was launched for this
addition.

### Polish DynaWord: separate language-adaptation candidate, not Instruct SFT

[SlayerLab/polish-dynaword-mix-extended-500M](https://huggingface.co/datasets/SlayerLab/polish-dynaword-mix-extended-500M),
observed at revision `c547dea0fb2bf52795774acd2ecba7986e7ba9c2`,
is a useful **raw Polish language-modeling** lead, but its 500M denotes
approximately 500 million 32K-BPE tokens, not instruction examples. Its card
reports 14,412,284 train and 155,182 validation rows in a single `text`
column; the source mix combines documents with individual lines. It is
CC-BY-SA-4.0 with upstream attribution/share-alike obligations. Do **not**
convert arbitrary text rows into assistant answers or add the corpus to
Stage-1/Stage-2 chat SFT. Candidate uses are a separately evaluated Polish
continued-pretraining/language-adaptation run, or carefully constructed
grounded Polish tasks from specific cleared sources. Before either, pin the
revision, resolve source-level reuse and downstream share-alike implications,
sample documents for extraction quality and residual privacy issues, dedup
against existing base/SFT data, and compare Polish improvement with forgetting
on other target languages. Its published PII check covers emails and phone
numbers; that alone is not a complete privacy or source-rights clearance.

## Immediate checks

- Complete the sample-only two-track pilot manifest and its tiny GPU canary
  to verify the training path and the absence of empty `<think>` labels.
- Inspect the full structural profiles and native-language samples from the
  newly staged multilingual sources. Make a source-level terms decision,
  especially for Excellent's mixed-license subcorpora and Dolci ancestry.
- Materialize the large Stage-1 train/validation manifest only after
  prompt-family dedup, decontamination, full render/token audit, and a
  supervised-token budget; then submit a separately budgeted production run.
- Continue Stage-2 Reasoning source curation independently. Do not train it
  on the direct-only Stage-1 manifest or on unverified translated traces.
