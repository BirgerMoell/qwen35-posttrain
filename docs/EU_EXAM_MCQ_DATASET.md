# European Exam MCQ Dataset

Dataset artifact:

`data/exam_mcq/oellm-eu-exam-mcq-v1/`

Hugging Face release:

`https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1`

This is a real-source multiple-choice dataset for European-language GRPO/RLVR
and DPO tuning. It now includes national exams, medical/licensing exams,
academic-exam benchmarks, and reading/causal reasoning MCQ sources.

## Current Build

Version: `v0.5.0`

- 582,983 GRPO/RLVR rows
- 1,936,764 DPO preference pairs
- 35 language codes
- 28 sources
- mixed licenses, filterable per row

Files:

- `grpo/train.jsonl`
- `grpo/validation.jsonl`
- `grpo/test.jsonl`
- `dpo/train.jsonl`
- `dpo/validation.jsonl`
- `dpo/test.jsonl`
- `manifest.json`
- `source_registry.json`

## Real Exam Sources

The real national/official exam layer currently includes:

- `exams_qa`: 19,058 multilingual school-exam rows, `cc-by-sa-4.0`
- `hogskoleprovet_ord`: 145 Swedish Högskoleprovet ORD rows parsed from official
  Studera/UHR PDFs and answer keys, `unknown`
- `llmzszl`: 14,269 Polish national/professional exam rows, `unknown`
- `polish_pes_medical`: 178,392 Polish specialist medical exam rows from
  PES 2007-2024, `unknown`
- `polish_lek_medical_pl`: 4,217 Polish LEK medical licensing rows, `unknown`
- `polish_ldek_medical_pl`: 4,238 Polish LDEK dental licensing rows, `unknown`
- `polish_lek_medical_en`: 2,725 English LEK medical licensing rows, `unknown`
- `polish_ldek_medical_en`: 2,726 English LDEK dental licensing rows, `unknown`
- `danish_citizenship_test`: 605 Danish citizenship/permanent-residence test
  rows, `unknown`
- `estonian_language_exams`: 462 Estonian language-proficiency rows,
  `cc-unspecified`
- `estonian_school_tests`: 476 Estonian school-test rows, `cc-unspecified`
- `bulgarian_culture_exams`: 2,729 Bulgarian language/literature/geography/history
  rows parsed from custom-delimited raw files, `unknown`
- `albanian_medical_systems`: 400 Albanian medical systems rows,
  custom open license needing review
- `albanian_medical_chemistry`: 270 Albanian medical chemistry rows,
  custom open license needing review
- `czech_bio_exam`: 50 Czech biology university-entry rows,
  `cc-by-nc-sa-2.0`
- `czech_lit_exam`: 31 Czech grammar/literature rows, `cc-by-nc-sa-2.0`
- `italian_bio_quiz`: 891 Italian biology university quiz rows,
  `cc-by-nc-sa-2.0`
- `swedish_medical_exams_hf`: 1,006 Swedish medical licensing rows, `unknown`
- `polish_matura_dokato`: 52 Polish matura rows, `cc-by-nc-sa-2.0`
- `slovak_mathbio_dokato`: 131 Slovak math/biology university-entry rows,
  `cc-by-nc-sa-2.0`
- `slovak_financial_exam`: 1,086 Slovak financial certification rows,
  `cc-by-sa-4.0`
- `basque_public_exams`: 719 Basque public-service legal exam rows,
  custom open license needing review
- `catalan_public_exams`: 772 Catalan public-service legal exam rows,
  Open Information Use License - Catalonia, needing review
- `spanish_public_exams`: 466 Spanish public-service legal exam rows,
  AGPL/GPL metadata, needing review

The broader MCQ layer adds Global-MMLU, MMMLU, Belebele, and XCOPA for
exam-style academic knowledge, reading comprehension, and causal reasoning.

## License Filtering

Every GRPO and DPO row includes:

- `source_id`
- `source_url`
- `source_license`
- `license_id`
- `license_category`
- `license_filter_tags`
- `redistribution_status`

Current license counts:

- `apache-2.0`: 243,558 rows
- `mit`: 70,209 rows
- `cc-by-sa-4.0`: 51,644 rows
- `cc-by-4.0`: 1,800 rows
- `cc-by-nc-sa-2.0`: 1,155 rows
- `cc-unspecified`: 938 rows
- `open-license`: 1,389 rows
- `open-information-use-license-catalonia`: 772 rows
- `agpl-gpl`: 466 rows
- `unknown`: 211,052 rows

Examples:

```bash
python3 scripts/build_exam_mcq_dataset.py \
  --sources exams_qa,global_mmlu,mmmlu \
  --license-allowlist apache-2.0,mit,cc-by-sa-4.0
```

```bash
python3 scripts/build_exam_mcq_dataset.py \
  --redistribution-status-allowlist redistributable_declared_license,redistributable_sharealike
```

## Row Design

GRPO rows use deterministic reward:

```text
reward_type = mcq_letter_exact
```

DPO rows are generated from the same questions:

```text
chosen = correct answer letter
rejected = one incorrect answer letter
preference_type = mcq_correct_over_incorrect
```

This makes the dataset usable for both RLVR/GRPO and preference training.

## Build

Clone EXAMS first:

```bash
git clone https://github.com/mhardalov/exams-qa /private/tmp/exams-qa
```

Then build all default sources:

```bash
HF_HOME=/private/tmp/hf-cache \
HF_DATASETS_CACHE=/private/tmp/hf-cache/datasets \
python3 scripts/build_exam_mcq_dataset.py --exams-repo /private/tmp/exams-qa

python3 scripts/validate_exam_mcq_dataset.py
```

Optional local Swedish medical rows:

```bash
python3 scripts/build_exam_mcq_dataset.py \
  --exams-repo /private/tmp/exams-qa \
  --include-swedish-medical
```

Rows from the local Swedish Medical Benchmark are marked `local_review_required`.

## Högskoleprovet

The Högskoleprovet manifest is:

`data/exam_mcq/oellm-eu-exam-mcq-v1/source_manifests/hogskoleprovet_sources.json`

It records:

- 28 official Studera/UHR exam pages
- 233 official PDF links

The builder currently parses only the high-confidence `ORD` vocabulary subtest
from verbal PDFs with matching official answer keys. Other Högskoleprovet
subtests remain in the manifest until parser quality and reuse terms are cleared.

Refresh:

```bash
python3 scripts/discover_hogskoleprovet_sources.py
```

## Next Official Sources

High-priority parser targets:

- Spanish MIR/FSE medical residency exams
- Italian MUR medicine admission tests
- French baccalauréat annales
- Czech CERMAT maturita
- Latvian VISC/VIAA state exam archives
- Lithuanian NŠA brandos exam archives
- WJEC/CBAC Welsh and English past papers
- Maltese MATSEC past papers
- Slovenian RIC matura archives
- Croatian NCVVO državna matura archives
- German Einbürgerungstest catalogue once official answer keys are joined
- Dutch Examenblad
- Norwegian Udir exam archives
- Finnish ylioppilaskoe
- Hungarian érettségi
- Romanian bacalaureat
- national bar/law/state-service exams where official MCQ PDFs and answer keys
  are public

For sources with unclear rights, publish rows only with `license_id=unknown` and
source-level provenance, or publish manifests/hashes until legal review clears
redistribution.
