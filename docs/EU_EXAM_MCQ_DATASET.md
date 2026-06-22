# European Exam MCQ Dataset

Dataset artifact:

`data/exam_mcq/oellm-eu-exam-mcq-v1/`

Hugging Face release:

`https://huggingface.co/datasets/birgermoell/oellm-eu-exam-mcq-v1`

This is the first real-source multiple-choice exam dataset for European-language
GRPO/RLVR and DPO tuning.

## Current Build

The current redistributable build uses EXAMS QA only:

- 19,058 GRPO/RLVR rows
- 56,298 DPO preference pairs
- 13 European/OpenEuroLLM target languages
- source license: CC-BY-SA-4.0

Languages:

`bg, de, es, fr, hr, hu, it, lt, pl, pt, sq, sr, tr`

Files:

- `grpo/train.jsonl`
- `grpo/validation.jsonl`
- `grpo/test.jsonl`
- `dpo/train.jsonl`
- `dpo/validation.jsonl`
- `dpo/test.jsonl`
- `manifest.json`

## Row Design

GRPO rows use deterministic reward:

```text
reward_type = mcq_letter_exact
```

The model is prompted to return one answer letter. The verifier should reward
only the correct first letter.

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

Then build:

```bash
python3 scripts/build_exam_mcq_dataset.py --exams-repo /private/tmp/exams-qa
python3 scripts/validate_exam_mcq_dataset.py
```

Optional local Swedish medical rows:

```bash
python3 scripts/build_exam_mcq_dataset.py \
  --exams-repo /private/tmp/exams-qa \
  --include-swedish-medical
```

Rows from Swedish Medical Benchmark are marked `local_review_required` and should
not be uploaded as a release dataset before source-term review.

## Source Registry

The source registry is:

`data/exam_mcq/source_registry.json`

It separates:

- `redistributable_sharealike`: can be included in the public dataset
- `link_only_review_required`: official public exam archives, but raw text is not
  redistributed until reuse terms are cleared
- `needs_live_verification`: high-value targets for crawler work

Current verified/link-only official sources include:

- Högskoleprovet archive from Studera/UHR
- Polish CKE matura arkusze
- Czech CERMAT maturita tests

## Högskoleprovet

The Högskoleprovet manifest is:

`data/exam_mcq/oellm-eu-exam-mcq-v1/source_manifests/hogskoleprovet_sources.json`

It currently records:

- 28 official Studera/UHR exam pages
- 233 official PDF links

It intentionally does not download or redistribute raw exam text. Studera says
previous exam booklets and answer keys are free to download, but the latest exam
page also says UHR only has permission to show ELF English texts for one week
after the exam. Keep Högskoleprovet as link-only until reuse is cleared.

Refresh:

```bash
python3 scripts/discover_hogskoleprovet_sources.py
```

## Next Sources To Convert

High-priority targets:

- Swedish Högskoleprovet PDF parser, if redistribution/use terms are cleared
- Spanish MIR/FSE official medical residency exams
- Polish medical LEK/LDEK/PES sources and published benchmark
- Italian MUR medicine admission tests
- CKE/CERMAT official school-leaving exam archives
- Finnish, Dutch, Hungarian, Romanian, Croatian, Slovenian, Lithuanian, Latvian,
  Estonian, Greek, Bulgarian official exam archives

For sources with unclear rights, publish only manifests and hashes. Build local
training rows on LUMI scratch after approval.
