---
license: other
task_categories:
- text-generation
- question-answering
- multiple-choice
pretty_name: OpenEuroLLM European Exam MCQ v1
tags:
- openeurollm
- grpo
- dpo
- rlvr
- multiple-choice
- exams
- european-languages
- license-filterable
---

# oellm-eu-exam-mcq-v1

Real-source multiple-choice data for European-language GRPO/RLVR and DPO
training.

This build contains:

- GRPO/RLVR rows: 381597
- DPO pairs: 1141467
- Languages: bg, ca, cs, da, de, el, en, es, et, eu, fi, fr, hr, hu, hy, is, it, ka, lt, lv, mk, mt, nb, nl, pl, pt, ro, ru, sk, sl, sq, sr, sv, tr, uk

## License Filtering

This is a mixed-license dataset. Filter rows before training or redistribution by
`license_id`, `license_category`, `source_id`, and `redistribution_status`.

Licenses in this build:

- `apache-2.0`: 243558 rows
- `cc-by-4.0`: 1800 rows
- `cc-by-nc-sa-2.0`: 52 rows
- `cc-by-sa-4.0`: 50558 rows
- `mit`: 70209 rows
- `unknown`: 15420 rows

Sources:

- `belebele`: 31500 rows, `cc-by-sa-4.0`, `redistributable_sharealike`
- `exams_qa`: 19058 rows, `cc-by-sa-4.0`, `redistributable_sharealike`
- `global_mmlu`: 243558 rows, `apache-2.0`, `redistributable_declared_license`
- `hogskoleprovet_ord`: 145 rows, `unknown`, `official_public_unknown_redistribution`
- `llmzszl`: 14269 rows, `unknown`, `unknown_missing_license`
- `mmmlu`: 70209 rows, `mit`, `redistributable_declared_license`
- `polish_matura_dokato`: 52 rows, `cc-by-nc-sa-2.0`, `redistributable_noncommercial_sharealike`
- `swedish_medical_exams_hf`: 1006 rows, `unknown`, `official_public_unknown_redistribution`
- `xcopa`: 1800 rows, `cc-by-4.0`, `redistributable_attribution`

## Files

- `grpo/train.jsonl`, `grpo/validation.jsonl`, `grpo/test.jsonl`
- `dpo/train.jsonl`, `dpo/validation.jsonl`, `dpo/test.jsonl`
- `manifest.json`
- `source_registry.json`

GRPO rows use `reward_type=mcq_letter_exact`: reward a response whose first
answer letter matches `answer`.

DPO rows are generated as correct-letter responses preferred over each incorrect
letter for the same prompt.

## Source Registry

See `source_registry.json` for official archives and source-level license notes.
Some official exam archives are link-only until redistribution rights are
cleared.
