---
license: cc-by-sa-4.0
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
---

# oellm-eu-exam-mcq-v1

Real-source multiple-choice exam data for European-language GRPO/RLVR and DPO
training.

This build contains:

- GRPO/RLVR rows: 19058
- DPO pairs: 56298
- Languages: bg, de, es, fr, hr, hu, it, lt, pl, pt, sq, sr, tr

The default redistributable source is EXAMS QA, licensed CC-BY-SA-4.0. Optional
local rows from Swedish Medical Benchmark are marked `local_review_required` and
should not be uploaded as a release dataset without source-term review.

## Files

- `grpo/train.jsonl`, `grpo/validation.jsonl`, `grpo/test.jsonl`
- `dpo/train.jsonl`, `dpo/validation.jsonl`, `dpo/test.jsonl`
- `manifest.json`

GRPO rows use `reward_type=mcq_letter_exact`: reward a response whose first
answer letter matches `answer`.

DPO rows are generated as correct-letter responses preferred over each incorrect
letter for the same prompt.

## Source Registry

See `source_registry.json` in the repository for official exam archives that
should be discovered and parsed locally, including Högskoleprovet, CKE, and
CERMAT. Some official archives are link-only until redistribution rights are
cleared.
