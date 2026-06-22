---
language:
- bg
- bs
- ca
- cs
- cy
- da
- de
- el
- en
- es
- et
- eu
- fi
- fr
- ga
- gl
- hr
- hu
- is
- it
- lb
- lt
- lv
- mk
- mt
- nl
- no
- pl
- pt
- ro
- ru
- sk
- sl
- sq
- sr
- sv
- tr
- uk
license: cc0-1.0
task_categories:
- text-generation
- question-answering
- summarization
- text2text-generation
pretty_name: OpenEuroLLM EU Eval Holdouts v1
tags:
- openeurollm
- multilingual
- european-languages
- evaluation
- long-context
- tool-calling
- synthetic
---

# oellm-eu-eval-holdouts-v1

Public synthetic canary eval holdouts for European-language post-training.

This release contains 200 examples for each of 38 languages
(7600 rows total), split into `dev` and `test_public`.
It covers ten buckets:

- `instruction_following`
- `grounded_qa`
- `long_context_retrieval`
- `summarization`
- `reasoning_math`
- `tool_calling`
- `translationese_preference`
- `civic_safety`
- `no_answer`
- `locale_formatting`

All contexts are synthetic and canary-tagged. The point is to provide a
contamination-safe public regression benchmark with the same schema that private
native-document holdouts can use later.

## Files

- `data/dev.jsonl`
- `data/test_public.jsonl`
- `data/all.jsonl`
- `metadata/summary.json`

## Scoring

Use the `scoring` and `rubric` fields per row. Some tasks are exact match
(`math_exact_integer`, `json_exact_fields`, `tool_call_sequence_exact`); others
are rubric/judge tasks (`safety_rubric`, `rubric_with_required_points`).

Long-context rows contain a deterministic materialization plan in
`metadata.materialization` instead of storing hundreds of thousands of tokens per
row. Evaluators should expand those contexts before inference.

## Contamination control

Every row includes:

- `source_doc_id`
- `template_id`
- `canary`
- `contamination_signature`
- `denylist`

Training data builders should denylist those values.

## Limitations

This is a public synthetic benchmark, not a substitute for private native-human
reviewed holdouts. It is useful for fast regression gates, format following,
language routing, long-context retrieval mechanics, and tool-call validation.
Native style and translationese buckets should be reviewed by fluent speakers
before being treated as a final leaderboard.
