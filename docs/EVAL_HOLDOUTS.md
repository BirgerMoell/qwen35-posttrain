# EU Eval Holdouts

The local dataset artifact is:

`data/eval_holdouts/oellm-eu-eval-holdouts-v1/`

It is a public synthetic canary benchmark for OpenEuroLLM European-language
post-training. The first release is intentionally safe to publish: all source
documents are synthetic, deterministic, and tagged with denylist metadata. Private
native-document holdouts should use the same schema but should not be uploaded
publicly unless the source text, licenses, and hidden-answer policy allow it.

## Contents

The generated v1 pack contains:

- 7,600 rows
- 38 languages
- 200 examples per language
- 10 buckets
- `dev` split: 1,520 rows
- `test_public` split: 6,080 rows

Buckets:

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

## Build

```bash
python3 scripts/build_eu_eval_holdouts.py --examples-per-language 200
```

## Validate

```bash
python3 scripts/validate_eu_eval_holdouts.py
```

The validator checks required fields, unique IDs, unique contamination
signatures, no `train` split, per-language coverage, per-bucket coverage, and
long-context materialization metadata.

## Long Context

Long-context rows store a deterministic materialization plan instead of enormous
contexts. Materialize them before running generation:

```bash
python3 scripts/materialize_long_context_holdouts.py \
  --input data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/dev.jsonl \
  --output /tmp/oellm-eu-dev-materialized-longctx.jsonl \
  --only-long-context
```

For quick smoke tests, first create a tiny JSONL containing one long-context row
and materialize that file. Full dev/test materialization can be large.

## Scoring

Use each row's `scoring` field:

| Scoring | Method |
| --- | --- |
| `json_exact_fields` | parse JSON and compare required fields |
| `must_contain_all` | require all expected aliases/facts |
| `long_context_exact_or_insufficient` | exact code or insufficient-info answer |
| `rubric_with_required_points` | judge/rubric |
| `math_exact_integer` | exact integer |
| `tool_call_sequence_exact` | exact tool names and required args |
| `preference_exact_letter` | exact A/B |
| `safety_rubric` | judge/rubric |
| `insufficient_information` | refuse to invent missing fact |

## Contamination Control

Every row includes:

- `source_doc_id`
- `source_url`
- `template_id`
- `canary`
- `contamination_signature`
- `denylist`

Training data generation should denylist these values and template families.

## Known Limitations

This is a synthetic public benchmark. It is not a replacement for private
native-authored holdouts with fluent-speaker review. The field
`review_status=synthetic_public_v1_needs_native_review` is deliberate. Treat
native style and translationese results as a regression signal, not as a final
leaderboard score, until native reviewers have approved those rows.

