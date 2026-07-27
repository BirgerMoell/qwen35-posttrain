#!/bin/bash
# EU holdouts on the two defect-repair -final models, to make the republish decision evidence-based
# (esp. the defect languages: is/ga/mt/et + da/hr/hu/lt/lv/sk/sl loops + bg). test_public split.
set -uo pipefail
O=/scratch/project_465002530/users/bmoell/qwen35-posttrain/output
OUT=eval/results/holdouts_finals_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
echo "== holdouts (finals) -> $OUT =="
for pair in "2b-final:$O/qwen35-2b-eu-sft-v3-final" "4b-final:$O/qwen35-4b-eu-sft-v3-final"; do
  tag=${pair%%:*}; path=${pair#*:}
  echo "== EU holdouts: $tag =="
  python scripts/eval_eu_holdouts.py --model "$path" \
    --data data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/test_public.jsonl \
    --out "$OUT/holdouts_$tag.json" --batch-size 32 || echo "!! $tag FAILED"
done
echo "== holdouts (finals) DONE -> $OUT =="; ls -la "$OUT"
