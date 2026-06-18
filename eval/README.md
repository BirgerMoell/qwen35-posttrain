# eval/

Run after **every** stage to track progress and catch regressions (especially long-context).

Suite:
- **General:** MMLU, IFEval, MT-Bench-style
- **Reasoning:** GSM8K, MATH, (code) HumanEval / MBPP
- **Long context (must not regress):** RULER (and OneRuler if multilingual focus is on)
- **Safety:** refusal/harmlessness probes

Tooling: `lm-evaluation-harness` + vLLM (ROCm) backend for fast generation. Baseline the
**Qwen3.5 base** first, then diff each post-training stage against it.
