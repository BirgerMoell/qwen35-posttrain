# stages/

Per-stage launchers/code. Each stage reads a `configs/*.yaml`, runs via `slurm/*.sbatch`,
and writes a checkpoint the next stage consumes. Keep saves **weights-only** between stages
(the next stage reloads weights, not optimizer state).

- `sft/`   — TRL SFTTrainer
- `dpo/`   — TRL DPOTrainer
- `grpo/`  — veRL (or TRL+vLLM) RLVR/GRPO with verifiable rewards
- `tools/` — (optional) tool-use / agentic SFT+RL

Order: **0 smoke (2B) → SFT → DPO → GRPO → (tools) → eval/merge.** See `docs/PLAN.md`.
