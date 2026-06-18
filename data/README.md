# data/

Data-mix specs per stage. Use **permissively-licensed** sources (open artifact). Stage all
data to LUMI scratch before submitting (compute nodes have no internet).

- **SFT:** general instruction + multi-turn chat + CoT/reasoning + tool-use traces +
  **long-context instruction data** (preserve 262K). Candidates: Tülu 3 SFT mix, OpenHermes,
  reasoning distillations. (+ multilingual/European sets if that focus is enabled.)
- **DPO:** preference pairs — UltraFeedback-style + on-policy pairs.
- **RLVR/GRPO:** prompts with **verifiers** — math (GSM8K/MATH/Numina), code (unit tests), logic.

Keep a frozen held-out eval split separate from training mixes.
