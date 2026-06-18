# Post-Training Data Sources (curated)

We follow the **Dolci** recipe (AllenAI's open OLMo-3 / Tülu-3 post-training suite:
SFT → DPO → RLVR). **OpenEuroLLM has already decontaminated + translated** the SFT/DPO
parts (multilingual, EU-focused), so we prefer the `openeurollm/*` versions and pull RLVR +
a few gaps directly from `allenai/*`. All data is **staged to LUMI scratch first**
(compute nodes have no internet) — see `scripts/stage_data_lumi.sh`.

Priority key: **P1** = backbone, add first · **P2** = needed for GRPO/tools · **P3** = breadth/optional.

## Stage 0b — Reasoning SFT  *(before general SFT, following OLMo3 Think-SFT pattern)*

A dedicated reasoning SFT pass on long-CoT traces **before** general instruction SFT. This
is how OLMo3 and Nemotron 3 achieve strong reasoning: think-style SFT first, then general
SFT on top. Skipping this means the general SFT data will dilute reasoning capability.

| Pri | Dataset | Why |
|-----|---------|-----|
| P1 | `openeurollm/dolci-think-sft-tokenized` (`Dolci-Think-SFT-32B-decontaminated`) | OLMo3-validated long-CoT distilled from 32B; multilingual |
| P1 | `open-thoughts/OpenThoughts-114k` | canonical open reasoning SFT (math+code+science+puzzles); high quality |
| P2 | `allenai/Dolci-Think-SFT-Python` | code reasoning traces |
| P2 | `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated` | Nemotron reasoning distillation (6 EU langs) |
| P2 | *LightOn multilingual reasoning SFT* (Kai Hakala) | EU-language reasoning traces — coordinate with LightOn; not yet public |
| P3 | `nvidia/OpenMathInstruct-2` | 14M math reasoning traces; strong signal for verifiable math |
| P3 | `open-thoughts/OpenThoughts2-1M` | larger follow-up to OT-114k if quality holds |

**Recommended mix for reasoning SFT stage:** ~50% Dolci-Think, ~30% OpenThoughts-114k, ~20% code/math reasoning.  
**Max seq length:** 16K+ (reasoning traces are long — packing essential).

## Stage 1 — SFT  *(general instruction, after reasoning SFT)*
| Pri | Dataset | Why |
|-----|---------|-----|
| P1 | `openeurollm/dolci-instruct-sft-tokenized` (`Dolci-Instruct-SFT-decontaminated`, 70 langs) | core instruction, multilingual, training-ready |
| P2 | `allenai/Dolci-Instruct-SFT-Tool-Use` | tool / function-calling (not mirrored by OELLM) |
| P3 | `openeurollm/smoltalk2-decontaminated`, `open-perfectblend-decontaminated`, `orca-agentinstruct-1M-v1-decontaminated` | chat breadth + agent/tool |

## Stage 2 — DPO / preference
| Pri | Dataset | Why |
|-----|---------|-----|
| P1 | `openeurollm/Dolci-Instruct-DPO-translated` (11 langs) | multilingual preference |
| P2 | `allenai/Dolci-Think-DPO-7B`, `allenai/Dolci-Think-DPO-32B` | reasoning preference |
| P3 | `allenai/Dolci-DPO-Model-Response-Pool` | response pool for on-policy pairs |

## Stage 3 — RLVR / GRPO  (the gap — from AllenAI)
| Pri | Dataset | Why |
|-----|---------|-----|
| P1 | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` | canonical verifiable mix (math answers + IF constraints) |
| P1 | `allenai/Dolci-RL-Zero-Mix-7B` | domain-mixed verifiable RL (or per-domain `-Math`/`-Code`/`-IF`/`-General`) |
| P2 | `allenai/rlvr-code-data-python-r1-format-filtered` | code RL with checks |
| P2 | `allenai/Dolci-Think-RL-7B` / `-32B`, `allenai/Dolci-Instruct-RL` | full Dolci RL prompts |
| P3 | `allenai/aime2024-25-rlvr`, `allenai/RLVR-MATH`, `allenai/RLVR-GSM`, `allenai/RLVR-IFeval` | targeted math/IF |

## Stage 1b — Long-context SFT  (protect the base's 262K)
| Pri | Dataset | Why |
|-----|---------|-----|
| P1 | *synthesize long-doc QA/summarization from the **project's own** long-ctx corpus* — `birgermoell/oellm-longctx-tokenized-streamed-all-v2` + `BirgerMoell/openeuro-longctx-datamix` | EU-multilingual long-ctx, on-distribution (best signal) |
| P2 | `nvidia/ChatQA2-Long-SFT-data` | modern long-ctx SFT (LongAlpaca + OpenOrca + Long Data Collections) |
| P3 | `THUDM/LongAlign-10k`, `Yukang/LongAlpaca-12k` | long reading-comprehension/QA |

Use **packing + length-mixing** in SFT so long examples are present; otherwise short-only
SFT erodes the 262K context.

## Project-native sources (prefer these — OELLM-aligned)
- **`BirgerMoell/grpo-data-bootstrap`** — OELLM GRPO data starter kit → feed Stage 3 (RLVR/GRPO).
- **`BirgerMoell/openeurollm-language-dataset-candidates`** — curated dataset candidates for
  OELLM language post-training → mine for Stage 1 SFT (esp. multilingual EU coverage).
- **`BirgerMoell/openeuro-longctx-datamix`** + HF `oellm-longctx-tokenized-streamed-all-v2`
  — the long-ctx corpus → source for synthesizing long-context SFT (Stage 1b).
- **`BirgerMoell/trl`** — the project's TRL fork → the SFT/DPO/GRPO trainer to use.
- **`BirgerMoell/OneRuler-OELLM`** — multilingual long-context eval → `eval/`.

## Eval (OELLM-native)
`openeurollm/ArenaHard-EU-v0` (EU Arena-Hard) + lm-eval-harness suites (see `eval/`).

## Recommended starting mix (validate on 2B first)
- **SFT:** ~70% Dolci-Instruct-SFT, ~20% Dolci-Think-SFT, ~5% Tool-Use, ~5% long-ctx (ChatQA2).
- **DPO:** Dolci-Instruct-DPO (+ Think-DPO for reasoning).
- **GRPO:** RLVR-GSM-MATH-IF-Mixed-Constraints (+ code RL once vLLM-ROCm is validated).
Tune proportions after the 2B smoke + eval baselines.
