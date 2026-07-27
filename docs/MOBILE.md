# Mobile / on-device European lineup — Qwen3.5-{2B,4B}-EU

A phone-ready European model family: take a Qwen3.5 Small base, post-train it for EU languages
with our pipeline, then quantize to **GGUF** (Android / cross-platform) and **MLX** (Apple) so it
runs on-device. Defaults to fast **no-think** answers with a **`/think`** toggle preserved.

| Model | Params | Q4_K_M size | Target device | Use |
|---|---|---|---|---|
| **Qwen3.5-0.8B-EU** | 0.8B | ~0.5 GB | any phone / testing | pipeline canary, low-end |
| **Qwen3.5-2B-EU** | 2B | ~1.4 GB | 4–6 GB RAM phones | fast assistant |
| **Qwen3.5-4B-EU** | 4B | ~2.5 GB | 8 GB+ RAM phones | best quality at mobile size |

Base: **Qwen3.5 Small Series** (0.8B/2B/4B/9B, Mar 2026) — built ground-up for on-device
(Gated Delta Net + MoE, 200+ languages, dual-mode thinking).

## Pipeline (reuses the EU post-training stack)
1. **SFT** — `configs/sft_qwen35_{2b,4b}_eu.yaml`: 85% general EU instruction data (no-think,
   direct answers) + 15% reasoning slice (think-format). The model defaults to no-think but keeps
   the `/think` path sharp. Full FT, 1 node × 8 GPUs (small models — no 27B-style CPU-OOM).
2. **SimPO** (reference-free) — `configs/simpo_qwen35_{2b,4b}_eu.yaml`: EU exam-MCQ preference
   pairs (~35 langs), one model, no OOM.
3. *(optional)* **GRPO/RLVR** on verifiable exam MCQs.

Launch: `sbatch ... --export=ALL,STAGE_LOAD=<base>,STAGE_SAVE=<out>,CFG=<cfg> slurm/sft_lumi.sbatch`
(then SimPO via `slurm/simpo_lumi.sbatch`). A 0.8B canary (`sft_qwen35_0.8b_eu_smoke.yaml`)
validates the data-mix wiring fast before the real runs.

## Quantize & export (net-new)
**GGUF (verified end-to-end on Qwen3.5-0.8B):** llama.cpp's converter AND runtime fully support
the Qwen3.5 arch. 0.8B → Q4_K_M = 506 MiB, generated coherent Swedish at ~39 tok/s on login CPU.
- Build llama.cpp **inside the container** (`build-container/`) — a login-node build links host
  libs missing in the container and hits a Cray-OpenMP crash at inference.
- `scripts/export_gguf.sh <hf_dir> <prefix> [Q4_K_M Q5_K_M Q8_0]` (convert → quantize), or batch:
  `MODEL=<ckpt> PREFIX=<…/qwen35-4b-eu> sbatch slurm/export_gguf_lumi.sbatch`.
- The GGUF embeds the native chat template; no-think is the default. To force no-think on a server
  pass `chat_template_kwargs={"enable_thinking":false}`; `/think` in the user turn enables reasoning.
- Note: `llama-server` returns reasoning in `reasoning_content` (separate from `content`) because
  `reasoning_format=deepseek`. The new `llama-cli` is interactive-only — use `llama-server` for
  one-shot/eval.

**MLX (Apple silicon — run on the Mac, not LUMI):** `scripts/export_mlx.sh <hf_dir_or_repo> <out>`
→ 4-bit MLX. Download the checkpoint from HF first (`hf download birgermoell/Qwen3.5-4B-EU`).

## Eval
- Full precision: `scripts/eval_eu_holdouts.py` + `scripts/run_euroeval.py` (Swedish) — existing sbatch.
- **Quantization retention**: serve the Q4_K_M with `llama-server`, then
  `scripts/eval_gguf_holdouts.py --base-url http://127.0.0.1:8080 …` (same scorers as full
  precision; reads `reasoning_content`+`content`). Report full-vs-Q4 accuracy delta.

## Run it on a phone / laptop
- **Ollama / LM Studio**: `ollama run hf.co/birgermoell/Qwen3.5-4B-EU` (point at the Q4_K_M GGUF).
- **llama.cpp (Android via Termux, desktop)**: `llama-server -m Qwen3.5-4B-EU-Q4_K_M.gguf -c 4096`.
- **Apple (iPhone/iPad/Mac)**: MLX 4-bit via `mlx_lm` or an MLX-Swift app.

See the plan at `~/.claude/plans/mossy-sleeping-lamport.md` for the full rollout.
