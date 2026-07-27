# GLM-5.2 — How It Was Trained

A consolidated report on GLM-5.2's training, from the **GLM-5 technical report** (arXiv 2602.15763),
the **IndexShare** paper (arXiv 2603.12201), the z.ai blog, and the HF model card. GLM-5.2 (released
2026-06-13, MIT license) is ZhipuAI's flagship **long-horizon / agentic-coding** model — a 744B-param
sparse MoE (40–44B active) with a **solid 1M-token context**.

## 1. Architecture (what's being trained)

- **744B total / ~40B active**, **256 routed experts** (8 active) + 1 shared, **80 layers**
  (first few dense, rest fine-grained MoE), **aux-loss-free** sigmoid routing (`noaux_tc`).
- **MLA (Multi-head Latent Attention)**, "MLA-256" variant: **576-dim latent KV-cache**,
  256-dim heads, plus a **"Muon Split"** for per-head orthogonalization.
- **DSA (DeepSeek Sparse Attention):** a dynamic lightning indexer selects the **top-2048** keys/query
  → ~1.5–2× less attention compute at long sequence; adapted with a 1,000-step warm-up + **20B-token**
  sparse-adaptation phase.
- **IndexShare (new in 5.2):** reuses the DSA indexer across **every 4 sparse-attention layers**,
  eliminating ~75% of indexer compute → **2.9× lower per-token FLOPs at 1M ctx**, 1.82× prefill /
  1.48× decode speedup.
- **MTP (Multi-Token Prediction):** **3 MTP layers** (vs DeepSeek-V3's 1) with parameter sharing;
  drives training signal + speculative decoding (acceptance length 2.76 vs DeepSeek-V3.2's 2.55).

## 2. Pretraining (~27T tokens)

- **27T-token** base corpus (**28.5T** including mid-training).
- **Web:** refined DCLM classifier + sentence-embedding quality scoring; a "World Knowledge"
  classifier to keep long-tail knowledge.
- **Code:** **+28% fuzzily-deduplicated unique tokens** vs GLM-4.5; broadened platform coverage and
  low-resource languages (Scala, Swift, Lua on top of the 9 core: Python/Java/Go/C/C++/JS/TS/PHP/Ruby).
- **Math & Science:** high-quality web/books/papers with **LLM-based educational-value filtering**.
- Bilingual core (English + Chinese), multilingual coverage.

## 3. Mid-training — long-context extension (~1.55T tokens)

Progressive window growth (bf16, packed):
| Stage | Context | Tokens |
|---|---|---|
| 1 | 32K | 1T |
| 2 | 128K | 500B |
| 3 | 200K | 50B |

- Long-context data: **natural** (books/papers, PPL+dedup filtered) + **synthetic** (NextLong /
  EntropyLong-style, interleaved packing to beat lost-in-the-middle); MRCR-like multi-turn at 200K.
- **Software-engineering data:** ~**10M issue–PR pairs** → ~**160B unique tokens** after filtering
  (the backbone of the later agentic-coding RL).

## 4. Post-training

### SFT
- Up to **202,752-token** context. Three data families: **General Chat / Reasoning / Coding&Agent**.
- Three thinking modes: **Interleaved**, **Preserved**, **Turn-level** thinking (the "flexible effort"
  reasoning controls).

### Reinforcement learning (staged)
Backbone: **GRPO + IcePop** (distinguishes training policy π_train from inference policy π_infer to
handle rollout/train mismatch). Hyperparams: β=2, ε_low=0.2, ε_high=0.28, group size 32, batch 32.
1. **Reasoning RL** — math, science, code, tool-integrated reasoning; difficulty-filtered; outcome
   rewards from domain judges.
2. **Agentic RL** — **>10,000 verifiable environments** (SWE + terminal), the 10M issue–PR corpus,
   multi-hop search over a Web Knowledge Graph; **asynchronous decoupled** rollouts.
3. **General RL** — hybrid rewards: **rule-based + Outcome Reward Models (ORM) + Generative Reward
   Models (GRM)**; multi-dim objectives (correctness, EQ, task quality) with human-in-the-loop anchors.

### GLM-5.2-specific post-training upgrades
- **Critic-based PPO** (switched from GRPO) for long-horizon tasks: per-token value baseline instead
  of group advantages, single trajectories; **compaction-aware** training cut group sampling from
  ~16h → ~2h.
- **Anti-hack validation:** rule-based filter + LLM intent judge; blocks reward-hacking exploits
  (curl downloads, eval-file scanning, credential extraction) by returning dummy data (keeps rollouts
  stable rather than crashing).
- **On-Policy cross-stage Distillation (OPD):** consolidates **10+ specialist** models into one
  generalist — student generates its own trajectories, teacher gives **token-level** supervision;
  group size 1, batch 1024; **mitigates catastrophic forgetting** across sequential RL stages;
  merged specialists in ~48h.
- **Speculative-decoding training:** 5-token draft window; architectural sharing + rejection
  sampling + a **joint total-variation loss** aligning draft/target distributions → acceptance length
  **4.56 → 5.47** tokens.

## 5. Training infrastructure — "slime"

- **slime**: unified RL infra with **decoupled inference & training engines on separate GPUs**
  (maximizes utilization), a server-based multi-task rollout orchestrator supporting **>1k concurrent
  rollouts**, DP-aware routing with consistent hashing for KV-cache locality. Built on **Megatron**
  (train) + **SGLang** (rollout).
- Memory/parallelism: interleaved pipeline parallelism with flexible MTP placement, **Pipeline-ZeRO2**
  gradient sharding, layer-granular activation offload to host, sequence-chunked output projection.
- Inference: **FP8 rollout**, **Prefill–Decode disaggregation**, **INT4 QAT** with bitwise-identical
  behavior; multi-node (e.g. **EP64 × DP64 over 8 nodes**).
- Portability: tuned across **7 domestic chip platforms** (Ascend, Moore Threads, Hygon, Cambricon,
  Kunlunxin, MetaX, Enflame).

## 6. What's worth borrowing for *our* project

- **On-Policy Distillation to consolidate specialists → one generalist.** We have exactly this
  problem: separate medical / tool-use / defect-repair / reasoning tracks. OPD (student rolls out,
  teacher gives token-level supervision, mitigates forgetting) is the principled way to merge them —
  far better than naively concatenating SFT data.
- **IcePop (π_train vs π_infer mismatch handling)** — directly relevant to our GRPO on ROCm, where the
  rollout engine and trainer can diverge numerically.
- **Anti-hack reward validation** — mirrors the reward-hack we just hit on the tool GRPO (abstain→0);
  their rule+intent-judge pattern is the general fix.
- **Aux-loss-free MoE routing + shared expert** — if we ever train MoE.
- **Staged RL (reasoning → agentic → general) with a final distillation consolidation** — a clean
  template for scaling our own post-training beyond single-stage SFT/GRPO.

## Sources
- GLM-5 Technical Report — https://arxiv.org/abs/2602.15763
- IndexShare — https://arxiv.org/abs/2603.12201
- GLM-5.2 blog — https://z.ai/blog/glm-5.2 · model card — https://huggingface.co/zai-org/GLM-5.2
- GLM-4.5 (ARC) Technical Report — https://arxiv.org/abs/2508.06471
