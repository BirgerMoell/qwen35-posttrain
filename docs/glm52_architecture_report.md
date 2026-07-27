# GLM-5.2 — Architecture Report

A technical teardown of **GLM-5.2** (ZhipuAI / z.ai), grounded in its actual `config.json`
(`zai-org/GLM-5.2`, `architectures: ["GlmMoeDsaForCausalLM"]`, `model_type: glm_moe_dsa`). The
short version: GLM-5.2 is a **~700B-parameter (~45B active) sparse MoE with a 1M-token context**
that assembles essentially the entire modern frontier-efficiency stack — **fine-grained MoE +
aux-loss-free routing + Multi-head Latent Attention + DeepSeek Sparse Attention + Multi-Token
Prediction**. It is ZhipuAI's implementation of the "DeepSeek recipe" at scale.

## 1. Headline shape (from config)

| Property | Value | Note |
|---|---|---|
| Hidden size | 6144 | |
| Layers | 78 | first 3 **dense**, remaining 75 **MoE** (`first_k_dense_replace: 3`) |
| Vocab | 154,880 | |
| Context | **1,048,576 (1M)** | `rope_theta: 8e6` (huge base for length) |
| Routed experts | **256**, top-**8** + **1 shared** | fine-grained; 9 experts active/token |
| Expert size | `moe_intermediate_size: 2048` | small → fine-grained |
| Dense FFN size | 12,288 | first 3 layers |
| Attention | **MLA** (`q_lora_rank 2048`, `kv_lora_rank 512`) | compressed latent KV |
| Sparse attn | **DSA** indexer (`index_topk: 2048`) | selects ≤2048 keys/query |
| MTP | `num_nextn_predict_layers: 1` | multi-token prediction |
| dtype | bfloat16 | 282 safetensors shards ≈ **~1.3 TB** on disk |

**Rough parameter budget:** 75 MoE layers × 256 experts × (3 × 2048 × 6144) ≈ **~725B** in routed
experts, plus shared expert / attention / dense / embeddings ⇒ **~700–750B total**. Active per token
≈ 9 experts × (3 × 2048 × 6144) × 75 + attention ≈ **~45B**. So ~6% of the model fires per token —
frontier-level sparsity (700B-class quality at ~45B-class inference compute).

## 2. The four ideas that make it work

### (a) Fine-grained MoE with aux-loss-free routing
- **256 small experts, 8 active + 1 shared** (`n_routed_experts 256`, `num_experts_per_tok 8`,
  `n_shared_experts 1`). Many small experts (vs a few big ones) gives finer specialization and a
  combinatorially larger set of active-expert combinations — more capacity per active FLOP.
- **`topk_method: noaux_tc`** + **`scoring_func: sigmoid`** + `routed_scaling_factor 2.5`,
  `norm_topk_prob`. This is DeepSeek-V3's **auxiliary-loss-free load balancing**: instead of an
  aux loss that fights the language-model loss, a per-expert **bias** is nudged to keep experts
  balanced. Result: no capacity drop from a competing objective, and stabler training. `float32`
  router (`moe_router_dtype`) keeps routing numerically stable.
- **Shared expert** always on → captures common patterns so the routed experts specialize.

### (b) MLA — Multi-head Latent Attention (cheap KV cache)
- `q_lora_rank 2048`, `kv_lora_rank 512`: Q and KV are projected through **low-rank latents**. The
  KV cache stores a **512-dim latent per token** instead of full K/V for 64 heads — a ~10–20×
  smaller cache, which is what makes a 1M-token context financially possible.
- **Decoupled RoPE:** `qk_head_dim 256 = qk_nope_head_dim 192 + qk_rope_head_dim 64` — most of each
  head carries **no positional rotation** (compressible), and a small 64-dim slice carries RoPE.
  This lets MLA compress the bulk while still encoding position. (`v_head_dim 256`.)

### (c) DSA — DeepSeek Sparse Attention (the "dsa" in glm_moe_dsa)
- A **lightning indexer** (`index_n_heads 32`, `index_head_dim 128`, `indexer_rope_interleave`)
  cheaply scores all past tokens; each query then attends only to the **top `index_topk: 2048`**.
- `indexer_types` alternate `full` / `shared` in a `[full, shared×3]` pattern (with the first 3
  dense layers `full`) → some layers index independently, most share the index to save compute.
- Effect: attention cost grows with a **fixed 2048-key budget**, not with sequence length, so the
  1M context stays tractable at inference. DSA is the key long-context efficiency lever.

### (d) MTP — Multi-Token Prediction
- `num_nextn_predict_layers: 1`: an auxiliary head predicts the **next-next** token during training
  (denser gradient signal → better sample efficiency) and doubles as a **draft head for speculative
  decoding** at inference (`index_share_for_mtp_iteration: true`). Free-ish speedup + better training.

## 3. Why this is a great frontier architecture

- **Quality-per-FLOP.** ~700B of stored knowledge/skills, but only ~45B active — you pay ~45B-model
  inference cost for ~700B-model capability. Fine-grained experts + shared expert push the
  capacity/compute frontier further than coarse MoE.
- **Long context is actually usable.** MLA shrinks the KV cache and DSA caps attention at 2048 keys,
  so **1M tokens** isn't just an advertised number — it's serveable. Most models' "long context"
  dies on KV-cache memory and O(n²) attention; MLA+DSA attack both.
- **Stable training at scale.** Aux-loss-free routing removes the classic MoE tug-of-war; a float32
  router and a shared expert further stabilize it. MTP densifies the learning signal.
- **Convergence to a proven recipe.** MLA + fine-grained MoE + noaux routing + MTP + sparse
  attention is the DeepSeek-V3/V3.2 stack. GLM-5.2 adopting it wholesale is a strong signal that
  this is *the* current frontier-efficiency design — and ZhipuAI executes it at ~700B with 1M ctx.
- **Open weights.** Unlike closed frontier models, GLM-5.2 ships weights (bf16), so it can be
  self-hosted and used as a distillation teacher — subject to the serving cost below.

## 4. Serving reality on our stack (LUMI / MI250X / vLLM 0.15.1)

**GLM-5.2's arch is not yet in our vLLM.** `GlmMoeDsaForCausalLM` is **absent** from vLLM 0.15.1's
registry (which has `Glm4MoeForCausalLM` = GLM-4.5/4.6, but not the DSA variant). It **is** in
`transformers` (`glm_moe_dsa`, requires the 5.12 overlay). So the options are:

1. **transformers backend** (works today, slow): serve via `transformers` generate, or vLLM's
   `--model-impl transformers` if it accepts the MoE+DSA modeling — no custom kernels, so throughput
   is poor for a 700B. Fine for small distillation batches, not for bulk.
2. **Wait for / add vLLM `GlmMoeDsa` support** — the fast path, but needs the kernels (MLA + DSA
   indexer + fine-grained MoE) registered for ROCm. Non-trivial.
3. **Use GLM-4.6 instead** (`Glm4MoeForCausalLM`, **vLLM-supported today**): 355B / 32B active,
   Glm4Moe (MLA + MoE, no DSA), 200K context. Frontier-class, fits **2 nodes bf16**, and works with
   `slurm/serve_glm_lumi.sbatch` **as-is**. **Recommended teacher for distillation right now.**

**Footprint (bf16, MI250X 64 GB/GCD, no native FP8):**
- GLM-5.2 (~1.3 TB) → ~24–28 GCDs → **3–4 nodes**; no published FP8 quant (`quant dirs: none`).
- GLM-4.6 (~710 GB) → ~14–16 GCDs → **2 nodes** (TP=8 × PP=2) — the practical choice.

## 5. Relevance to this project

GLM-5.2 (or the serveable GLM-4.6) is the **quality ceiling** for our distillation: our earlier 9B
self-distillation showed teacher errors (currency drift, misread numbers); a frontier teacher raises
that ceiling. But **serving GLM-5.2 self-hosted is a multi-node, no-vLLM-kernel project** — for
near-term distillation, the pragmatic path is **GLM-4.6 via vLLM (2 nodes, supported)** or the
**z.ai API** (reachable from LUMI login, needs a key). Reserve full GLM-5.2 self-hosting for when
vLLM adds `GlmMoeDsa` or we invest in the transformers-backend serving.
