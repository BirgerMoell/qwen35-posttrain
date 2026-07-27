# GLM-5.2 pure-transformers pipeline serving on LUMI — design + progress

Serving GLM-5.2 (744B, `glm_moe_dsa`, MLA+DSA+MoE+MTP) on MI250X where vLLM can't (its
transformers-MoE backend lacks GLM's `e_score_correction_bias` router; see
`glm52_architecture_report.md`). We build multi-node **pipeline parallelism in pure transformers**,
reusing the validated eager path (`experts_implementation="eager"`, `attn_implementation="eager"`).

## Validated so far
- **Arch runs on ROCm** (tiny random model, forward+generate OK) with eager experts.
- **Networking** — external_launcher/torch.distributed over Slingshot works (`glm_spmd_lumi.sbatch`).
- **INCREMENT 1 — per-stage loading ✅** (`scripts/glm_pp_load_test.py`, `slurm/glm_pp_lumi.sbatch`):
  one stage per node; a `device_map` that puts this stage's layers on its 8 GCDs and all other
  modules on `meta` — transformers loads + FP8-dequants only the slice. Result on 4 nodes: layers
  split [0,20)/[20,40)/[40,60)/[60,78), **only ~20–30 GB/GCD used** (FP8 stays compressed on GPU
  when placed directly), real weights resident. ⇒ **fits 2–3 nodes, not 4.** Load ~26 min/stage.

## Model forward (what increment 2 must replicate, split across stages)
`GlmMoeDsaModel.forward` loop, per layer:
```
position_embeddings = rotary_emb(hidden, position_ids)     # each stage can compute locally
causal_mask = create_causal_mask(..., past_key_values, position_ids)
for decoder_layer in layers:
    hidden, topk_indices = decoder_layer(hidden, attention_mask=causal_mask,
        position_embeddings=position_embeddings, position_ids=position_ids,
        past_key_values=past_key_values)   # DynamicCache
hidden = norm(hidden)                       # last stage
logits = lm_head(hidden)                    # last stage
```
**Key insight:** IndexShare reuses the DSA indexer across **4-layer groups**; our stage boundaries
(20/40/60) are multiples of 4, so `topk_indices` never crosses a stage boundary — each stage is
self-contained (no need to ship index state between nodes).

## INCREMENT 2 — pipelined generation (next)
Each stage keeps its own `DynamicCache`. Only `hidden_states` (+ scalar position bookkeeping) crosses
stage boundaries via `torch.distributed.isend/irecv`. `position_ids`/`causal_mask`/`position_embeddings`
are recomputed locally per stage from the shared position counter (all stages see the same seq length).
- **Prefill:** stage0 embeds prompt → runs layers [0,20) → send hidden to stage1 → … → stage3 runs
  [60,78) + norm + lm_head → logits[:, -1] → sample t0 → send t0 back to stage0.
- **Decode loop:** stage0 embeds the new token → pipe forward (each stage appends to its cache) →
  stage3 samples t_{k+1} → send to stage0. Repeat to max_new_tokens / EOS.
- **Dev strategy:** debug the mechanics on a TINY `glm_moe_dsa` (instant load) — the real model costs
  ~26 min/load per iteration — then scale to 2–3 nodes.
- Batch many prompts per pipeline fill to amortize the eager-MoE cost (this is for distillation
  data-gen, latency-insensitive).

## INCREMENT 3 — wrap as distillation producer
Feed `distill_from_9b.py`-style prompt sets through the pipeline generator; write `{prompt, answer}`
jsonl; filter (purity/length/loop). GLM-5.2 as the high-quality teacher.
