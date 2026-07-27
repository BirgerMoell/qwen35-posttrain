#!/usr/bin/env python3
"""GLM-5.2 pipeline-parallel generation in pure transformers (increment 2).

One pipeline STAGE per process (1/node for the real model, using its 8 GCDs via device_map). Only
hidden states cross stage boundaries (torch.distributed send/recv); each stage keeps its own
DynamicCache and recomputes rotary + causal mask locally (all stages advance in lockstep, same seq
length). Stage boundaries land on 4-layer IndexShare groups so no DSA index state crosses nodes.

Dev on a TINY random model first (GLM_TINY=1, instant load, 2 ranks) to validate the mechanics; then
the same code runs the real GLM-5.2-FP8 across nodes.
"""
import math, os, time, torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

RANK = int(os.environ["RANK"]); WORLD = int(os.environ["WORLD_SIZE"])
IS_LAST = RANK == WORLD - 1
LOCAL = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
DEV = f"cuda:{LOCAL}"                # this process's own GCD (distinct per task on a node)


def log(*a):
    if RANK == 0 or IS_LAST:
        print(f"[rank{RANK}]", *a, flush=True)


def patch_fp8_pure_torch():
    """Replace transformers' finegrained-fp8 matmul (needs the `kernels` Triton kernel + FP8 HW,
    neither on MI250X) with a pure-torch block dequant: dequant e4m3 weight × per-128×128 block
    scale → bf16, then a normal linear. Slow but hardware-agnostic and correct."""
    import torch.nn.functional as F
    import transformers.integrations.finegrained_fp8 as _fp8

    def _dequant(weight, scale):
        """Block-dequant e4m3 weight × per-block scale → fp32. Per dim, block = N//sN when evenly
        divisible (covers 128-blocks and fused 1024-blocks exactly), else the standard 128 with a
        partial tail. Scale is expanded by INTEGER INDEX floor(i/block) so partial tails map right."""
        wf = weight.to(torch.bfloat16)                 # bf16 (not fp32) to halve transient memory
        s = scale.to(torch.bfloat16)
        while s.dim() < wf.dim():
            s = s.unsqueeze(-1)
        for d in range(wf.dim()):
            N, sN = wf.shape[d], s.shape[d]
            if sN == N:
                continue
            block = N // sN if N % sN == 0 else 128
            idx = (torch.arange(N, device=s.device) // block).clamp(max=sN - 1)
            s = s.index_select(d, idx)
        return wf * s

    def fp8_linear_torch(input, weight, weight_scale_inv, block_size=None, bias=None,
                         activation_scale=None, output_dtype=None):
        od = output_dtype or input.dtype
        if weight.dtype == torch.float8_e4m3fn and weight_scale_inv is not None:
            w = _dequant(weight, weight_scale_inv).to(od)
        else:
            w = weight.to(od)
        return F.linear(input.to(od), w, bias.to(od) if bias is not None else None).to(od)

    _fp8.fp8_linear = fp8_linear_torch
    if RANK == 0:
        print("[patch] fp8_linear -> pure-torch block dequant", flush=True)


def build_config():
    tiny = os.environ.get("GLM_TINY") == "1"
    cfg = AutoConfig.from_pretrained(os.environ["GLM_MODEL"], trust_remote_code=True)
    if tiny:
        cfg.hidden_size = 1024; cfg.num_hidden_layers = 8; cfg.first_k_dense_replace = 1
        cfg.intermediate_size = 1024; cfg.moe_intermediate_size = 256
        cfg.n_routed_experts = 8; cfg.num_experts_per_tok = 2
        cfg.num_attention_heads = 8; cfg.num_key_value_heads = 8
        cfg.index_n_heads = 4; cfg.index_topk = 32; cfg.max_position_embeddings = 4096
        for k in ("indexer_types", "mlp_layer_types"):
            if isinstance(getattr(cfg, k, None), list):
                base = getattr(cfg, k); setattr(cfg, k, [base[min(i, len(base) - 1)] for i in range(cfg.num_hidden_layers)])
    cfg._experts_implementation = "eager"
    return cfg, tiny


def build_stage(cfg, tiny, lo, hi):
    ngpu = torch.cuda.device_count()
    if tiny:
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                 attn_implementation="eager").to(DEV, dtype=torch.bfloat16)
    else:
        L = cfg.num_hidden_layers
        dmap = {f"model.layers.{i}": (i - lo) % ngpu if lo <= i < hi else "meta" for i in range(L)}
        dmap["model.embed_tokens"] = 0 if RANK == 0 else "meta"
        dmap["model.rotary_emb"] = 0
        dmap["model.norm"] = ngpu - 1 if IS_LAST else "meta"
        dmap["lm_head"] = ngpu - 1 if IS_LAST else "meta"
        model = AutoModelForCausalLM.from_pretrained(
            os.environ["GLM_MODEL"], dtype=torch.bfloat16, device_map=dmap,
            experts_implementation="eager", attn_implementation="eager", trust_remote_code=True)
    return model


def stage_bounds(cfg, world):
    """Split layers into `world` contiguous stages, snapping each boundary to a 'full' DSA indexer
    layer so every stage starts a fresh index group (prev_topk_indices=None) — no DSA index state
    ever crosses a node boundary; only hidden states do."""
    L = cfg.num_hidden_layers
    types = list(getattr(cfg, "indexer_types", ["full"] * L))
    fulls = [i for i, t in enumerate(types) if t == "full"] or [0]
    starts = [0]
    for k in range(1, world):
        target = round(k * L / world)
        starts.append(min((f for f in fulls if f >= 1), key=lambda f: abs(f - target)))
    starts = sorted(set(starts))
    while len(starts) < world:                      # pad if fulls were too sparse
        starts.append(starts[-1])
    starts = starts[:world]
    bounds = [(starts[r], starts[r + 1] if r + 1 < world else L) for r in range(world)]
    return bounds


def run_layers(base, layers, hidden, cache, position_ids, cache_position, cfg):
    """Run this stage's decoder layers, threading DSA prev_topk_indices (starts None — each stage
    begins on a 'full' indexer layer)."""
    pos_emb = base.rotary_emb(hidden, position_ids=position_ids)
    causal = create_causal_mask(config=cfg, inputs_embeds=hidden, attention_mask=None,
                                past_key_values=cache, position_ids=position_ids)
    topk = None
    for layer in layers:
        hidden, topk = layer(hidden, attention_mask=causal, position_embeddings=pos_emb,
                             position_ids=position_ids, past_key_values=cache,
                             use_cache=True, cache_position=cache_position, prev_topk_indices=topk)
    return hidden


def main():
    import datetime
    torch.cuda.set_device(LOCAL)
    # generous timeout: stages load ~26 min and finish minutes apart — collectives must wait it out
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD,
                            timeout=datetime.timedelta(hours=1))
    patch_fp8_pure_torch()
    cfg, tiny = build_config()
    L = cfg.num_hidden_layers
    lo, hi = stage_bounds(cfg, WORLD)[RANK]
    H = cfg.hidden_size
    log(f"stage layers [{lo},{hi}) of {L}, hidden={H}, tiny={tiny}")

    t = time.time()
    model = build_stage(cfg, tiny, lo, hi)
    base = model.model
    layers = base.layers[lo:hi]
    log(f"loaded in {time.time()-t:.0f}s")
    dist.barrier()                     # wait until EVERY stage finished loading before the pipeline
    if RANK == 0:
        print("[spmd] all stages loaded — starting generation", flush=True)

    tok = None if tiny else AutoTokenizer.from_pretrained(os.environ["GLM_MODEL"], trust_remote_code=True)
    if tiny:
        ids = torch.randint(0, cfg.vocab_size, (1, 6), device=DEV)
    else:
        prompt = os.environ.get("PROMPT", "Vad är meningen med livet? Svara kort.")
        # return_dict=True -> BatchEncoding; take input_ids so `ids` is a plain tensor (`.shape`)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(DEV)
    max_new = int(os.environ.get("MAX_NEW", "24"))

    cache = DynamicCache(config=cfg)
    pos = 0
    generated = []
    gen_ctx = torch.inference_mode()   # no autograd graph — critical: otherwise every on-the-fly
    gen_ctx.__enter__()                # dequantized weight copy is retained → OOM on a 744B model
    for step in range(max_new + 1):
        is_prefill = step == 0
        T = ids.shape[1] if is_prefill else 1
        position_ids = torch.arange(pos, pos + T, device=DEV).unsqueeze(0)
        cache_position = torch.arange(pos, pos + T, device=DEV)

        if RANK == 0:
            cur = ids if is_prefill else next_tok.view(1, 1)
            hidden = base.embed_tokens(cur)
        else:
            hidden = torch.empty((1, T, H), dtype=torch.bfloat16, device=DEV)
            dist.recv(hidden, src=RANK - 1)

        hidden = run_layers(base, layers, hidden, cache, position_ids, cache_position, cfg)

        if not IS_LAST:
            dist.send(hidden.to(DEV).contiguous(), dst=RANK + 1)
            next_tok = torch.zeros(1, dtype=torch.long, device=DEV)
        else:
            hidden = base.norm(hidden)
            logits = model.lm_head(hidden[:, -1, :])
            next_tok = logits.argmax(-1)
        # last stage tells rank 0 (and everyone) the next token
        dist.broadcast(next_tok, src=WORLD - 1)
        pos += T
        if step < max_new:
            generated.append(int(next_tok.item()))
            if not tiny and next_tok.item() in (getattr(tok, "eos_token_id", None) or -1,):
                break

    if RANK == 0:
        text = tok.decode(generated, skip_special_tokens=True) if not tiny else str(generated)
        print(f"\n[PIPELINE OK] generated {len(generated)} tokens\n[GLM-5.2] {text}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
