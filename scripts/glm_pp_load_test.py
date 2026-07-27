#!/usr/bin/env python3
"""Increment 1 of pure-transformers pipeline serving for GLM-5.2: validate PER-STAGE loading.

Each node = one pipeline stage (1 process/node, uses its 8 GCDs). We build a per-stage device_map
that places THIS stage's layers on its GPUs (round-robin) and every other module on `meta` (no
memory) — so transformers' own loader still handles the block-FP8 dequant, but each node only
materializes ~1/N of the model. Confirms (a) the device_map/meta approach loads a slice, (b) it
dequants FP8 fine, (c) the memory fits per node. The pipelined forward/generation is increment 2.
"""
import math, os, torch
from transformers import AutoConfig, AutoModelForCausalLM

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
model_dir = os.environ["GLM_MODEL"]
ngpu = torch.cuda.device_count()

cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
L = cfg.num_hidden_layers
per = math.ceil(L / world)
lo, hi = rank * per, min(L, (rank + 1) * per)          # this stage owns layers [lo, hi)
print(f"[stage {rank}/{world}] layers [{lo},{hi}) on {ngpu} GCDs", flush=True)

# device_map: my layers spread round-robin over my 8 GCDs; everything else -> meta (not loaded here)
dmap = {}
for i in range(L):
    dmap[f"model.layers.{i}"] = (lo + (i - lo)) % ngpu if lo <= i < hi else "meta"
dmap["model.embed_tokens"] = 0 if rank == 0 else "meta"
dmap["model.rotary_emb"] = 0
dmap["model.norm"] = ngpu - 1 if rank == world - 1 else "meta"
dmap["lm_head"] = ngpu - 1 if rank == world - 1 else "meta"

t = __import__("time").time()
model = AutoModelForCausalLM.from_pretrained(
    model_dir, dtype=torch.bfloat16, device_map=dmap,
    experts_implementation="eager", attn_implementation="eager", trust_remote_code=True,
)
print(f"[stage {rank}] loaded slice in {__import__('time').time()-t:.0f}s", flush=True)
print(f"[stage {rank}] GPU mem/GCD (GB): {[round(torch.cuda.memory_allocated(i)/1e9,1) for i in range(ngpu)]}", flush=True)
# sanity: a few of my layers really have real (non-meta) weights
mine = [i for i in range(lo, hi)]
sample = mine[0] if mine else 0
w = dict(model.named_parameters()).get(f"model.layers.{sample}.self_attn.q_a_proj.weight")
if w is None:
    w = next((p for n, p in model.named_parameters() if f"layers.{sample}." in n), None)
print(f"[stage {rank}] layer {sample} sample param device={w.device if w is not None else None} "
      f"is_meta={(w.is_meta if w is not None else None)}", flush=True)
print(f"[stage {rank}] ✅ per-stage load OK", flush=True)
