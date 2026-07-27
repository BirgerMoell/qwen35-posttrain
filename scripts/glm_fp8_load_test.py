#!/usr/bin/env python3
"""Resolve the GLM-5.2-FP8 unknowns on ROCm, single node first (8 GCDs + CPU offload):
  1) does block-wise FP8 (e4m3, 128x128) load/dequantize on MI250X at all?
  2) does it stay FP8 in memory (~744 GB, fits 1 node w/ offload) or balloon to bf16 (~1.5 TB, needs 3-4 nodes)?
  3) does a real forward+generate work with eager experts + eager attention?

device_map='auto' spreads across the 8 visible GCDs then CPU; slow (offload + eager MoE) but a single
generated token answers all three questions cheaply before committing to multi-node serving.
"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/scratch/project_465002530/users/bmoell/models/GLM-5.2-FP8")
ap.add_argument("--gpu-gb", type=int, default=58, help="max VRAM/GCD to leave headroom")
ap.add_argument("--cpu-gb", type=int, default=440)
a = ap.parse_args()

n = torch.cuda.device_count()
max_mem = {i: f"{a.gpu_gb}GiB" for i in range(n)}
max_mem["cpu"] = f"{a.cpu_gb}GiB"
print(f"visible GCDs: {n}, max_memory: {max_mem}", flush=True)

t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, device_map="auto", max_memory=max_mem,
    experts_implementation="eager", attn_implementation="eager", trust_remote_code=True,
).eval()
print(f"[load] OK in {time.time()-t:.0f}s", flush=True)

# where did layers land? (GPU vs CPU) + rough footprint
from collections import Counter
dev = Counter(str(p.device) for p in model.parameters())
print("[load] param device spread:", dict(dev), flush=True)
print("[load] GPU mem/GCD (GB):", [round(torch.cuda.memory_allocated(i)/1e9, 1) for i in range(n)], flush=True)

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
msgs = [{"role": "user", "content": "Vad är meningen med livet? Svara kort på svenska."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                              return_dict=True).to("cuda:0")
t = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=32, do_sample=False)
dt = time.time() - t
txt = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"[gen] {out.shape[1]-ids['input_ids'].shape[1]} tokens in {dt:.0f}s ({(out.shape[1])/dt:.2f} tok/s)")
print(f"[gen] ->\n{txt}")
print("[VERDICT] FP8 GLM-5.2 loads + generates on ROCm ✓")
