#!/usr/bin/env python3
"""Risk-gate GLM-5.2 (glm_moe_dsa) serving BEFORE downloading ~1.5 TB of weights.

Build a TINY random-init GlmMoeDsa from the real config (shrunk scale, real head/lora/vocab dims so
the modeling code exercises the same MLA + DSA + MoE + MTP machinery), then check:
  1) transformers 5.12 overlay can forward + generate it              -> serving via transformers OK
  2) vLLM 0.15.1 can serve it with --model-impl transformers          -> fast serving path OK
If (1) passes we can serve GLM-5.2 (slow but real); (2) is the throughput upgrade.
"""
import json, sys, os, tempfile, traceback
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "zai-org/GLM-5.2"
cfg = json.load(open(hf_hub_download(REPO, "config.json")))

# shrink SCALE only; keep head_dim / lora ranks / vocab so tokenizer + MLA/DSA dims stay valid
cfg.update({
    "hidden_size": 1024, "num_hidden_layers": 4, "first_k_dense_replace": 1,
    "intermediate_size": 1024, "moe_intermediate_size": 256,
    "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1,
    "num_attention_heads": 8, "num_key_value_heads": 8,
    "index_n_heads": 4, "index_topk": 32, "num_nextn_predict_layers": 1,
    "max_position_embeddings": 4096,
})
# per-layer lists must match num_hidden_layers
for k in ("indexer_types", "mlp_layer_types"):
    if k in cfg and isinstance(cfg[k], list):
        base = cfg[k]
        cfg[k] = [base[min(i, len(base)-1)] for i in range(cfg["num_hidden_layers"])]

tmp = tempfile.mkdtemp(prefix="glmdsa_tiny_")
json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
# grab tokenizer files from the real repo (small) so vocab matches
for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
    try:
        p = hf_hub_download(REPO, fn); import shutil; shutil.copy(p, os.path.join(tmp, os.path.basename(p)))
    except Exception:
        pass
print("tiny model dir:", tmp)

# ---- (1) transformers forward + generate — test the ARCH with raw ids (no tokenizer dependency) ----
tf_ok = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    print("building tiny GlmMoeDsa (random init)...", flush=True)
    # force the NAIVE/eager MoE (pure torch loop) — the default DeepGEMM grouped-gemm kernel is
    # CUDA-only and fails on ROCm ("grouped gemm is not supported on ROCM").
    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(tmp, trust_remote_code=True),
        trust_remote_code=True, experts_implementation="eager",
        attn_implementation="eager").to("cuda", dtype=torch.bfloat16).eval()
    ids = torch.randint(0, cfg["vocab_size"], (1, 16), device="cuda")
    with torch.no_grad():
        logits = model(ids).logits              # forward: exercises MLA + DSA + MoE + MTP
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    print(f"[1] transformers forward+generate OK -> logits {tuple(logits.shape)}, gen {tuple(gen.shape)}")
    tf_ok = True
except Exception:
    print("[1] transformers FAILED:\n" + traceback.format_exc()[-1800:])

# save the random tiny model so vLLM can load it
if tf_ok:
    model.save_pretrained(tmp, safe_serialization=True)

# ---- (2) vLLM transformers backend ----
try:
    from vllm import LLM, SamplingParams
    print("trying vLLM --model-impl transformers ...", flush=True)
    llm = LLM(model=tmp, model_impl="transformers", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.5, max_model_len=256)
    r = llm.generate(["Hello"], SamplingParams(max_tokens=8))
    print("[2] vLLM transformers-backend OK ->", r[0].outputs[0].text[:60])
except Exception:
    print("[2] vLLM transformers-backend FAILED:\n" + traceback.format_exc()[-1500:])
