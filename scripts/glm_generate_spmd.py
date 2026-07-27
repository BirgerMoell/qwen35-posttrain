#!/usr/bin/env python3
"""SPMD offline generation with GLM-5.2 via vLLM external_launcher (NO Ray).

Every rank runs this identically under srun (1 process/GCD); torch.distributed is initialized from
the Slurm env exactly like our multi-node TRAINING (which has working RCCL over Slingshot), and vLLM
uses that existing process group (`distributed_executor_backend="external_launcher"`). This avoids
the Ray device-masking + RCCL plumbing entirely.

For distillation we don't need a live HTTP server — we need batch answers for a prompt file. All
ranks cooperate on the TP forward; rank 0 writes the output.

Launched by slurm/glm_spmd_lumi.sbatch.
"""
import json, os, sys
import torch


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    torch.cuda.set_device(local)
    # init the process group from env (MASTER_ADDR/PORT/RANK/WORLD_SIZE) — same as training
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world)
    # TP must stay coarse enough that block-quantized (FP8 block_n=128) weight shards remain
    # divisible by 128 — TP=world (32) shrank a MoE weight to 64 and failed. Use TP=8 within a node
    # x PP across nodes (world = TP*PP).
    tp = int(os.environ.get("TP", "8"))
    pp = max(1, world // tp)
    if rank == 0:
        print(f"[spmd] world={world} — vLLM external_launcher TP={tp} PP={pp}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=os.environ["GLM_MODEL"],
        distributed_executor_backend="external_launcher",
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        model_impl="transformers",
        trust_remote_code=True,
        max_model_len=int(os.environ.get("MAXLEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
        enforce_eager=True,
    )

    prompts_file = os.environ.get("PROMPTS", "")
    if prompts_file and os.path.exists(prompts_file):
        rows = [json.loads(l) for l in open(prompts_file) if l.strip()]
        prompts = [r["prompt"] for r in rows]
    else:  # smoke test
        rows = None
        prompts = ["Vad är meningen med livet? Svara kort på svenska.",
                   "Write a haiku about the Baltic sea."]

    sp = SamplingParams(temperature=0.3, max_tokens=int(os.environ.get("MAX_NEW", "256")))
    chats = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(chats, sp)

    if rank == 0:
        out_path = os.environ.get("OUT", "")
        results = [{"prompt": p, "answer": o.outputs[0].text} for p, o in zip(prompts, outs)]
        if out_path:
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[spmd] wrote {len(results)} -> {out_path}", flush=True)
        for r in results[:3]:
            print(f"\n[spmd] PROMPT: {r['prompt']}\n[spmd] GLM-5.2: {r['answer'][:400]}", flush=True)
        print("\n[spmd] ✅ GLM-5.2 generated via external_launcher (no Ray)", flush=True)


if __name__ == "__main__":
    main()
