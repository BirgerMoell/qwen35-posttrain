#!/usr/bin/env python3
"""Text-only SFT entrypoint (TRL) — loads AutoModelForCausalLM + explicit tokenizer.

Why this exists: the stock `trl.scripts.sft` loads 2026 multimodal-origin models
(Gemma 4, Qwen 3.5) via AutoModelForImageTextToText. That loader:
  - pulls in the vision/audio towers we don't need for text SFT, and
  - ignores FSDP `cpu_ram_efficient_loading`, so every rank loads a full model copy
    to CPU (8 x 19 GB ~= 152 GB on one node) and OOMs at 9B+.

This wrapper loads the text-only AutoModelForCausalLM with low_cpu_mem_usage, which
respects the FSDP rank-0-only loading path, and passes an explicit tokenizer. Reuses
TRL's config dataclasses + dataset mixture loader, so the same YAML configs work.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_peft_config,
)


def _prune_no_split_modules(m):
    """Drop phantom vision/audio block classes that don't exist in the text-only model
    (otherwise FSDP TRANSFORMER_BASED_WRAP fails to find them)."""
    names = getattr(m, "_no_split_modules", None)
    if names:
        present = {type(sub).__name__ for sub in m.modules()}
        m._no_split_modules = [c for c in names if c in present]


def main(script_args, training_args, model_args, dataset_args):
    dtype_str = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_str if dtype_str in (None, "auto") else getattr(torch, dtype_str)

    # Initialize the distributed process group before loading so we know our rank.
    from accelerate import PartialState
    state = PartialState()

    # transformers 5.x REMOVED the env-based FSDP rank-0 (cpu_ram_efficient) loading, so by default
    # every rank reads the full checkpoint into CPU RAM (~120 GB/rank peak at 27B incl. the FSDP
    # shard transient) and OOMs the node. Opt in with RANK0_META_LOAD=1 for large models: only
    # global rank 0 reads weights; other ranks build the model on `meta` (no CPU/GPU memory) and
    # FSDP `sync_module_states=True` broadcasts rank-0's weights at wrap time. This decouples CPU
    # load from ranks/node (load is ~54 GB on ONE node), so we can use all 8 GPUs/node again.
    # Default (unset) keeps the prior per-rank load — unchanged for the 2B/4B/9B single-node runs.
    if os.environ.get("RANK0_META_LOAD") == "1" and not state.is_main_process:
        cfg = AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                cfg, dtype=dtype, attn_implementation=model_args.attn_implementation,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            dtype=dtype,
            trust_remote_code=model_args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
    _prune_no_split_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(dataset_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Resume from a saved checkpoint if requested (RESUME_FROM_CHECKPOINT=1 -> auto-find the latest
    # checkpoint-* in output_dir; or set it to an explicit checkpoint path). Restores model +
    # optimizer + step, so a run cut short by a wall-clock timeout continues instead of restarting.
    _resume = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
    if _resume in ("", "0"):
        resume = None          # disabled (default) — note "0" is a truthy string, must map to None
    elif _resume == "1":
        resume = True          # auto-find the latest checkpoint-* in output_dir
    else:
        resume = _resume       # explicit checkpoint path
    trainer.train(resume_from_checkpoint=resume)
    trainer.accelerator.print("✅ Training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig))
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
