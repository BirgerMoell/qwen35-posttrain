#!/usr/bin/env python3
"""Reference-free preference optimization (SimPO) entrypoint — the practical replacement for DPO.

Why: DPO loads TWO 9B models (policy + frozen reference). On one LUMI node the reference copy
repeatedly OOMs CPU during load (FSDP rank-0-only loading wouldn't engage). SimPO (and CPO/ORPO)
are **reference-free** — a single model — so the memory problem vanishes. SimPO also matches or
beats DPO on AlpacaEval-2 / Arena-Hard, so this is an upgrade, not a workaround.

Implemented via TRL's CPOTrainer with loss_type="simpo" (CPO + SimPO gamma). Same flat YAML
schema (ModelConfig + DatasetMixtureConfig + CPOConfig), same {prompt,chosen,rejected} data.
"""
from __future__ import annotations

import os

import torch
from accelerate import PartialState
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import (
    CPOConfig,
    CPOTrainer,
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)


def main(script_args, training_args, model_args, dataset_args):
    dtype_str = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_str if dtype_str in (None, "auto") else getattr(torch, dtype_str)

    state = PartialState()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loading/parsing the parquet and mapping bare strings creates Arrow cache shards. Only the
    # global main process may write those shards; otherwise every distributed rank races to chmod
    # the same temporary files. Other ranks wait, then read the completed deterministic cache.
    with training_args.main_process_first(
        local=False, desc="loading and formatting the preference dataset"
    ):
        dataset = get_dataset(dataset_args)
        if os.environ.get("WRAP_PREFERENCE_STRINGS") == "1":
            # The exam preference parquet stores bare strings. Convert them to conversational
            # records so TRL applies this model's chat template instead of training raw text.
            def as_conversation(example):
                if not all(
                    isinstance(example.get(key), str)
                    for key in ("prompt", "chosen", "rejected")
                ):
                    return example
                return {
                    "prompt": [{"role": "user", "content": example["prompt"]}],
                    "chosen": [{"role": "assistant", "content": example["chosen"]}],
                    "rejected": [{"role": "assistant", "content": example["rejected"]}],
                }

            dataset = dataset.map(
                as_conversation,
                num_proc=training_args.dataset_num_proc,
                load_from_cache_file=not state.is_main_process,
                desc="Formatting preference strings as conversations",
            )

    if state.is_main_process:
        train_split = dataset[script_args.dataset_train_split]
        print(
            f"[simpo] dataset cache={os.environ.get('HF_DATASETS_CACHE')} "
            f"train_rows={len(train_split):,}",
            flush=True,
        )
        if os.environ.get("WRAP_PREFERENCE_STRINGS") == "1":
            preview_row = train_split[0]
            preview = tokenizer.apply_chat_template(
                preview_row["prompt"] + preview_row["chosen"],
                tokenize=False,
                add_generation_prompt=False,
            )
            if "<start_of_turn>" in (tokenizer.chat_template or ""):
                if "<start_of_turn>user" not in preview or "<start_of_turn>model" not in preview:
                    raise ValueError("Gemma-style preference chat-template validation failed")
            print(f"[simpo] preference template preview: {preview[:240]!r}", flush=True)

    if os.environ.get("SIMPO_PREPROCESS_ONLY") == "1":
        state.wait_for_everyone()
        if state.is_main_process:
            print("[simpo] preprocessing smoke test completed", flush=True)
        return

    # ONE model only — no reference. For 9B+, load its weights only on global rank 0;
    # the other ranks build a zero-storage meta model and FSDP syncs the parameters.
    if os.environ.get("RANK0_META_LOAD") == "1" and not state.is_main_process:
        cfg = AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                cfg,
                dtype=dtype,
                attn_implementation=model_args.attn_implementation,
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

    # Drop phantom vision/audio classes so FSDP TRANSFORMER_BASED_WRAP can wrap the text model.
    names = getattr(model, "_no_split_modules", None)
    if names:
        present = {type(m).__name__ for m in model.modules()}
        model._no_split_modules = [c for c in names if c in present]

    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    if os.environ.get("SIMPO_SKIP_FINAL_SAVE") == "1":
        trainer.accelerator.print("[simpo] canary requested; skipping final model save")
        return
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CPOConfig, ModelConfig, DatasetMixtureConfig))
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
