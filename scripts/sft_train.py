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

import copy
import glob
import os
from pathlib import Path

import torch
from datasets import DatasetDict, interleave_datasets, load_dataset
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


def _parse_mix_probabilities(value: str, dataset_count: int) -> list[float]:
    """Parse an opt-in row-sampling mixture from the environment.

    DatasetMixtureConfig concatenates sources by default. Long-context repair needs
    deliberate sampling because one 64K row carries far more tokens than a normal
    instruction row. Keep the legacy behavior unless DATASET_MIX_PROBABILITIES is
    explicitly set. A colon separator is supported because Slurm --export uses
    commas as delimiters.
    """
    raw = value.replace(",", ":").split(":")
    try:
        probabilities = [float(item.strip()) for item in raw if item.strip()]
    except ValueError as exc:
        raise ValueError(
            "DATASET_MIX_PROBABILITIES must contain numbers separated by ':'"
        ) from exc
    if len(probabilities) != dataset_count:
        raise ValueError(
            "DATASET_MIX_PROBABILITIES has "
            f"{len(probabilities)} values for {dataset_count} configured datasets"
        )
    if any(value < 0 for value in probabilities) or sum(probabilities) <= 0:
        raise ValueError("dataset mixture probabilities must be non-negative and sum above zero")
    total = sum(probabilities)
    return [value / total for value in probabilities]


def _get_dataset(dataset_args):
    """Load the normal TRL mixture, or reproducibly interleave its sources.

    The environment-controlled branch is intentionally generic and opt-in so all
    existing configs preserve their exact behavior. Only the columns required by
    SFT are retained before interleaving; project datasets carry heterogeneous
    provenance columns that otherwise make Arrow feature alignment fail.
    """
    explicit_train = os.environ.get("SFT_TRAIN_DATA_GLOB", "").strip()
    explicit_eval = os.environ.get("SFT_EVAL_DATA_GLOB", "").strip()
    if bool(explicit_train) != bool(explicit_eval):
        raise ValueError("SFT_TRAIN_DATA_GLOB and SFT_EVAL_DATA_GLOB must be set together")
    if explicit_train:
        train_files = [path for path in sorted(glob.glob(explicit_train)) if os.path.getsize(path)]
        eval_files = [path for path in sorted(glob.glob(explicit_eval)) if os.path.getsize(path)]
        if not train_files or not eval_files:
            raise FileNotFoundError("Explicit train/eval JSONL glob matched no files")
        # Pre-tokenized builds (input_ids + assistant_masks) are Parquet; TRL then skips
        # its own tokenization and trains on exactly the audited tokens.
        builder = "parquet" if all(p.endswith(".parquet") for p in train_files + eval_files) else "json"
        dataset = load_dataset(
            builder, data_files={"train": train_files, "test": eval_files}
        )
        eval_limit = int(os.environ.get("SFT_EVAL_MAX_ROWS", "0"))
        if eval_limit < 0:
            raise ValueError("SFT_EVAL_MAX_ROWS must be nonnegative")
        if eval_limit:
            eval_seed = int(os.environ.get("SFT_EVAL_SEED", "42"))
            dataset["test"] = dataset["test"].shuffle(seed=eval_seed).select(
                range(min(eval_limit, len(dataset["test"])))
            )
        print(
            f"[dataset] explicit split: {len(dataset['train'])} train rows "
            f"from {len(train_files)} files; {len(dataset['test'])} eval rows "
            f"from {len(eval_files)} files",
            flush=True,
        )
        return dataset

    mix_value = os.environ.get("DATASET_MIX_PROBABILITIES", "").strip()
    if not mix_value:
        return get_dataset(dataset_args)

    specs = list(dataset_args.datasets)
    probabilities = _parse_mix_probabilities(mix_value, len(specs))
    seed = int(os.environ.get("DATASET_MIX_SEED", "20260817"))
    stopping_strategy = os.environ.get("DATASET_MIX_STOPPING_STRATEGY", "all_exhausted")
    if stopping_strategy not in {"first_exhausted", "all_exhausted"}:
        raise ValueError(
            "DATASET_MIX_STOPPING_STRATEGY must be first_exhausted or all_exhausted"
        )

    loaded = []
    for spec in specs:
        one = copy.deepcopy(dataset_args)
        one.datasets = [spec]
        loaded.append(get_dataset(one))

    split_names = set.intersection(*(set(part.keys()) for part in loaded))
    if not split_names:
        raise ValueError("configured datasets do not share a train/eval split")

    mixed = {}
    for split in sorted(split_names):
        parts = []
        for part in loaded:
            current = part[split]
            if "messages" not in current.column_names:
                raise ValueError(f"dataset split {split!r} is missing the required messages column")
            parts.append(current.select_columns(["messages"]))
        mixed[split] = interleave_datasets(
            parts,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    print(
        "[dataset-mix] interleaving "
        f"{len(specs)} sources with row probabilities={probabilities}, "
        f"seed={seed}, stopping_strategy={stopping_strategy}",
        flush=True,
    )
    return DatasetDict(mixed)


def _load_chat_template_override(tokenizer) -> None:
    """Install an optional assistant-mask-aware template for this training run."""
    template_path = os.environ.get("CHAT_TEMPLATE_FILE", "").strip()
    if not template_path:
        return
    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"CHAT_TEMPLATE_FILE does not exist: {path}")
    tokenizer.chat_template = path.read_text(encoding="utf-8")
    print(f"[chat-template] loaded override from {path}", flush=True)


def _validate_assistant_mask(tokenizer, training_args) -> None:
    """Fail closed if assistant-only loss was requested with an unmaskable template."""
    if not getattr(training_args, "assistant_only_loss", False):
        return
    messages = [
        {"role": "user", "content": "Read record alpha."},
        {"role": "assistant", "content": "Record alpha is valid."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
    if mask is None or not any(mask) or all(mask):
        raise RuntimeError(
            "assistant_only_loss=True but the chat template did not return a valid "
            "assistant token mask; use CHAT_TEMPLATE_FILE with {% generation %} blocks"
        )
    masked_ids = [token for token, active in zip(rendered["input_ids"], mask) if active]
    if any(token in (17, 18) for token in masked_ids):
        raise RuntimeError("Direct-answer assistant labels contain <think> or </think> token IDs")
    generation_prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True,
    )
    if "<think>" in generation_prompt or "</think>" in generation_prompt:
        raise RuntimeError("Generation prompt injects thinking delimiters")
    print(
        f"[chat-template] assistant mask validated ({sum(mask)}/{len(mask)} target tokens)",
        flush=True,
    )


class _NoThinkLabelsCollator:
    """Abort before an optimizer step if labels contain forbidden think-delimiter IDs.

    Stage 1 forbids both <think> (18) and </think> (17); Stage 2 forbids only <think>,
    which the v2 template always places prompt-side.
    """

    def __init__(self, inner, forbidden=(17, 18)):
        self.inner = inner
        self.forbidden = tuple(forbidden)

    def __call__(self, features):
        batch = self.inner(features)
        labels = batch["labels"]
        if not (labels == -100).any().item():
            raise RuntimeError("batch has no masked labels; assistant-only loss is not applied")
        if any((labels == token).any().item() for token in self.forbidden):
            raise RuntimeError(f"batch contains supervised forbidden think tokens {self.forbidden}")
        return batch


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
    actual_dtype = next(model.parameters()).dtype
    required_dtype = os.environ.get("REQUIRED_MODEL_DTYPE", "").strip()
    if required_dtype and actual_dtype != getattr(torch, required_dtype):
        raise ValueError(
            f"Loaded model dtype {actual_dtype} does not match REQUIRED_MODEL_DTYPE={required_dtype}"
        )
    if state.is_main_process:
        print(f"[model] loaded parameter dtype {actual_dtype}", flush=True)

    tokenizer_path = os.environ.get("TOKENIZER_NAME_OR_PATH", "").strip() or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} IDs but model has only "
            f"{model.get_input_embeddings().num_embeddings} embedding rows"
        )
    if tokenizer_path != model_args.model_name_or_path:
        print(f"[tokenizer] loaded explicit tokenizer from {tokenizer_path}", flush=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.environ.get("TOKENIZER_NAME_OR_PATH", "").strip():
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    _load_chat_template_override(tokenizer)
    _validate_assistant_mask(tokenizer, training_args)

    dataset = _get_dataset(dataset_args)
    assistant_only = training_args.assistant_only_loss
    train_columns = dataset[script_args.dataset_train_split].column_names
    if assistant_only and "input_ids" in train_columns:
        # Pre-tokenized data: TRL only accepts assistant_only_loss for conversational rows,
        # but its collator applies a precomputed `assistant_masks` column unconditionally.
        if "assistant_masks" not in train_columns:
            raise ValueError("pre-tokenized data with assistant_only_loss needs an assistant_masks column")
        training_args.assistant_only_loss = False
        print("[dataset] pre-tokenized; loss masked by the precomputed assistant_masks", flush=True)

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
    if os.environ.get("SFT_FORBID_THINK_LABELS") == "1":
        if not assistant_only:
            raise ValueError("SFT_FORBID_THINK_LABELS requires assistant_only_loss=True")
        trainer.data_collator = _NoThinkLabelsCollator(trainer.data_collator)
        print("[safety] aborting on supervised think tokens in every batch", flush=True)
    elif os.environ.get("SFT_FORBID_LABEL_IDS", "").strip():
        if not assistant_only:
            raise ValueError("SFT_FORBID_LABEL_IDS requires assistant_only_loss=True")
        forbidden = [int(t) for t in os.environ["SFT_FORBID_LABEL_IDS"].replace(":", ",").split(",") if t.strip()]
        trainer.data_collator = _NoThinkLabelsCollator(trainer.data_collator, forbidden)
        print(f"[safety] aborting on supervised token IDs {forbidden} in every batch", flush=True)

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
