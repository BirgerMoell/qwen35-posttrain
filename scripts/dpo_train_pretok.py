#!/usr/bin/env python3
"""Stage-4 DPO on pre-tokenized pairs (TRL 0.29): trains on exactly the audited tokens.

Input: parquet shards from build_oellm_dpo_v1_pretok.py with `prompt_ids`, `chosen_ids`, `rejected_ids`.
TRL 0.29's DPOTrainer always re-runs prompt extraction and chat-template tokenization in `_prepare_dataset`;
`PretokenizedDPOTrainer` replaces that with column checks, so the data is never re-rendered.

Guards (all abort before training):
- the build manifest's kind, template sha256 and tokenizer.json sha256 must match the files used here, and the
  tokenizer's bundled chat_template (if any) must equal the template;
- every shard is scanned with pyarrow: no forbidden token id (default: chat tags 11-18) anywhere in a pair, no
  empty completion, every completion contains <|im_end|>, and prompt+completion <= max_length (so TRL's
  truncation never fires);
- every batch is re-checked for forbidden ids before the forward pass.

Environment:
  DPO_TRAIN_DATA_GLOB / DPO_EVAL_DATA_GLOB   parquet shards (required)
  DPO_DATA_MANIFEST                          manifest.json of the build (required)
  CHAT_TEMPLATE_FILE                         template the data was built with (required)
  TOKENIZER_NAME_OR_PATH                     defaults to the model path
  DPO_FORBID_TOKEN_IDS                       default "11,12,13,14,15,16,17,18"
  DPO_EVAL_MAX_ROWS / DPO_EVAL_SEED          eval subsample (default 1024 / 42)
  DPO_TRAIN_MAX_ROWS                         engineering smoke only: first N shuffled rows
  REQUIRED_MODEL_DTYPE                       e.g. bfloat16
  RESUME_FROM_CHECKPOINT                     1 = latest checkpoint-* in output_dir, or an explicit path
  DPO_STOP_AFTER_STEP                        engineering smoke only: save a checkpoint at step N and exit
                                             (simulates a walltime kill, so the resume path can be tested)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = ("prompt_ids", "chosen_ids", "rejected_ids")
DEFAULT_FORBIDDEN = "11,12,13,14,15,16,17,18"
MANIFEST_KINDS = {"oellm_dpo_v1_pretokenized"}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_ids(value: str) -> list[int]:
    return sorted({int(t) for t in value.replace(":", ",").split(",") if t.strip()})


def check_manifest(manifest_path: str, template_path: str, tokenizer_json: str, max_length: int) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("kind") not in MANIFEST_KINDS:
        raise ValueError(f"unexpected manifest kind {manifest.get('kind')!r}")
    if manifest["template_sha256"] != sha256(template_path):
        raise ValueError("chat template differs from the one the DPO data was built with")
    if manifest["tokenizer_json_sha256"] != sha256(tokenizer_json):
        raise ValueError("tokenizer.json differs from the one the DPO data was built with")
    if manifest["max_length"] > max_length:
        raise ValueError(f"data built for max_length {manifest['max_length']} > training max_length {max_length}")
    return manifest


def scan_shard(path: str, forbidden: list[int], eos_id: int, max_length: int) -> int:
    """Vectorized integrity check of one shard; returns its row count or raises."""
    table = pq.read_table(path, columns=list(COLUMNS))
    prompt_len = pc.list_value_length(table["prompt_ids"])
    for side in ("chosen_ids", "rejected_ids"):
        col = table[side]
        length = pc.list_value_length(col)
        if pc.any(pc.equal(length, 0)).as_py():
            raise ValueError(f"{path}: empty {side}")
        if pc.max(pc.add(prompt_len, length)).as_py() > max_length:
            raise ValueError(f"{path}: prompt+{side} longer than {max_length}")
        is_eos = pc.equal(pc.list_flatten(col), eos_id)
        rows_with_eos = pc.count_distinct(pc.filter(pc.list_parent_indices(col), is_eos)).as_py()
        if rows_with_eos != table.num_rows:
            raise ValueError(f"{path}: {table.num_rows - rows_with_eos} {side} without <|im_end|>")
    value_set = pa.array(forbidden, type=pa.int32())
    for name in COLUMNS:
        if pc.any(pc.is_in(pc.list_flatten(table[name]), value_set=value_set)).as_py():
            raise ValueError(f"{path}: forbidden token id in {name}")
    return table.num_rows


class ForbiddenIdGuard:
    """Wraps the preference collator; aborts if a batch contains a forbidden token id."""

    def __init__(self, inner, forbidden: list[int]):
        self.inner = inner
        self.forbidden = forbidden

    def __call__(self, features):
        batch = self.inner(features)
        import torch
        ids = batch["input_ids"][batch["attention_mask"].bool()]
        if torch.isin(ids, torch.tensor(self.forbidden, device=ids.device)).any().item():
            raise RuntimeError(f"forbidden token id {self.forbidden} in a DPO batch")
        return batch


def _files(pattern: str) -> list[str]:
    files = [p for p in sorted(glob.glob(pattern)) if os.path.getsize(p)]
    if not files:
        raise FileNotFoundError(f"no parquet shards match {pattern}")
    return files


def _prune_no_split_modules(m):
    """Drop phantom vision/audio block classes so FSDP TRANSFORMER_BASED_WRAP finds the real layers."""
    names = getattr(m, "_no_split_modules", None)
    if names:
        present = {type(sub).__name__ for sub in m.modules()}
        m._no_split_modules = [c for c in names if c in present]


def main() -> None:
    import torch
    from accelerate import PartialState
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer, ModelConfig, ScriptArguments, TrlParser

    class PretokenizedDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            missing = [c for c in COLUMNS if c not in dataset.column_names]
            if missing:
                raise ValueError(f"{dataset_name} dataset lacks pre-tokenized columns {missing}")
            return dataset

    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    _, args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    state = PartialState()
    log = (lambda *a: print(*a, flush=True)) if state.is_main_process else (lambda *a: None)
    if args.max_length is None:
        raise ValueError("set max_length (it must cover the longest pre-tokenized pair)")

    template_path = os.environ["CHAT_TEMPLATE_FILE"]
    tokenizer_path = os.environ.get("TOKENIZER_NAME_OR_PATH", "").strip() or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    template = Path(template_path).read_text(encoding="utf-8")
    bundled = Path(tokenizer_path) / "chat_template.jinja"
    if bundled.exists() and bundled.read_text(encoding="utf-8") != template:
        raise ValueError(f"{bundled} differs from {template_path}")
    tokenizer.chat_template = template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    check_manifest(os.environ["DPO_DATA_MANIFEST"], template_path,
                   str(Path(tokenizer_path) / "tokenizer.json"), args.max_length)
    forbidden = parse_ids(os.environ.get("DPO_FORBID_TOKEN_IDS", DEFAULT_FORBIDDEN))

    train_files = _files(os.environ["DPO_TRAIN_DATA_GLOB"])
    eval_files = _files(os.environ["DPO_EVAL_DATA_GLOB"])
    rows = sum(scan_shard(p, forbidden, tokenizer.eos_token_id, args.max_length) for p in train_files + eval_files)
    log(f"[data] {len(train_files)} train + {len(eval_files)} eval shards, {rows} pairs passed the scan "
        f"(forbidden ids {forbidden}, eos {tokenizer.eos_token_id}, max_length {args.max_length})")

    with state.main_process_first():  # one rank builds the Arrow cache, the others reuse it
        data = load_dataset("parquet", data_files={"train": train_files, "eval": eval_files})
    data = data.select_columns(list(COLUMNS))
    if (n := int(os.environ.get("DPO_TRAIN_MAX_ROWS", "0"))):
        data["train"] = data["train"].shuffle(seed=args.seed).select(range(min(n, len(data["train"]))))
        log(f"[data] ENGINEERING SMOKE: train limited to {len(data['train'])} rows")
    if (n := int(os.environ.get("DPO_EVAL_MAX_ROWS", "1024"))):
        data["eval"] = data["eval"].shuffle(seed=int(os.environ.get("DPO_EVAL_SEED", "42"))).select(
            range(min(n, len(data["eval"]))))
    log(f"[data] train {len(data['train'])} pairs, eval {len(data['eval'])} pairs")

    dtype_name = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_name if dtype_name in (None, "auto") else getattr(torch, dtype_name)
    load = dict(revision=model_args.model_revision, attn_implementation=model_args.attn_implementation,
                dtype=dtype, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **load)
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **load)
    for m in (model, ref_model):
        _prune_no_split_modules(m)
    required = os.environ.get("REQUIRED_MODEL_DTYPE", "").strip()
    if required and next(model.parameters()).dtype != getattr(torch, required):
        raise ValueError(f"model dtype {next(model.parameters()).dtype} != REQUIRED_MODEL_DTYPE={required}")
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        raise ValueError("tokenizer has more ids than the model has embedding rows")

    trainer = PretokenizedDPOTrainer(
        model, ref_model, args=args, train_dataset=data["train"],
        eval_dataset=data["eval"] if args.eval_strategy != "no" else None, processing_class=tokenizer)
    trainer.data_collator = ForbiddenIdGuard(trainer.data_collator, forbidden)
    stop_after = int(os.environ.get("DPO_STOP_AFTER_STEP", "0"))
    if stop_after:
        class StopAfter(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == stop_after:
                    control.should_save = control.should_training_stop = True
        trainer.add_callback(StopAfter())

    resume = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
    resume = None if resume in ("", "0") else (True if resume == "1" else resume)
    trainer.train(resume_from_checkpoint=resume)
    if stop_after and trainer.state.global_step == stop_after:
        log(f"[smoke] stopped after step {stop_after} with a checkpoint; resume to continue")
        return
    trainer.save_model(args.output_dir)
    log(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
