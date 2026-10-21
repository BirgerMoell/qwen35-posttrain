#!/usr/bin/env python3
"""Exercise the production wrapper's explicit train/eval data-loading path."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from sft_train import _get_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--eval-glob", required=True)
    parser.add_argument("--eval-max-rows", type=int, default=128)
    args = parser.parse_args()
    os.environ["SFT_TRAIN_DATA_GLOB"] = args.train_glob
    os.environ["SFT_EVAL_DATA_GLOB"] = args.eval_glob
    os.environ["SFT_EVAL_MAX_ROWS"] = str(args.eval_max_rows)
    loaded = _get_dataset(SimpleNamespace())
    if len(loaded["train"]) == 0 or len(loaded["test"]) == 0:
        raise ValueError("Missing explicit training or validation examples")
    if "messages" not in loaded["train"].column_names:
        raise ValueError("Training data lacks messages")
    train_prompts = set(loaded["train"]["prompt_sha256"])
    eval_prompts = set(loaded["test"]["prompt_sha256"])
    if train_prompts & eval_prompts:
        raise ValueError("Prompt overlap between training and held-out validation")
    for split in ("train", "test"):
        for row in loaded[split].select(range(min(32, len(loaded[split])))):
            if not row["messages"] or row["messages"][-1]["role"] != "assistant":
                raise ValueError(f"Invalid final assistant turn in {split}")
            if any("<think>" in m["content"] or "</think>" in m["content"]
                   for m in row["messages"] if m["role"] == "assistant"):
                raise ValueError(f"Thinking-tagged source row in {split}")
    print(json.dumps({
        "status": "passed", "train_rows": len(loaded["train"]),
        "eval_rows": len(loaded["test"]), "overlap": 0,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
