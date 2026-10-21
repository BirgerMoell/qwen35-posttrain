#!/usr/bin/env python3
"""Package the published Excellent-SFT token protocol for a 256K base warm start.

Only reserved token IDs are renamed. This never edits the base weights or adds
embedding rows. Supply pinned, locally downloaded tokenizer.json files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REMAPPED_IDS = {
    3: ("<start_of_turn>", "<|im_start|>"),
    4: ("<end_of_turn>", "<|im_end|>"),
    13: ("<unused_0>", "<tools>"),
    14: ("<unused_1>", "</tools>"),
    15: ("<unused_2>", "<tool_response>"),
    16: ("<unused_3>", "</tool_response>"),
    17: ("<unused_4>", "</think>"),
    18: ("<unused_5>", "<think>"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer-json", type=Path, required=True)
    parser.add_argument("--reference-tokenizer-json", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    base = Tokenizer.from_file(str(args.base_tokenizer_json))
    reference = Tokenizer.from_file(str(args.reference_tokenizer_json))
    if base.get_vocab_size(with_added_tokens=True) != reference.get_vocab_size(with_added_tokens=True):
        raise ValueError("base and reference tokenizers have different vocabulary sizes")
    for token_id in range(base.get_vocab_size(with_added_tokens=True)):
        before, after = base.id_to_token(token_id), reference.id_to_token(token_id)
        if token_id in REMAPPED_IDS:
            if (before, after) != REMAPPED_IDS[token_id]:
                raise ValueError(f"unexpected reserved-token mapping at ID {token_id}: {before!r}, {after!r}")
        elif before != after:
            raise ValueError(f"ordinary token ID {token_id} changed: {before!r}, {after!r}")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.reference_tokenizer_json),
        unk_token="<unk>", bos_token="<bos>", eos_token="<|im_end|>", pad_token="<pad>",
        model_max_length=262144,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = args.template.read_text(encoding="utf-8")
    if len(tokenizer) != 262145:
        raise ValueError(f"unexpected tokenizer length {len(tokenizer)}")
    for token_id, (_, token) in REMAPPED_IDS.items():
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"{token!r} did not encode as reserved ID {token_id}")
    if tokenizer.eos_token_id != 4 or tokenizer.pad_token_id != 262144:
        raise ValueError("chat EOS or padding ID is wrong")

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(args.output))
    loaded = AutoTokenizer.from_pretrained(str(args.output), local_files_only=True)
    if loaded.eos_token_id != 4 or len(loaded) != len(tokenizer):
        raise ValueError("saved tokenizer failed reload verification")
    print(json.dumps({
        "output": str(args.output),
        "base_tokenizer_json_sha256": sha256(args.base_tokenizer_json),
        "reference_tokenizer_json_sha256": sha256(args.reference_tokenizer_json),
        "saved_tokenizer_json_sha256": sha256(args.output / "tokenizer.json"),
        "template_sha256": sha256(args.template),
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "remapped_ids": REMAPPED_IDS,
    }, indent=2))


if __name__ == "__main__":
    main()
