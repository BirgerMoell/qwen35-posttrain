#!/usr/bin/env python3
"""Pin and audit the OpenEuroLLM identity data for direct-answer SFT.

This stages a candidate, not a production mixture. Validation is never mixed
into train, and the exact training tokenizer/template must preserve a nonempty
assistant mask without inventing thinking tags.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from audit_two_stage_sft import verify_render


REPO = "birgermoell/openeurollm-model-identity"
REVISION = "9dddd45d932c8bdde12414ea0201ab2d5c6d52ec"
EXPECTED = {
    "train": (1000, "d6220e6ccb7f2f6a1ebae3f7f49c8cf0b76901433b5fe86a9afff7e214a9ca95"),
    "validation": (120, "e0650d94b0bdd693ac96f70445d33391139fdf468a97fdb508f5bac0b97389b5"),
}
LANGUAGES = frozenset("en de es fr it pt pl nl cs ro el sv".split())
ROLES = ("user", "assistant", "user", "assistant")
# A real spelling error seen in the published French preview; these rows need
# correction upstream, not silent rewriting of a third-party dataset.
KNOWN_BAD_TEXT = ("OpenEuroLLL",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_digest(messages: list[dict]) -> str:
    normalized = "\n".join(
        " ".join(message["content"].split()).casefold()
        for message in messages if message["role"] == "user"
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_split(path: Path, split: str, tokenizer, max_length: int) -> tuple[list[dict], Counter]:
    expected_rows, expected_hash = EXPECTED[split]
    if sha256(path) != expected_hash:
        raise ValueError(f"{split} source SHA-256 does not match pinned manifest")
    admitted: list[dict] = []
    counts: Counter = Counter()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            counts["source_rows"] += 1
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id in ids:
                raise ValueError(f"{split}:{line_number} missing or duplicate ID")
            ids.add(row_id)
            if row.get("language") not in LANGUAGES:
                raise ValueError(f"{split}:{line_number} unsupported language")
            if row.get("identity_profile") != "openeurollm-stable-project-identity-v1":
                raise ValueError(f"{split}:{line_number} unexpected identity profile")
            if row.get("license") != "cc-by-4.0":
                raise ValueError(f"{split}:{line_number} unexpected row license")
            messages = row.get("messages")
            if not isinstance(messages, list) or tuple(m.get("role") for m in messages) != ROLES:
                raise ValueError(f"{split}:{line_number} unexpected conversation roles")
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                raise ValueError(f"{split}:{line_number} empty or nontext message")
            if any(bad in m["content"] for bad in KNOWN_BAD_TEXT for m in messages):
                counts["quarantined_known_text_error"] += 1
                continue
            try:
                rendered = verify_render(tokenizer, messages, "instruct", max_length)
            except (KeyError, TypeError, ValueError) as exc:
                counts["quarantined_render_error"] += 1
                print(f"Quarantined {split}:{line_number} {row_id}: {exc}", flush=True)
                continue
            row.update({
                "source": REPO,
                "source_revision": REVISION,
                "source_split": split,
                "prompt_sha256": prompt_digest(messages),
                "rendered_tokens": rendered["tokens"],
                "supervised_tokens": rendered["supervised_tokens"],
            })
            admitted.append(row)
            counts[f"language:{row['language']}"] += 1
            counts[f"category:{row.get('category', 'unknown')}"] += 1
            counts["supervised_tokens"] += rendered["supervised_tokens"]
    if counts["source_rows"] != expected_rows:
        raise ValueError(f"{split} row count {counts['source_rows']} != {expected_rows}")
    return admitted, counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("max-length must be positive")
    info = HfApi().dataset_info(REPO, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected revision: {info.sha}")
    args.source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO, repo_type="dataset", revision=REVISION,
        allow_patterns=["README.md", "manifest.json", "identity_facts.json",
                        "validation_report.json", "data/*.jsonl"],
        local_dir=args.source_dir, max_workers=4,
    )
    source_manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("dataset") != REPO or source_manifest.get("version") != "1.0.0":
        raise ValueError("unexpected source manifest")
    for split, (rows, digest) in EXPECTED.items():
        if source_manifest["splits"][split] != {"rows": rows, "sha256": digest}:
            raise ValueError(f"{split} manifest differs from pinned split")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.chat_template = args.template.read_text(encoding="utf-8")
    accepted = {}
    counts = {}
    for split in EXPECTED:
        accepted[split], counts[split] = audit_split(
            args.source_dir / "data" / f"{split}.jsonl", split, tokenizer, args.max_length
        )

    val_ids = {row["id"] for row in accepted["validation"]}
    val_prompts = {row["prompt_sha256"] for row in accepted["validation"]}
    train = []
    train_prompts: set[str] = set()
    for row in sorted(accepted["train"], key=lambda item: item["id"]):
        if row["id"] in val_ids or row["prompt_sha256"] in val_prompts:
            counts["train"]["quarantined_validation_overlap"] += 1
        elif row["prompt_sha256"] in train_prompts:
            # The release contains different answer phrasings for identical
            # user turns. Keep one deterministic representative per prompt.
            counts["train"]["quarantined_duplicate_train_prompt"] += 1
        else:
            train_prompts.add(row["prompt_sha256"])
            train.append(row)
    accepted["train"] = train

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in accepted.items():
        path = args.output_dir / f"identity_{split}.jsonl"
        write_jsonl(path, rows)
        files[split] = {
            "path": str(path), "rows": len(rows), "sha256": sha256(path),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
        }
    report = {
        "kind": "audited_identity_candidate_not_production_mixture",
        "source": REPO, "revision": REVISION, "source_license": "cc-by-4.0",
        "source_attribution": "OpenEuroLLM Model Identity v1.0.0",
        "identity_profile": source_manifest["identity_profile"],
        "template_sha256": sha256(args.template), "max_length": args.max_length,
        "source_files": {split: {"rows": rows, "sha256": digest} for split, (rows, digest) in EXPECTED.items()},
        "files": files, "counts": {split: dict(counter) for split, counter in counts.items()},
        "release_blockers": [
            "Native-speaker review of translated identity answers is not complete",
            "Evaluate identity accuracy and over-triggering before production inclusion",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
