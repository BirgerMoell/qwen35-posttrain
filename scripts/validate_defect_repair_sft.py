#!/usr/bin/env python3
"""Validate oellm-eu-defect-repair-sft-v1 strict SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_DIR = Path("data/defect_repair/oellm-eu-defect-repair-sft-v1")
WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿĀ-žƀ-ɏΆ-ώЀ-ӿ]+\b", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

LANG_MARKERS = {
    "is": "þðæöáéíóúý",
    "ga": "áéíóúÁÉÍÓÚ",
    "mt": "ċġħżĊĠĦŻ",
    "et": "õäöüšž",
    "hr": "čćđšž",
    "sl": "čšž",
    "lt": "ąčęėįšųūž",
    "lv": "āčēģīķļņšūž",
    "da": "æøå",
    "hu": "áéíóöőúüű",
    "sk": "áäčďéíĺľňóôŕšťúýž",
    "bg": "ъщѝ",
    "ro": "ăâîșț",
    "pl": "ąćęłńóśźż",
    "fi": "äöå",
}
CYRILLIC_LANGS = {"bg"}
RUSSIAN_CHARS = {"ы", "э", "ё"}
RUSSIAN_WORD_MARKERS = {"вопрос", "ответ", "что", "это", "который", "является"}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: {exc}") from exc


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def repeated_ngrams(text: str, n: int = 4, limit: int = 3) -> bool:
    words = [w.lower() for w in WORD_RE.findall(text)]
    counts = Counter(tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))
    return any(value > limit for value in counts.values())


def char_ratio(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang in CYRILLIC_LANGS:
        matching = [c for c in letters if "CYRILLIC" in unicodedata.name(c, "")]
    else:
        matching = [c for c in letters if "LATIN" in unicodedata.name(c, "")]
    return len(matching) / len(letters)


def has_marker(text: str, lang: str) -> bool:
    markers = LANG_MARKERS.get(lang, "")
    lowered = text.lower()
    return not markers or any(ch.lower() in lowered for ch in markers)


def has_bg_russian_leak(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in RUSSIAN_CHARS):
        return True
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in RUSSIAN_WORD_MARKERS)


def validate_row(row: dict, split: str, line_no: int, min_words: int, max_words: int, errors: list[str], summary: dict) -> None:
    prefix = f"{split}:{line_no}"
    if set(row) != {"messages", "lang"}:
        errors.append(f"{prefix}: row must contain exactly messages/lang")
        return
    lang = row.get("lang")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        errors.append(f"{prefix}: messages must contain exactly two turns")
        return
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        errors.append(f"{prefix}: roles must be user/assistant")
        return
    prompt = str(messages[0].get("content") or "")
    answer = str(messages[1].get("content") or "")
    if not prompt or not answer:
        errors.append(f"{prefix}: empty prompt or answer")
        return
    words = count_words(answer)
    if words < min_words or words > max_words:
        errors.append(f"{prefix}: assistant has {words} words, expected {min_words}-{max_words}")
    if URL_RE.search(prompt + " " + answer):
        errors.append(f"{prefix}: contains URL")
    if repeated_ngrams(answer):
        errors.append(f"{prefix}: repeated 4-gram over limit")
    if char_ratio(prompt + " " + answer, str(lang)) < 0.65:
        errors.append(f"{prefix}: low target-script ratio")
    if not has_marker(prompt + " " + answer, str(lang)):
        errors.append(f"{prefix}: missing native marker characters for {lang}")
    if lang == "bg" and has_bg_russian_leak(prompt + " " + answer):
        errors.append(f"{prefix}: Bulgarian row contains Russian marker")
    summary["rows"] += 1
    summary["languages"][lang] += 1
    summary["splits"][split] += 1
    summary["assistant_words"][lang].append(words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--min-assistant-words", type=int, default=80)
    parser.add_argument("--max-assistant-words", type=int, default=400)
    parser.add_argument("--max-errors", type=int, default=100)
    args = parser.parse_args()

    summary = {"rows": 0, "languages": Counter(), "splits": Counter(), "assistant_words": {}}
    summary["assistant_words"] = {lang: [] for lang in LANG_MARKERS}
    errors: list[str] = []

    for split in ["train", "validation"]:
        path = args.dataset_dir / "data" / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        for line_no, row in iter_jsonl(path):
            validate_row(row, split, line_no, args.min_assistant_words, args.max_assistant_words, errors, summary)

    if errors:
        for error in errors[: args.max_errors]:
            print(f"ERROR: {error}")
        if len(errors) > args.max_errors:
            print(f"... {len(errors) - args.max_errors} more errors")
        raise SystemExit(1)

    printable = {
        "rows": summary["rows"],
        "languages": dict(sorted(summary["languages"].items())),
        "splits": dict(sorted(summary["splits"].items())),
        "assistant_word_stats": {
            lang: {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 2),
            }
            for lang, values in sorted(summary["assistant_words"].items())
            if values
        },
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
