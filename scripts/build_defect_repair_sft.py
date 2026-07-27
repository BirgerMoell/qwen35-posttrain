#!/usr/bin/env python3
"""Build monolingual SFT repair data for weak European-language generations.

The exported SFT rows intentionally keep the strict training schema:

{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "lang": "is"}

Source provenance, license, and quality metadata are written as sidecars so the
training pipeline can consume the JSONL directly while data reviewers can still
filter or audit by source/license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

DATASET_ID = "oellm-eu-defect-repair-sft-v1"
VERSION = "v0.1.0"
DEFAULT_OUT = Path("data/defect_repair/oellm-eu-defect-repair-sft-v1")

WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿĀ-žƀ-ɏΆ-ώЀ-ӿ]+\b", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
FOUR_GRAM_LIMIT = 3

WIKIMEDIA_SOURCE = {
    "source_id": "wikimedia_wikipedia_20231101",
    "name": "Wikimedia Wikipedia 2023-11-01 dumps",
    "source_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    "source_license": "CC-BY-SA-4.0 / GFDL under Wikimedia Terms of Use; verify imported-page attribution when needed",
    "license_id": "cc-by-sa-4.0",
    "license_category": "sharealike",
    "license_filter_tags": ["attribution_required", "sharealike", "native_wikipedia"],
    "redistribution_status": "redistributable_sharealike",
}

STOP_HEADINGS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "notes",
    "sources",
    "tenglar",
    "heimildir",
    "sjá einnig",
    "foinsí",
    "tagairtí",
    "naisc sheachtracha",
    "ara wkoll",
    "referenzi",
    "lingi esterni",
    "viited",
    "välislingid",
    "vaata ka",
    "izvori",
    "literatura",
    "vanjske poveznice",
    "glej tudi",
    "viri",
    "zunanje povezave",
    "šaltiniai",
    "nuorodos",
    "taip pat skaitykite",
    "avoti",
    "ārējās saites",
    "skatīt arī",
    "kilder",
    "eksterne henvisninger",
    "se også",
    "források",
    "külső hivatkozások",
    "lásd még",
    "zdroje",
    "externé odkazy",
    "pozri aj",
    "източници",
    "външни препратки",
    "вижте също",
    "bibliografie",
    "note",
    "legături externe",
    "zobacz też",
    "przypisy",
    "linki zewnętrzne",
    "lähteet",
    "aiheesta muualla",
    "katso myös",
}

BAD_TITLE_FRAGMENTS = {
    "wikipedia",
    "wikimedia",
    "commons",
    "template:",
    "category:",
    "flokkur:",
    "catagóir:",
    "kategorija:",
    "категория:",
    "kategoria:",
    "luokka:",
    "il-paġna prinċipali",
}

RUSSIAN_MARKERS = {
    "вопрос",
    "ответ",
    "что",
    "это",
    "если",
    "который",
    "которые",
    "является",
    "может быть",
}


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    tier: str
    target: int
    wiki_config: str
    prompt_templates: tuple[str, ...]
    marker_chars: str
    script: str
    max_chunks_per_article: int = 2


LANGUAGES: dict[str, LanguageSpec] = {
    "is": LanguageSpec(
        "is",
        "Icelandic",
        "tier1_floor",
        8000,
        "20231101.is",
        (
            "Útskýrðu helstu atriðin um {title}.",
            "Hvað er mikilvægt að vita um {title}?",
            "Gefðu stutt en efnisríkt yfirlit um {title}.",
            "Skrifaðu fræðandi svar um {title}.",
            "Segðu mér frá {title} á skýran hátt.",
        ),
        "þðæöáéíóúý",
        "latin",
    ),
    "ga": LanguageSpec(
        "ga",
        "Irish",
        "tier1_floor",
        8000,
        "20231101.ga",
        (
            "Mínigh na príomhphointí faoi {title}.",
            "Cad is fiú a fhios faoi {title}?",
            "Tabhair achoimre shoiléir ar {title}.",
            "Scríobh freagra eolasach faoi {title}.",
            "Inis dom faoi {title} i nGaeilge nádúrtha.",
        ),
        "áéíóúÁÉÍÓÚ",
        "latin",
    ),
    "mt": LanguageSpec(
        "mt",
        "Maltese",
        "tier1_floor",
        8000,
        "20231101.mt",
        (
            "Spjega l-punti ewlenin dwar {title}.",
            "X'għandu wieħed ikun jaf dwar {title}?",
            "Agħti ħarsa ġenerali ċara dwar {title}.",
            "Ikteb tweġiba informattiva dwar {title}.",
            "Għidli dwar {title} b'mod naturali.",
        ),
        "ċġħżĊĠĦŻ",
        "latin",
        3,
    ),
    "et": LanguageSpec(
        "et",
        "Estonian",
        "tier1_floor",
        8000,
        "20231101.et",
        (
            "Selgita peamisi asjaolusid teema „{title}“ kohta.",
            "Mida on oluline teada teemast „{title}“?",
            "Anna selge ja sisukas ülevaade teemast „{title}“.",
            "Kirjuta hariv vastus teemal „{title}“.",
            "Räägi teemast „{title}“ loomulikus eesti keeles.",
        ),
        "õäöüšž",
        "latin",
    ),
    "hr": LanguageSpec(
        "hr",
        "Croatian",
        "tier2_small_model",
        3000,
        "20231101.hr",
        (
            "Objasni glavne činjenice o temi „{title}“.",
            "Što je važno znati o temi „{title}“?",
            "Napiši jasan i sadržajan pregled teme „{title}“.",
            "Daj informativan odgovor o temi „{title}“.",
            "Predstavi temu „{title}“ prirodnim hrvatskim jezikom.",
        ),
        "čćđšž",
        "latin",
    ),
    "sl": LanguageSpec(
        "sl",
        "Slovenian",
        "tier2_small_model",
        3000,
        "20231101.sl",
        (
            "Pojasni glavna dejstva o temi „{title}“.",
            "Kaj je pomembno vedeti o temi „{title}“?",
            "Napiši jasen in vsebinski pregled teme „{title}“.",
            "Podaj informativen odgovor o temi „{title}“.",
            "Predstavi temo „{title}“ v naravni slovenščini.",
        ),
        "čšž",
        "latin",
    ),
    "lt": LanguageSpec(
        "lt",
        "Lithuanian",
        "tier2_small_model",
        3000,
        "20231101.lt",
        (
            "Paaiškink svarbiausius dalykus apie „{title}“.",
            "Ką svarbu žinoti apie „{title}“?",
            "Pateik aiškią ir turiningą „{title}“ apžvalgą.",
            "Parašyk informatyvų atsakymą apie „{title}“.",
            "Papasakok apie „{title}“ natūralia lietuvių kalba.",
        ),
        "ąčęėįšųūž",
        "latin",
    ),
    "lv": LanguageSpec(
        "lv",
        "Latvian",
        "tier2_small_model",
        3000,
        "20231101.lv",
        (
            "Izskaidro galvenos lietas par „{title}“.",
            "Kas ir svarīgi jāzina par „{title}“?",
            "Sniedz skaidru un saturīgu pārskatu par „{title}“.",
            "Uzraksti informatīvu atbildi par „{title}“.",
            "Pastāsti par „{title}“ dabiskā latviešu valodā.",
        ),
        "āčēģīķļņšūž",
        "latin",
    ),
    "da": LanguageSpec(
        "da",
        "Danish",
        "tier2_small_model",
        3000,
        "20231101.da",
        (
            "Forklar de vigtigste forhold om {title}.",
            "Hvad er vigtigt at vide om {title}?",
            "Giv et klart og indholdsrigt overblik over {title}.",
            "Skriv et informativt svar om {title}.",
            "Fortæl om {title} på naturligt dansk.",
        ),
        "æøå",
        "latin",
    ),
    "hu": LanguageSpec(
        "hu",
        "Hungarian",
        "tier2_small_model",
        3000,
        "20231101.hu",
        (
            "Magyarázd el a legfontosabb tudnivalókat erről: {title}.",
            "Mit érdemes tudni erről: {title}?",
            "Adj világos és tartalmas áttekintést erről: {title}.",
            "Írj informatív választ erről: {title}.",
            "Mutasd be természetes magyar nyelven ezt a témát: {title}.",
        ),
        "áéíóöőúüű",
        "latin",
    ),
    "sk": LanguageSpec(
        "sk",
        "Slovak",
        "tier2_small_model",
        3000,
        "20231101.sk",
        (
            "Vysvetli hlavné fakty o téme „{title}“.",
            "Čo je dôležité vedieť o téme „{title}“?",
            "Napíš jasný a vecný prehľad témy „{title}“.",
            "Daj informatívnu odpoveď o téme „{title}“.",
            "Predstav tému „{title}“ prirodzenou slovenčinou.",
        ),
        "áäčďéíĺľňóôŕšťúýž",
        "latin",
    ),
    "bg": LanguageSpec(
        "bg",
        "Bulgarian",
        "tier3_anti_russian_leak",
        2000,
        "20231101.bg",
        (
            "Обяснете основните неща за {title}.",
            "Какво е важно да се знае за {title}?",
            "Дайте кратък, но ясен обзор на {title}.",
            "Напишете съдържателен отговор за {title}.",
            "Представете темата {title} на естествен български език.",
        ),
        "ъщѝ",
        "cyrillic",
    ),
    "ro": LanguageSpec(
        "ro",
        "Romanian",
        "tier4_touchup",
        1000,
        "20231101.ro",
        (
            "Explică principalele lucruri despre „{title}“.",
            "Ce este important de știut despre „{title}“?",
            "Oferă o prezentare clară și substanțială despre „{title}“.",
            "Scrie un răspuns informativ despre „{title}“.",
            "Prezintă tema „{title}“ în limba română naturală.",
        ),
        "ăâîșț",
        "latin",
    ),
    "pl": LanguageSpec(
        "pl",
        "Polish",
        "tier4_touchup",
        1000,
        "20231101.pl",
        (
            "Wyjaśnij najważniejsze informacje na temat „{title}“.",
            "Co warto wiedzieć o temacie „{title}“?",
            "Przedstaw jasny i treściwy opis tematu „{title}“.",
            "Napisz informacyjną odpowiedź o temacie „{title}“.",
            "Opowiedz o temacie „{title}“ naturalną polszczyzną.",
        ),
        "ąćęłńóśźż",
        "latin",
    ),
    "fi": LanguageSpec(
        "fi",
        "Finnish",
        "tier4_touchup",
        1000,
        "20231101.fi",
        (
            "Selitä tärkeimmät asiat aiheesta {title}.",
            "Mitä aiheesta {title} on hyvä tietää?",
            "Anna selkeä ja sisällöllinen yleiskuva aiheesta {title}.",
            "Kirjoita informatiivinen vastaus aiheesta {title}.",
            "Kerro aiheesta {title} luontevalla suomen kielellä.",
        ),
        "äöå",
        "latin",
    ),
}


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("„", '"')
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def is_bad_title(title: str) -> bool:
    lowered = title.strip().lower()
    if not lowered or len(lowered) < 2:
        return True
    return any(fragment in lowered for fragment in BAD_TITLE_FRAGMENTS)


def looks_like_heading(line: str) -> bool:
    stripped = line.strip().strip("=:")
    lowered = stripped.lower()
    if lowered in STOP_HEADINGS:
        return True
    if len(stripped) <= 45 and stripped and not re.search(r"[.!?。؟]$", stripped):
        words = count_words(stripped)
        return 1 <= words <= 5
    return False


def paragraphs_from_text(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in unicodedata.normalize("NFC", text).splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(normalize_text(" ".join(current)))
                current = []
            continue
        clean_heading = line.strip().strip("=:").lower()
        if clean_heading in STOP_HEADINGS:
            break
        if URL_RE.search(line):
            continue
        if looks_like_heading(line):
            if current:
                paragraphs.append(normalize_text(" ".join(current)))
                current = []
            continue
        if "wikipedia" in line.lower() and count_words(line) < 40:
            continue
        current.append(line)
    if current:
        paragraphs.append(normalize_text(" ".join(current)))
    return [p for p in paragraphs if count_words(p) >= 30]


def build_chunks(text: str, min_words: int, max_words: int, max_chunks: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs_from_text(text):
        paragraph_words = count_words(paragraph)
        if paragraph_words > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                sentence = normalize_text(sentence)
                sentence_words = count_words(sentence)
                if sentence_words < 8:
                    continue
                if current_words + sentence_words > max_words and current_words >= min_words:
                    chunks.append(normalize_text(" ".join(current)))
                    current = []
                    current_words = 0
                current.append(sentence)
                current_words += sentence_words
                if len(chunks) >= max_chunks:
                    return chunks
            continue

        if current_words + paragraph_words > max_words and current_words >= min_words:
            chunks.append(normalize_text(" ".join(current)))
            current = []
            current_words = 0
            if len(chunks) >= max_chunks:
                return chunks

        if paragraph_words <= max_words:
            current.append(paragraph)
            current_words += paragraph_words

    if current_words >= min_words and len(chunks) < max_chunks:
        chunks.append(normalize_text(" ".join(current)))
    return chunks[:max_chunks]


def repeated_ngrams(text: str, n: int = 4, limit: int = FOUR_GRAM_LIMIT) -> bool:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < n:
        return False
    counts: Counter[tuple[str, ...]] = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(count > limit for count in counts.values())


def char_ratio(text: str, script: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if script == "cyrillic":
        matching = [c for c in letters if "CYRILLIC" in unicodedata.name(c, "")]
    else:
        matching = [
            c
            for c in letters
            if "LATIN" in unicodedata.name(c, "")
            or c in "áéíóúýþðæöÁÉÍÓÚÝÞÐÆÖċġħżĊĠĦŻõäöüšžÕÄÖÜŠŽ"
        ]
    return len(matching) / len(letters)


def contains_marker_chars(text: str, markers: str) -> bool:
    lowered = text.lower()
    return any(ch.lower() in lowered for ch in markers)


def has_bulgarian_russian_leak(text: str) -> bool:
    lowered = text.lower()
    if any(ch in lowered for ch in ("ы", "э", "ё")):
        return True
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in RUSSIAN_MARKERS)


def quality_rejection(lang: LanguageSpec, prompt: str, answer: str, min_words: int, max_words: int) -> str | None:
    answer_words = count_words(answer)
    if answer_words < min_words:
        return "assistant_too_short"
    if answer_words > max_words:
        return "assistant_too_long"
    if repeated_ngrams(answer):
        return "repeated_4gram"
    if URL_RE.search(answer):
        return "contains_url"
    if char_ratio(prompt + " " + answer, lang.script) < 0.65:
        return "script_ratio_low"
    if lang.marker_chars and not contains_marker_chars(prompt + " " + answer, lang.marker_chars):
        return "missing_native_marker_chars"
    if lang.code == "bg" and has_bulgarian_russian_leak(prompt + " " + answer):
        return "bg_russian_marker"
    return None


def make_prompt(lang: LanguageSpec, title: str, index: int) -> tuple[str, int]:
    template_id = index % len(lang.prompt_templates)
    template = lang.prompt_templates[template_id]
    return normalize_text(template.format(title=title)), template_id


def make_row_id(lang: str, source_record_id: str, chunk_index: int, prompt_template_id: int, answer: str) -> str:
    digest = hashlib.sha256(f"{lang}|{source_record_id}|{chunk_index}|{prompt_template_id}|{answer}".encode()).hexdigest()[:16]
    return f"{DATASET_ID}_{lang}_{digest}"


def iter_wikipedia_rows(lang: LanguageSpec) -> Iterable[dict]:
    dataset = load_dataset("wikimedia/wikipedia", lang.wiki_config, split="train", streaming=True)
    yield from dataset


def split_name(rng: random.Random, validation_fraction: float) -> str:
    return "validation" if rng.random() < validation_fraction else "train"


def strict_sft_row(prompt: str, answer: str, lang: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "lang": lang,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def render_readme(out_dir: Path, manifest: dict) -> str:
    counts = manifest["counts_by_language"]
    count_lines = "\n".join(f"- `{lang}`: {count}" for lang, count in sorted(counts.items()))
    return f"""---
dataset_info:
  pretty_name: OpenEuroLLM Defect Repair SFT v1
  license: cc-by-sa-4.0
---

# {DATASET_ID}

Monolingual SFT repair data for European-language generation defects observed
after Qwen 2B/4B/9B post-training. This dataset is designed to repair
degeneration, repetition loops, short answers, morphology damage, and
Bulgarian/Russian language leakage.

## Strict SFT Schema

Each row in `data/*.jsonl` uses exactly:

```json
{{"messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}],"lang":"is"}}
```

Provenance, source URL, license, source id, quality statistics, and hashes are
kept in `metadata/all_metadata.jsonl`, `source_registry.json`,
`metadata/quality_report.json`, and `manifest.json`.

## Source Policy

Version `{VERSION}` is built from native Wikipedia text via
`wikimedia/wikipedia` 2023-11-01 snapshots. Rows are not raw machine
translations. Prompt templates are short target-language instructions and
assistant turns are human-authored native article passages.

License: Wikimedia text is redistributed under CC-BY-SA terms, with attribution
and share-alike requirements. Keep the sidecar provenance files when using or
redistributing filtered subsets.

## Counts

{count_lines}

Total strict SFT rows: `{manifest["total_rows"]}`.

## Intended Use

Use as an upweighted repair shard in SFT:

- Tier 1 (`is`, `ga`, `mt`, `et`): high upweight for fluency and answer length.
- Tier 2 (`hr`, `sl`, `lt`, `lv`, `da`, `hu`, `sk`): medium upweight for loop repair.
- `bg`: anti-Russian leakage shard with Russian-marker filters.
- Tier 4 (`ro`, `pl`, `fi`): light touch-up.

Run `scripts/validate_defect_repair_sft.py --dataset-dir {out_dir}` before
packing to parquet or mixing into SFT.
"""


def build_dataset(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    out_dir: Path = args.out
    target_scale = args.target_scale
    selected_langs = args.languages or list(LANGUAGES)

    data_rows: dict[str, list[dict]] = {"train": [], "validation": []}
    metadata_rows: list[dict] = []
    counters = {
        "accepted": Counter(),
        "seen_articles": Counter(),
        "rejections": defaultdict(Counter),
        "assistant_words": defaultdict(list),
    }
    seen_answer_hashes: set[str] = set()

    for code in selected_langs:
        lang = LANGUAGES[code]
        target = max(1, int(lang.target * target_scale))
        max_source_rows = args.max_source_rows_per_language or target * args.source_scan_multiplier
        print(f"[{code}] target={target} source_scan_limit={max_source_rows}")

        try:
            source_iter = iter_wikipedia_rows(lang)
            for source_index, source_row in enumerate(source_iter):
                if counters["accepted"][code] >= target:
                    break
                if source_index >= max_source_rows:
                    break

                counters["seen_articles"][code] += 1
                title = normalize_text(str(source_row.get("title") or ""))
                if is_bad_title(title):
                    counters["rejections"][code]["bad_title"] += 1
                    continue
                text = str(source_row.get("text") or "")
                chunks = build_chunks(text, args.min_assistant_words, args.max_assistant_words, lang.max_chunks_per_article)
                if not chunks:
                    counters["rejections"][code]["no_usable_chunk"] += 1
                    continue

                for chunk_index, answer in enumerate(chunks):
                    if counters["accepted"][code] >= target:
                        break
                    prompt, template_id = make_prompt(lang, title, counters["accepted"][code] + chunk_index)
                    rejection = quality_rejection(lang, prompt, answer, args.min_assistant_words, args.max_assistant_words)
                    if rejection:
                        counters["rejections"][code][rejection] += 1
                        continue

                    answer_hash = sha256_text(answer)
                    if answer_hash in seen_answer_hashes:
                        counters["rejections"][code]["duplicate_answer"] += 1
                        continue
                    seen_answer_hashes.add(answer_hash)

                    row_id = make_row_id(code, str(source_row.get("id") or source_index), chunk_index, template_id, answer)
                    split = split_name(rng, args.validation_fraction)
                    strict = strict_sft_row(prompt, answer, code)
                    data_rows[split].append(strict)
                    counters["accepted"][code] += 1
                    counters["assistant_words"][code].append(count_words(answer))
                    metadata_rows.append(
                        {
                            "id": row_id,
                            "dataset_id": DATASET_ID,
                            "version": VERSION,
                            "split": split,
                            "lang": code,
                            "language_name": lang.name,
                            "tier": lang.tier,
                            "source_id": WIKIMEDIA_SOURCE["source_id"],
                            "source_record_id": str(source_row.get("id") or source_index),
                            "source_url": source_row.get("url"),
                            "source_title": title,
                            "source_config": lang.wiki_config,
                            "source_license": WIKIMEDIA_SOURCE["source_license"],
                            "license_id": WIKIMEDIA_SOURCE["license_id"],
                            "license_category": WIKIMEDIA_SOURCE["license_category"],
                            "license_filter_tags": WIKIMEDIA_SOURCE["license_filter_tags"],
                            "redistribution_status": WIKIMEDIA_SOURCE["redistribution_status"],
                            "prompt_template_id": template_id,
                            "assistant_word_count": count_words(answer),
                            "provenance_hash": sha256_text(prompt + "\n\n" + answer),
                        }
                    )
        except Exception as exc:  # keep partial artifact useful if one language source fails
            counters["rejections"][code]["source_error"] += 1
            print(f"[{code}] ERROR {type(exc).__name__}: {exc}")

        print(f"[{code}] accepted={counters['accepted'][code]} seen={counters['seen_articles'][code]}")

    all_rows = data_rows["train"] + data_rows["validation"]
    write_jsonl(out_dir / "data" / "train.jsonl", data_rows["train"])
    write_jsonl(out_dir / "data" / "validation.jsonl", data_rows["validation"])
    write_jsonl(out_dir / "data" / "all.jsonl", all_rows)
    write_jsonl(out_dir / "metadata" / "all_metadata.jsonl", metadata_rows)

    counts_by_language = Counter(row["lang"] for row in all_rows)
    counts_by_split = {
        "train": len(data_rows["train"]),
        "validation": len(data_rows["validation"]),
    }
    word_stats = {}
    for code, values in counters["assistant_words"].items():
        if values:
            word_stats[code] = {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 2),
            }

    manifest = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "schema": "chat_messages_strict_sft",
        "created_by": Path(__file__).name,
        "total_rows": len(all_rows),
        "counts_by_split": counts_by_split,
        "counts_by_language": dict(sorted(counts_by_language.items())),
        "target_counts_by_language": {
            code: max(1, int(LANGUAGES[code].target * target_scale)) for code in selected_langs
        },
        "source_ids": [WIKIMEDIA_SOURCE["source_id"]],
        "license_categories": [WIKIMEDIA_SOURCE["license_category"]],
        "strict_export_files": ["data/train.jsonl", "data/validation.jsonl", "data/all.jsonl"],
        "metadata_files": ["metadata/all_metadata.jsonl", "metadata/quality_report.json"],
    }
    quality_report = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "parameters": {
            "min_assistant_words": args.min_assistant_words,
            "max_assistant_words": args.max_assistant_words,
            "validation_fraction": args.validation_fraction,
            "target_scale": args.target_scale,
            "source_scan_multiplier": args.source_scan_multiplier,
            "max_source_rows_per_language": args.max_source_rows_per_language,
        },
        "accepted_by_language": dict(sorted(counters["accepted"].items())),
        "seen_articles_by_language": dict(sorted(counters["seen_articles"].items())),
        "rejections_by_language": {
            code: dict(sorted(reason_counts.items())) for code, reason_counts in sorted(counters["rejections"].items())
        },
        "assistant_word_stats_by_language": word_stats,
        "hard_filters": [
            "assistant_80_to_400_words_by_default",
            "no_4gram_repeated_more_than_3_times",
            "target_script_ratio",
            "language_marker_char_presence",
            "bg_russian_marker_ban",
            "url_ban",
        ],
    }

    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "source_registry.json").write_text(
        json.dumps({WIKIMEDIA_SOURCE["source_id"]: WIKIMEDIA_SOURCE}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metadata" / "quality_report.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(render_readme(out_dir, manifest), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--languages", nargs="*", choices=sorted(LANGUAGES))
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--min-assistant-words", type=int, default=80)
    parser.add_argument("--max-assistant-words", type=int, default=400)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--source-scan-multiplier", type=int, default=18)
    parser.add_argument("--max-source-rows-per-language", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    sys.stdout.flush()
    # The HF streaming stack can leave cleanup threads alive after all rows have
    # been written. Exit explicitly so long LUMI/data-build runs do not hang.
    os._exit(0)


if __name__ == "__main__":
    main()
