#!/usr/bin/env python3
"""Build real-source MCQ exam data for GRPO/RLVR and DPO.

Default source:
- EXAMS QA repository (`mhardalov/exams-qa`), CC-BY-SA-4.0.

Optional source:
- local Swedish Medical Benchmark GRPO JSONL rows, only when explicitly enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DATASET_ID = "oellm-eu-exam-mcq-v1"
VERSION = "v0.1.0"
DEFAULT_OUT = Path("data/exam_mcq/oellm-eu-exam-mcq-v1")
DEFAULT_SOURCE_REGISTRY = Path("data/exam_mcq/source_registry.json")
DEFAULT_EXAMS_REPO = Path("/private/tmp/exams-qa")
DEFAULT_SWEDISH_GRPO = Path("/Users/birgermoell/AI-Sweden/swedish-medical-benchmark/data/grpo")

LANGUAGE_NAME_TO_ISO = {
    "Albanian": "sq",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "French": "fr",
    "German": "de",
    "Hungarian": "hu",
    "Italian": "it",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Polish": "pl",
    "Portuguese": "pt",
    "Serbian": "sr",
    "Spanish": "es",
    "Turkish": "tr",
    "Arabic": "ar",
    "Vietnamese": "vi",
}

EUROPEAN_TARGET_LANGS = {
    "sq",
    "bg",
    "hr",
    "fr",
    "de",
    "hu",
    "it",
    "lt",
    "mk",
    "pl",
    "pt",
    "sr",
    "es",
    "tr",
    "sv",
}

ANSWER_RE = re.compile(r"^\s*([A-Z])\s*[\).:]?\s*$")

PROMPT_TEXT = {
    "bg": ("Изберете най-добрия отговор. Отговорете само с буквата.", "Въпрос", "Варианти", "Отговор"),
    "de": ("Wählen Sie die beste Antwortoption. Antworten Sie nur mit dem Buchstaben.", "Frage", "Antwortoptionen", "Antwort"),
    "es": ("Elige la mejor opción de respuesta. Responde solo con la letra.", "Pregunta", "Opciones", "Respuesta"),
    "fr": ("Choisissez la meilleure réponse. Répondez uniquement par la lettre.", "Question", "Options", "Réponse"),
    "hr": ("Odaberite najbolji odgovor. Odgovorite samo slovom.", "Pitanje", "Mogućnosti", "Odgovor"),
    "hu": ("Válassza ki a legjobb választ. Csak a betűjellel válaszoljon.", "Kérdés", "Válaszlehetőségek", "Válasz"),
    "it": ("Scegli l'opzione di risposta migliore. Rispondi solo con la lettera.", "Domanda", "Opzioni", "Risposta"),
    "lt": ("Pasirinkite geriausią atsakymo variantą. Atsakykite tik raide.", "Klausimas", "Pasirinkimai", "Atsakymas"),
    "mk": ("Изберете го најдобриот одговор. Одговорете само со буквата.", "Прашање", "Опции", "Одговор"),
    "pl": ("Wybierz najlepszą odpowiedź. Odpowiedz tylko literą.", "Pytanie", "Opcje", "Odpowiedź"),
    "pt": ("Escolha a melhor opção de resposta. Responda apenas com a letra.", "Pergunta", "Opções", "Resposta"),
    "sq": ("Zgjidhni përgjigjen më të mirë. Përgjigjuni vetëm me shkronjën.", "Pyetje", "Alternativa", "Përgjigje"),
    "sr": ("Изаберите најбољи одговор. Одговорите само словом.", "Питање", "Опције", "Одговор"),
    "sv": ("Välj det bästa svarsalternativet. Svara endast med bokstaven.", "Fråga", "Svarsalternativ", "Svar"),
    "tr": ("En iyi yanıt seçeneğini seçin. Yalnızca harfle cevap verin.", "Soru", "Seçenekler", "Cevap"),
}


@dataclass
class McqRow:
    id: str
    dataset_id: str
    version: str
    split: str
    source_id: str
    source_url: str
    source_license: str
    redistribution_status: str
    source_record_id: str
    language: str
    language_name: str
    domain: str
    subject: str
    grade: str | int | None
    task_type: str
    question: str
    options: list[dict]
    answer: str
    answer_text: str
    prompt: str
    reward_type: str
    provenance_hash: str


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def normalize_ws(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def prompt_for_mcq(question: str, options: list[dict], language: str) -> str:
    option_lines = "\n".join(f"{opt['label']}) {opt['text']}" for opt in options)
    instruction, question_label, options_label, answer_label = PROMPT_TEXT.get(
        language, ("Choose the best answer option. Answer with the letter only.", "Question", "Options", "Answer")
    )
    return f"{instruction}\n\n{question_label}:\n{question}\n\n{options_label}:\n{option_lines}\n\n{answer_label}:"


def row_to_dpo_pairs(row: McqRow) -> Iterable[dict]:
    for option in row.options:
        rejected = option["label"]
        if rejected == row.answer:
            continue
        yield {
            "id": f"{row.id}_dpo_reject_{rejected}",
            "dataset_id": DATASET_ID,
            "version": VERSION,
            "split": row.split,
            "source_id": row.source_id,
            "source_record_id": row.source_record_id,
            "language": row.language,
            "domain": row.domain,
            "subject": row.subject,
            "prompt": row.prompt,
            "chosen": row.answer,
            "rejected": rejected,
            "chosen_text": row.answer_text,
            "rejected_text": option["text"],
            "preference_type": "mcq_correct_over_incorrect",
            "source_license": row.source_license,
            "redistribution_status": row.redistribution_status,
            "provenance_hash": row.provenance_hash,
        }


def read_tar_jsonl(path: Path) -> Iterable[dict]:
    with tarfile.open(path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if len(members) != 1:
            raise ValueError(f"Expected one file in {path}, found {len(members)}")
        extracted = tar.extractfile(members[0])
        if extracted is None:
            return
        for raw in extracted:
            if raw.strip():
                yield json.loads(raw.decode("utf-8"))


def convert_exams_row(raw: dict, split: str) -> McqRow | None:
    info = raw.get("info") or {}
    language_name = normalize_ws(info.get("language"))
    language = LANGUAGE_NAME_TO_ISO.get(language_name)
    if not language or language not in EUROPEAN_TARGET_LANGS:
        return None
    question_obj = raw.get("question") or {}
    question = normalize_ws(question_obj.get("stem"))
    choices = question_obj.get("choices") or []
    options = [
        {"label": normalize_ws(choice.get("label")).upper(), "text": normalize_ws(choice.get("text"))}
        for choice in choices
        if normalize_ws(choice.get("label")) and normalize_ws(choice.get("text"))
    ]
    answer = normalize_ws(raw.get("answerKey")).upper()
    if not question or not options or not ANSWER_RE.match(answer):
        return None
    answer_text = ""
    for option in options:
        if option["label"] == answer:
            answer_text = option["text"]
            break
    if not answer_text:
        return None
    source_record_id = normalize_ws(raw.get("id"))
    provenance = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    row_id = f"exams_{split}_{language}_{stable_hash(source_record_id + provenance, 12)}"
    return McqRow(
        id=row_id,
        dataset_id=DATASET_ID,
        version=VERSION,
        split=split,
        source_id="exams_qa",
        source_url="https://github.com/mhardalov/exams-qa",
        source_license="CC-BY-SA-4.0",
        redistribution_status="redistributable_sharealike",
        source_record_id=source_record_id,
        language=language,
        language_name=language_name,
        domain="school_exam",
        subject=normalize_ws(info.get("subject")),
        grade=info.get("grade"),
        task_type="exam_mcq",
        question=question,
        options=options,
        answer=answer,
        answer_text=answer_text,
        prompt=prompt_for_mcq(question, options, language),
        reward_type="mcq_letter_exact",
        provenance_hash=f"sha256:{hashlib.sha256(provenance.encode('utf-8')).hexdigest()}",
    )


def load_exams(exams_repo: Path) -> list[McqRow]:
    files = {
        "train": exams_repo / "data/exams/multilingual/train.jsonl.tar.gz",
        "validation": exams_repo / "data/exams/multilingual/dev.jsonl.tar.gz",
        "test": exams_repo / "data/exams/multilingual/test.jsonl.tar.gz",
    }
    rows = []
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing EXAMS file: {path}")
        for raw in read_tar_jsonl(path):
            row = convert_exams_row(raw, split)
            if row:
                rows.append(row)
    return rows


def load_swedish_grpo(dataset_dir: Path) -> list[McqRow]:
    rows = []
    for split_file, split in [("train.jsonl", "train"), ("validation.jsonl", "validation"), ("test.jsonl", "test")]:
        path = dataset_dir / split_file
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                answer = normalize_ws(raw.get("answer")).upper()
                if not ANSWER_RE.match(answer):
                    continue
                options = []
                for opt in raw.get("options") or []:
                    match = re.match(r"^\s*([A-Z])\s*[\).:]\s*(.+)$", opt)
                    if match:
                        options.append({"label": match.group(1), "text": normalize_ws(match.group(2))})
                answer_text = normalize_ws(raw.get("answer_text"))
                provenance = json.dumps(raw, ensure_ascii=False, sort_keys=True)
                row_id = f"smlb_{split}_{stable_hash(raw.get('id', '') + provenance, 12)}"
                rows.append(
                    McqRow(
                        id=row_id,
                        dataset_id=DATASET_ID,
                        version=VERSION,
                        split=split,
                        source_id="swedish_medical_benchmark_local",
                        source_url="https://github.com/BirgerMoell/swedish-medical-benchmark",
                        source_license=normalize_ws(raw.get("source_license")) or "review_required",
                        redistribution_status="local_review_required",
                        source_record_id=normalize_ws(raw.get("id")),
                        language="sv",
                        language_name="Swedish",
                        domain=normalize_ws(raw.get("domain")) or "medical_exam",
                        subject=normalize_ws(raw.get("benchmark")),
                        grade=None,
                        task_type="medical_exam_mcq",
                        question=normalize_ws(raw.get("question")),
                        options=options,
                        answer=answer,
                        answer_text=answer_text,
                        prompt=normalize_ws(raw.get("prompt")),
                        reward_type="mcq_letter_exact",
                        provenance_hash=f"sha256:{hashlib.sha256(provenance.encode('utf-8')).hexdigest()}",
                    )
                )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def split_rows(rows: list[McqRow]) -> dict[str, list[McqRow]]:
    by_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        by_split.setdefault(row.split, []).append(row)
    return by_split


def build_manifest(rows: list[McqRow], dpo_pairs: list[dict], include_swedish_medical: bool) -> dict:
    by_lang = Counter(row.language for row in rows)
    by_source = Counter(row.source_id for row in rows)
    by_split = Counter(row.split for row in rows)
    by_subject = Counter(f"{row.language}:{row.subject}" for row in rows)
    return {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "created": "2026-06-22",
        "rows_grpo": len(rows),
        "rows_dpo": len(dpo_pairs),
        "include_swedish_medical": include_swedish_medical,
        "license": "CC-BY-SA-4.0 for EXAMS-derived rows; source-specific for optional local rows",
        "by_language": dict(sorted(by_lang.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "top_language_subjects": dict(by_subject.most_common(50)),
    }


def write_readme(out_dir: Path, manifest: dict) -> None:
    text = f"""---
license: cc-by-sa-4.0
task_categories:
- text-generation
- question-answering
- multiple-choice
pretty_name: OpenEuroLLM European Exam MCQ v1
tags:
- openeurollm
- grpo
- dpo
- rlvr
- multiple-choice
- exams
- european-languages
---

# {DATASET_ID}

Real-source multiple-choice exam data for European-language GRPO/RLVR and DPO
training.

This build contains:

- GRPO/RLVR rows: {manifest['rows_grpo']}
- DPO pairs: {manifest['rows_dpo']}
- Languages: {', '.join(manifest['by_language'].keys())}

The default redistributable source is EXAMS QA, licensed CC-BY-SA-4.0. Optional
local rows from Swedish Medical Benchmark are marked `local_review_required` and
should not be uploaded as a release dataset without source-term review.

## Files

- `grpo/train.jsonl`, `grpo/validation.jsonl`, `grpo/test.jsonl`
- `dpo/train.jsonl`, `dpo/validation.jsonl`, `dpo/test.jsonl`
- `manifest.json`

GRPO rows use `reward_type=mcq_letter_exact`: reward a response whose first
answer letter matches `answer`.

DPO rows are generated as correct-letter responses preferred over each incorrect
letter for the same prompt.

## Source Registry

See `source_registry.json` in the repository for official exam archives that
should be discovered and parsed locally, including Högskoleprovet, CKE, and
CERMAT. Some official archives are link-only until redistribution rights are
cleared.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--exams-repo", type=Path, default=DEFAULT_EXAMS_REPO)
    parser.add_argument("--include-swedish-medical", action="store_true")
    parser.add_argument("--swedish-grpo-dir", type=Path, default=DEFAULT_SWEDISH_GRPO)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    rows = load_exams(args.exams_repo)
    if args.include_swedish_medical:
        rows.extend(load_swedish_grpo(args.swedish_grpo_dir))

    random.Random(args.seed).shuffle(rows)
    dpo_pairs = [pair for row in rows for pair in row_to_dpo_pairs(row)]

    by_split = split_rows(rows)
    for split, split_rows_ in by_split.items():
        write_jsonl(args.out_dir / "grpo" / f"{split}.jsonl", (asdict(row) for row in split_rows_))
        split_pairs = [pair for pair in dpo_pairs if pair["split"] == split]
        write_jsonl(args.out_dir / "dpo" / f"{split}.jsonl", split_pairs)

    manifest = build_manifest(rows, dpo_pairs, args.include_swedish_medical)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.source_registry.exists():
        (args.out_dir / "source_registry.json").write_text(args.source_registry.read_text(encoding="utf-8"), encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
