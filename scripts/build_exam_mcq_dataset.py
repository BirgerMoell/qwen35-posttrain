#!/usr/bin/env python3
"""Build real-source MCQ data for GRPO/RLVR and DPO.

Default sources:
- EXAMS QA (`mhardalov/exams-qa`), CC-BY-SA-4.0.
- Global-MMLU (`CohereLabs/Global-MMLU`), Apache-2.0.
- MMMLU (`openai/MMMLU`), MIT.
- Belebele (`facebook/belebele`), CC-BY-SA-4.0.
- XCOPA (`cambridgeltl/xcopa`), CC-BY-4.0.

Optional local source:
- Swedish Medical Benchmark GRPO JSONL rows, only when explicitly enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tarfile
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DATASET_ID = "oellm-eu-exam-mcq-v1"
VERSION = "v0.3.0"
DEFAULT_OUT = Path("data/exam_mcq/oellm-eu-exam-mcq-v1")
DEFAULT_SOURCE_REGISTRY = Path("data/exam_mcq/source_registry.json")
DEFAULT_EXAMS_REPO = Path("/private/tmp/exams-qa")
DEFAULT_SWEDISH_GRPO = Path("/Users/birgermoell/AI-Sweden/swedish-medical-benchmark/data/grpo")
DEFAULT_HOGSKOLEPROVET_MANIFEST = Path("data/exam_mcq/oellm-eu-exam-mcq-v1/source_manifests/hogskoleprovet_sources.json")
DEFAULT_PDF_CACHE = Path("/private/tmp/oellm-exam-pdf-cache")
DEFAULT_SOURCES = (
    "exams_qa",
    "hogskoleprovet_ord",
    "llmzszl",
    "polish_pes_medical",
    "swedish_medical_exams_hf",
    "polish_matura_dokato",
    "slovak_mathbio_dokato",
    "slovak_financial_exam",
    "basque_public_exams",
    "catalan_public_exams",
    "global_mmlu",
    "mmmlu",
    "belebele",
    "xcopa",
)

LANGUAGE_NAMES = {
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "is": "Icelandic",
    "it": "Italian",
    "ka": "Georgian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mt": "Maltese",
    "nb": "Norwegian Bokmal",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
}

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

EUROPEAN_TARGET_LANGS = set(LANGUAGE_NAMES)
ANSWER_RE = re.compile(r"^\s*([A-Z])\s*[\).:]?\s*$")

PROMPT_TEXT = {
    "bg": ("Изберете най-добрия отговор. Отговорете само с буквата.", "Въпрос", "Варианти", "Отговор"),
    "cs": ("Vyberte nejlepší odpověď. Odpovězte pouze písmenem.", "Otázka", "Možnosti", "Odpověď"),
    "da": ("Vælg det bedste svar. Svar kun med bogstavet.", "Spørgsmål", "Muligheder", "Svar"),
    "de": ("Wählen Sie die beste Antwortoption. Antworten Sie nur mit dem Buchstaben.", "Frage", "Antwortoptionen", "Antwort"),
    "el": ("Επιλέξτε την καλύτερη απάντηση. Απαντήστε μόνο με το γράμμα.", "Ερώτηση", "Επιλογές", "Απάντηση"),
    "en": ("Choose the best answer option. Answer with the letter only.", "Question", "Options", "Answer"),
    "es": ("Elige la mejor opción de respuesta. Responde solo con la letra.", "Pregunta", "Opciones", "Respuesta"),
    "et": ("Valige parim vastus. Vastake ainult tähega.", "Küsimus", "Valikud", "Vastus"),
    "fi": ("Valitse paras vastausvaihtoehto. Vastaa vain kirjaimella.", "Kysymys", "Vaihtoehdot", "Vastaus"),
    "fr": ("Choisissez la meilleure réponse. Répondez uniquement par la lettre.", "Question", "Options", "Réponse"),
    "hr": ("Odaberite najbolji odgovor. Odgovorite samo slovom.", "Pitanje", "Mogućnosti", "Odgovor"),
    "hu": ("Válassza ki a legjobb választ. Csak a betűjellel válaszoljon.", "Kérdés", "Válaszlehetőségek", "Válasz"),
    "it": ("Scegli l'opzione di risposta migliore. Rispondi solo con la lettera.", "Domanda", "Opzioni", "Risposta"),
    "lt": ("Pasirinkite geriausią atsakymo variantą. Atsakykite tik raide.", "Klausimas", "Pasirinkimai", "Atsakymas"),
    "lv": ("Izvēlieties labāko atbildi. Atbildiet tikai ar burtu.", "Jautājums", "Varianti", "Atbilde"),
    "mk": ("Изберете го најдобриот одговор. Одговорете само со буквата.", "Прашање", "Опции", "Одговор"),
    "nl": ("Kies het beste antwoord. Antwoord alleen met de letter.", "Vraag", "Opties", "Antwoord"),
    "pl": ("Wybierz najlepszą odpowiedź. Odpowiedz tylko literą.", "Pytanie", "Opcje", "Odpowiedź"),
    "pt": ("Escolha a melhor opção de resposta. Responda apenas com a letra.", "Pergunta", "Opções", "Resposta"),
    "ro": ("Alegeți cea mai bună variantă de răspuns. Răspundeți doar cu litera.", "Întrebare", "Opțiuni", "Răspuns"),
    "ru": ("Выберите лучший вариант ответа. Ответьте только буквой.", "Вопрос", "Варианты", "Ответ"),
    "sk": ("Vyberte najlepšiu odpoveď. Odpovedzte iba písmenom.", "Otázka", "Možnosti", "Odpoveď"),
    "sl": ("Izberite najboljši odgovor. Odgovorite samo s črko.", "Vprašanje", "Možnosti", "Odgovor"),
    "sq": ("Zgjidhni përgjigjen më të mirë. Përgjigjuni vetëm me shkronjën.", "Pyetje", "Alternativa", "Përgjigje"),
    "sr": ("Изаберите најбољи одговор. Одговорите само словом.", "Питање", "Опције", "Одговор"),
    "sv": ("Välj det bästa svarsalternativet. Svara endast med bokstaven.", "Fråga", "Svarsalternativ", "Svar"),
    "tr": ("En iyi yanıt seçeneğini seçin. Yalnızca harfle cevap verin.", "Soru", "Seçenekler", "Cevap"),
    "uk": ("Виберіть найкращу відповідь. Відповідайте лише літерою.", "Питання", "Варіанти", "Відповідь"),
}

SOURCE_META = {
    "exams_qa": {
        "name": "EXAMS QA",
        "source_url": "https://github.com/mhardalov/exams-qa",
        "source_license": "CC-BY-SA-4.0",
        "license_id": "cc-by-sa-4.0",
        "license_category": "sharealike",
        "license_filter_tags": ["attribution_required", "sharealike"],
        "redistribution_status": "redistributable_sharealike",
    },
    "hogskoleprovet_ord": {
        "name": "Högskoleprovet ORD",
        "source_url": "https://www.studera.nu/hogskoleprov/om/forbereda/tidigare/",
        "source_license": "unknown/missing; official PDFs state publication permission was obtained for protected material",
        "license_id": "unknown",
        "license_category": "unknown_or_missing",
        "license_filter_tags": ["unknown_or_missing", "official_public_exam", "publication_permission_note"],
        "redistribution_status": "official_public_unknown_redistribution",
    },
    "llmzszl": {
        "name": "LLMzSzL Polish national exams",
        "source_url": "https://huggingface.co/datasets/amu-cai/llmzszl-dataset",
        "source_license": "unknown/missing on Hugging Face dataset card",
        "license_id": "unknown",
        "license_category": "unknown_or_missing",
        "license_filter_tags": ["unknown_or_missing", "national_exam"],
        "redistribution_status": "unknown_missing_license",
    },
    "polish_pes_medical": {
        "name": "Polish PES specialist medical exams",
        "source_url": "https://huggingface.co/datasets/amu-cai/medical-exams-PES-PL-2007-2024",
        "source_license": "unknown/missing on Hugging Face dataset card",
        "license_id": "unknown",
        "license_category": "unknown_or_missing",
        "license_filter_tags": ["unknown_or_missing", "official_public_exam", "medical_exam", "specialist_exam"],
        "redistribution_status": "unknown_missing_license",
    },
    "swedish_medical_exams_hf": {
        "name": "Swedish medical exam MCQs",
        "source_url": "https://huggingface.co/datasets/sarafuyu/swedish-medical-exams-mcq-1006-json",
        "source_license": "unknown",
        "license_id": "unknown",
        "license_category": "unknown_or_missing",
        "license_filter_tags": ["unknown_or_missing", "official_public_exam", "medical_exam"],
        "redistribution_status": "official_public_unknown_redistribution",
    },
    "polish_matura_dokato": {
        "name": "Polish matura MCQs",
        "source_url": "https://huggingface.co/datasets/dokato/exam-polish-matura",
        "source_license": "CC-BY-NC-SA-2.0",
        "license_id": "cc-by-nc-sa-2.0",
        "license_category": "noncommercial_sharealike",
        "license_filter_tags": ["attribution_required", "noncommercial", "sharealike"],
        "redistribution_status": "redistributable_noncommercial_sharealike",
    },
    "slovak_mathbio_dokato": {
        "name": "Slovak math/biology university-entry exams",
        "source_url": "https://huggingface.co/datasets/dokato/exam-slovak-mathbio",
        "source_license": "CC-BY-NC-SA-2.0",
        "license_id": "cc-by-nc-sa-2.0",
        "license_category": "noncommercial_sharealike",
        "license_filter_tags": ["attribution_required", "noncommercial", "sharealike"],
        "redistribution_status": "redistributable_noncommercial_sharealike",
    },
    "slovak_financial_exam": {
        "name": "Slovak financial certification exam",
        "source_url": "https://huggingface.co/datasets/TUKE-KEMT/slovak-financial-exam",
        "source_license": "CC-BY-SA-4.0",
        "license_id": "cc-by-sa-4.0",
        "license_category": "sharealike",
        "license_filter_tags": ["attribution_required", "sharealike", "financial_exam"],
        "redistribution_status": "redistributable_sharealike",
    },
    "basque_public_exams": {
        "name": "Basque public-service legal exams",
        "source_url": "https://huggingface.co/datasets/amayuelas/aya-global-exams-basque",
        "source_license": "Open License (dataset row metadata; exact terms not normalized)",
        "license_id": "open-license",
        "license_category": "custom_open_needs_review",
        "license_filter_tags": ["custom_open_license", "needs_license_review", "public_service_exam"],
        "redistribution_status": "declared_open_license_needs_review",
    },
    "catalan_public_exams": {
        "name": "Catalan public-service legal exams",
        "source_url": "https://huggingface.co/datasets/amayuelas/aya-global-exams-catalan",
        "source_license": "Open Information Use License - Catalonia",
        "license_id": "open-information-use-license-catalonia",
        "license_category": "custom_open_needs_review",
        "license_filter_tags": ["custom_open_license", "needs_license_review", "public_service_exam"],
        "redistribution_status": "declared_open_license_needs_review",
    },
    "global_mmlu": {
        "name": "Global-MMLU",
        "source_url": "https://huggingface.co/datasets/CohereLabs/Global-MMLU",
        "source_license": "Apache-2.0",
        "license_id": "apache-2.0",
        "license_category": "permissive",
        "license_filter_tags": ["permissive"],
        "redistribution_status": "redistributable_declared_license",
    },
    "mmmlu": {
        "name": "MMMLU",
        "source_url": "https://huggingface.co/datasets/openai/MMMLU",
        "source_license": "MIT",
        "license_id": "mit",
        "license_category": "permissive",
        "license_filter_tags": ["permissive"],
        "redistribution_status": "redistributable_declared_license",
    },
    "belebele": {
        "name": "Belebele",
        "source_url": "https://huggingface.co/datasets/facebook/belebele",
        "source_license": "CC-BY-SA-4.0",
        "license_id": "cc-by-sa-4.0",
        "license_category": "sharealike",
        "license_filter_tags": ["attribution_required", "sharealike"],
        "redistribution_status": "redistributable_sharealike",
    },
    "xcopa": {
        "name": "XCOPA",
        "source_url": "https://huggingface.co/datasets/cambridgeltl/xcopa",
        "source_license": "CC-BY-4.0",
        "license_id": "cc-by-4.0",
        "license_category": "attribution",
        "license_filter_tags": ["attribution_required"],
        "redistribution_status": "redistributable_attribution",
    },
    "swedish_medical_benchmark_local": {
        "name": "Swedish Medical Benchmark local rows",
        "source_url": "https://github.com/BirgerMoell/swedish-medical-benchmark",
        "source_license": "unknown",
        "license_id": "unknown",
        "license_category": "unknown_or_missing",
        "license_filter_tags": ["unknown_or_missing"],
        "redistribution_status": "local_review_required",
    },
}

SOURCE_ALIASES = {
    "exams": "exams_qa",
    "exams_qa": "exams_qa",
    "hogskoleprovet": "hogskoleprovet_ord",
    "hogskoleprovet_ord": "hogskoleprovet_ord",
    "llmzszl": "llmzszl",
    "polish-national-exams": "llmzszl",
    "pes": "polish_pes_medical",
    "polish-pes": "polish_pes_medical",
    "polish_pes_medical": "polish_pes_medical",
    "polish_matura": "polish_matura_dokato",
    "polish_matura_dokato": "polish_matura_dokato",
    "slovak_mathbio": "slovak_mathbio_dokato",
    "slovak_mathbio_dokato": "slovak_mathbio_dokato",
    "slovak_financial": "slovak_financial_exam",
    "slovak_financial_exam": "slovak_financial_exam",
    "basque-public": "basque_public_exams",
    "basque_public_exams": "basque_public_exams",
    "catalan-public": "catalan_public_exams",
    "catalan_public_exams": "catalan_public_exams",
    "swedish-medical": "swedish_medical_exams_hf",
    "swedish_medical_exams_hf": "swedish_medical_exams_hf",
    "global-mmlu": "global_mmlu",
    "global_mmlu": "global_mmlu",
    "mmmlu": "mmmlu",
    "belebele": "belebele",
    "xcopa": "xcopa",
}

GLOBAL_MMLU_CONFIGS = {
    "cs": ("cs", "Czech"),
    "de": ("de", "German"),
    "el": ("el", "Greek"),
    "en": ("en", "English"),
    "es": ("es", "Spanish"),
    "fr": ("fr", "French"),
    "it": ("it", "Italian"),
    "lt": ("lt", "Lithuanian"),
    "nl": ("nl", "Dutch"),
    "pl": ("pl", "Polish"),
    "pt": ("pt", "Portuguese"),
    "ro": ("ro", "Romanian"),
    "ru": ("ru", "Russian"),
    "sr": ("sr", "Serbian"),
    "sv": ("sv", "Swedish"),
    "tr": ("tr", "Turkish"),
    "uk": ("uk", "Ukrainian"),
}

MMMLU_CONFIGS = {
    "DE_DE": ("de", "German"),
    "ES_LA": ("es", "Spanish"),
    "FR_FR": ("fr", "French"),
    "IT_IT": ("it", "Italian"),
    "PT_BR": ("pt", "Portuguese"),
}

BELEBELE_CONFIGS = {
    "als_Latn": ("sq", "Albanian"),
    "bul_Cyrl": ("bg", "Bulgarian"),
    "cat_Latn": ("ca", "Catalan"),
    "ces_Latn": ("cs", "Czech"),
    "dan_Latn": ("da", "Danish"),
    "deu_Latn": ("de", "German"),
    "ell_Grek": ("el", "Greek"),
    "eng_Latn": ("en", "English"),
    "est_Latn": ("et", "Estonian"),
    "eus_Latn": ("eu", "Basque"),
    "fin_Latn": ("fi", "Finnish"),
    "fra_Latn": ("fr", "French"),
    "hrv_Latn": ("hr", "Croatian"),
    "hun_Latn": ("hu", "Hungarian"),
    "hye_Armn": ("hy", "Armenian"),
    "isl_Latn": ("is", "Icelandic"),
    "ita_Latn": ("it", "Italian"),
    "kat_Geor": ("ka", "Georgian"),
    "lit_Latn": ("lt", "Lithuanian"),
    "lvs_Latn": ("lv", "Latvian"),
    "mkd_Cyrl": ("mk", "Macedonian"),
    "mlt_Latn": ("mt", "Maltese"),
    "nld_Latn": ("nl", "Dutch"),
    "nob_Latn": ("nb", "Norwegian Bokmal"),
    "pol_Latn": ("pl", "Polish"),
    "por_Latn": ("pt", "Portuguese"),
    "ron_Latn": ("ro", "Romanian"),
    "rus_Cyrl": ("ru", "Russian"),
    "slk_Latn": ("sk", "Slovak"),
    "slv_Latn": ("sl", "Slovenian"),
    "spa_Latn": ("es", "Spanish"),
    "srp_Cyrl": ("sr", "Serbian"),
    "swe_Latn": ("sv", "Swedish"),
    "tur_Latn": ("tr", "Turkish"),
    "ukr_Cyrl": ("uk", "Ukrainian"),
}

XCOPA_CONFIGS = {
    "et": ("et", "Estonian"),
    "it": ("it", "Italian"),
    "tr": ("tr", "Turkish"),
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
    license_id: str
    license_category: str
    license_filter_tags: list[str]
    redistribution_status: str
    source_split: str
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


def provenance_json(raw: object) -> str:
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)


def normalize_ws(value: object | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_license_id(value: object | None) -> str:
    normalized = normalize_ws(value).lower()
    if not normalized or normalized in {"review_required", "unknown", "missing", "none"}:
        return "unknown"
    for license_id in ["apache-2.0", "mit", "cc-by-nc-sa-2.0", "cc-by-sa-4.0", "cc-by-4.0"]:
        if license_id in normalized:
            return license_id
    return re.sub(r"[^a-z0-9.+-]+", "-", normalized).strip("-") or "unknown"


def license_category_for(license_id: str) -> str:
    if license_id in {"apache-2.0", "mit"}:
        return "permissive"
    if license_id == "cc-by-4.0":
        return "attribution"
    if license_id == "cc-by-sa-4.0":
        return "sharealike"
    if license_id == "cc-by-nc-sa-2.0":
        return "noncommercial_sharealike"
    if license_id in {"open-license", "open-information-use-license-catalonia"}:
        return "custom_open_needs_review"
    return "unknown_or_missing"


def license_tags_for(license_id: str) -> list[str]:
    if license_id in {"apache-2.0", "mit"}:
        return ["permissive"]
    if license_id == "cc-by-4.0":
        return ["attribution_required"]
    if license_id == "cc-by-sa-4.0":
        return ["attribution_required", "sharealike"]
    if license_id == "cc-by-nc-sa-2.0":
        return ["attribution_required", "noncommercial", "sharealike"]
    if license_id in {"open-license", "open-information-use-license-catalonia"}:
        return ["custom_open_license", "needs_license_review"]
    return ["unknown_or_missing"]


def prompt_for_mcq(question: str, options: list[dict], language: str) -> str:
    option_lines = "\n".join(f"{opt['label']}) {opt['text']}" for opt in options)
    instruction, question_label, options_label, answer_label = PROMPT_TEXT.get(
        language, ("Choose the best answer option. Answer with the letter only.", "Question", "Options", "Answer")
    )
    return f"{instruction}\n\n{question_label}:\n{question}\n\n{options_label}:\n{option_lines}\n\n{answer_label}:"


def answer_text(options: list[dict], answer: str) -> str:
    for option in options:
        if option["label"] == answer:
            return option["text"]
    return ""


def make_options(values: Iterable[object]) -> list[dict]:
    options = []
    for label, value in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", values):
        text = normalize_ws(value)
        if text:
            options.append({"label": label, "text": text})
    return options


def parse_labeled_question_options(text: object) -> tuple[str, list[dict]]:
    stem_lines: list[str] = []
    options: list[dict] = []
    current_label = ""
    current_text = ""
    for raw_line in str(text or "").splitlines():
        line = normalize_ws(raw_line)
        if not line:
            continue
        option_match = re.match(r"^([A-Z])[\).]\s*(.+)$", line)
        if option_match:
            if current_label and current_text:
                options.append({"label": current_label, "text": normalize_ws(current_text)})
            current_label = option_match.group(1)
            current_text = option_match.group(2)
        elif current_label:
            current_text = normalize_ws(f"{current_text} {line}")
        else:
            stem_lines.append(line)
    if current_label and current_text:
        options.append({"label": current_label, "text": normalize_ws(current_text)})
    return normalize_ws(" ".join(stem_lines)), options


def numeric_answer_to_letter(value: object, *, one_based: bool) -> str:
    try:
        index = int(normalize_ws(value))
    except ValueError:
        return ""
    if one_based:
        index -= 1
    if 0 <= index < 26:
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]
    return ""


def output_split(source_id: str, source_split: str) -> str:
    if source_id == "exams_qa":
        return {"dev": "validation"}.get(source_split, source_split)
    if source_split in {"dev", "validation"}:
        return "validation"
    return "train"


def make_mcq_row(
    *,
    source_id: str,
    source_split: str,
    source_record_id: str,
    language: str,
    language_name: str,
    domain: str,
    subject: str,
    grade: str | int | None,
    task_type: str,
    question: str,
    options: list[dict],
    answer: str,
    raw: object,
) -> McqRow | None:
    if language not in EUROPEAN_TARGET_LANGS:
        return None
    question = normalize_ws(question)
    answer = normalize_ws(answer).upper()
    if not question or not options or not ANSWER_RE.match(answer):
        return None
    labels = [normalize_ws(option.get("label")).upper() for option in options]
    if len(labels) != len(set(labels)):
        return None
    correct_text = answer_text(options, answer)
    if not correct_text:
        return None
    source_meta = SOURCE_META[source_id]
    provenance = provenance_json(raw)
    split = output_split(source_id, source_split)
    row_id = f"{source_id}_{split}_{language}_{stable_hash(source_record_id + provenance, 12)}"
    return McqRow(
        id=row_id,
        dataset_id=DATASET_ID,
        version=VERSION,
        split=split,
        source_id=source_id,
        source_url=source_meta["source_url"],
        source_license=source_meta["source_license"],
        license_id=source_meta["license_id"],
        license_category=source_meta["license_category"],
        license_filter_tags=list(source_meta["license_filter_tags"]),
        redistribution_status=source_meta["redistribution_status"],
        source_split=source_split,
        source_record_id=source_record_id,
        language=language,
        language_name=language_name,
        domain=domain,
        subject=normalize_ws(subject),
        grade=grade,
        task_type=task_type,
        question=question,
        options=options,
        answer=answer,
        answer_text=correct_text,
        prompt=prompt_for_mcq(question, options, language),
        reward_type="mcq_letter_exact",
        provenance_hash=f"sha256:{hashlib.sha256(provenance.encode('utf-8')).hexdigest()}",
    )


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
            "source_url": row.source_url,
            "source_split": row.source_split,
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
            "license_id": row.license_id,
            "license_category": row.license_category,
            "license_filter_tags": row.license_filter_tags,
            "redistribution_status": row.redistribution_status,
            "provenance_hash": row.provenance_hash,
        }


def require_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install datasets") from exc
    return load_dataset


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


def download_to_cache(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    basename = Path(parsed.path).name or "document.pdf"
    path = cache_dir / f"{stable_hash(url, 12)}_{basename}"
    if path.exists() and path.stat().st_size > 0:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": "OpenEuroLLM exam MCQ builder"})
    with urllib.request.urlopen(req, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_exam_date(*values: str) -> str:
    text = " ".join(values)
    match = re.search(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})", text)
    if match:
        return "-".join(match.groups())
    compact = re.search(r"\b(\d{2})(\d{2})(\d{2})\b", text)
    if compact:
        year, month, day = compact.groups()
        return f"20{year}-{month}-{day}"
    return "unknown-date"


def extract_provpass(*values: str) -> str:
    text = " ".join(values).lower()
    for pattern in [r"provpass[-\s_]*(\d)", r"del[-\s_]*(\d)", r"pass[-\s_]*(\d)"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def parse_hogskoleprovet_answer_key(text: str) -> dict[str, dict[int, str]]:
    header_match = re.search(r"Provpass\s+\d.*?(?=\n\s*\d+\s+[A-E])", text, flags=re.S)
    header_text = header_match.group(0) if header_match else text[:1000]
    provpasses = []
    for provpass in re.findall(r"Provpass\s+(\d)", header_text):
        if provpass not in provpasses:
            provpasses.append(provpass)
    provpasses = provpasses[:4]
    if not provpasses:
        provpasses = ["1", "2", "3", "4"]

    answer_key: dict[str, dict[int, str]] = {provpass: {} for provpass in provpasses}
    for line in text.splitlines():
        tokens = re.findall(r"\b(?:\d{1,2}|[A-E])\b", line)
        if len(tokens) < 2 * len(provpasses):
            continue
        if not tokens[0].isdigit():
            continue
        for column, provpass in enumerate(provpasses):
            q_token = tokens[column * 2]
            a_token = tokens[column * 2 + 1]
            if q_token.isdigit() and a_token in {"A", "B", "C", "D", "E"}:
                answer_key[provpass][int(q_token)] = a_token
    return answer_key


def parse_hogskoleprovet_ord_questions(text: str) -> list[dict]:
    questions = []
    current: dict | None = None
    in_ord = False
    for raw_line in text.splitlines():
        line = normalize_ws(raw_line)
        if not line:
            continue
        if "ORD" in line and "Ordförståelse" in line:
            in_ord = True
            continue
        if not in_ord:
            continue
        if re.match(r"^(LÄS|MEK|ELF|XYZ|KVA|NOG|DTK)\b", line):
            break
        question_match = re.match(r"^([1-9]|10)\.\s+(.+)$", line)
        if question_match:
            if current and len(current["options"]) == 5:
                questions.append(current)
            current = {
                "number": int(question_match.group(1)),
                "stem": normalize_ws(question_match.group(2)),
                "options": [],
            }
            continue
        option_match = re.match(r"^([A-E])\s+(.+)$", line)
        if current and option_match:
            current["options"].append({"label": option_match.group(1), "text": normalize_ws(option_match.group(2))})
        elif current and current["options"]:
            current["options"][-1]["text"] = normalize_ws(current["options"][-1]["text"] + " " + line)
    if current and len(current["options"]) == 5:
        questions.append(current)
    by_number = {question["number"]: question for question in questions if 1 <= question["number"] <= 10}
    return [by_number[number] for number in sorted(by_number)]


def load_hogskoleprovet_ord(manifest_path: Path, cache_dir: Path) -> list[McqRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Högskoleprovet manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for page in manifest.get("pages", []):
        pdf_links = page.get("pdf_links") or []
        facit_links = [
            link for link in pdf_links if "facit" in normalize_ws(link.get("text")).lower() or "facit" in link.get("url", "").lower()
        ]
        if not facit_links:
            continue
        facit_url = facit_links[0]["url"]
        try:
            answer_key = parse_hogskoleprovet_answer_key(pdf_text(download_to_cache(facit_url, cache_dir)))
        except Exception as exc:  # noqa: BLE001 - keep building other official PDFs.
            print(f"Skipping Högskoleprovet answer key {facit_url}: {exc}", flush=True)
            continue

        verbal_links = [
            link
            for link in pdf_links
            if "verbal" in normalize_ws(link.get("text")).lower()
            and "elf" in normalize_ws(link.get("text")).lower()
            and "pdf" in link.get("url", "").lower()
        ]
        for link in verbal_links:
            url = link["url"]
            provpass = extract_provpass(link.get("text", ""), url)
            if not provpass or provpass not in answer_key:
                continue
            exam_date = extract_exam_date(page.get("title", ""), page.get("url", ""), url)
            try:
                questions = parse_hogskoleprovet_ord_questions(pdf_text(download_to_cache(url, cache_dir)))
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping Högskoleprovet PDF {url}: {exc}", flush=True)
                continue
            for question in questions:
                answer = answer_key[provpass].get(question["number"])
                if not answer:
                    continue
                raw = {
                    "exam_date": exam_date,
                    "provpass": provpass,
                    "question_number": question["number"],
                    "question": question["stem"],
                    "options": question["options"],
                    "answer": answer,
                    "source_pdf_url": url,
                    "answer_key_url": facit_url,
                    "source_page_url": page.get("url", ""),
                    "source_page_title": page.get("title", ""),
                    "license_note": SOURCE_META["hogskoleprovet_ord"]["source_license"],
                }
                row = make_mcq_row(
                    source_id="hogskoleprovet_ord",
                    source_split="official_archive",
                    source_record_id=f"{exam_date}/provpass-{provpass}/ord/{question['number']}",
                    language="sv",
                    language_name="Swedish",
                    domain="university_admission_exam",
                    subject="Högskoleprovet ORD",
                    grade=None,
                    task_type="vocabulary_exam_mcq",
                    question=question["stem"],
                    options=question["options"],
                    answer=answer,
                    raw=raw,
                )
                if row:
                    rows.append(row)
    return rows


def convert_exams_row(raw: dict, source_split: str) -> McqRow | None:
    info = raw.get("info") or {}
    language_name = normalize_ws(info.get("language"))
    language = LANGUAGE_NAME_TO_ISO.get(language_name)
    question_obj = raw.get("question") or {}
    options = [
        {"label": normalize_ws(choice.get("label")).upper(), "text": normalize_ws(choice.get("text"))}
        for choice in question_obj.get("choices") or []
        if normalize_ws(choice.get("label")) and normalize_ws(choice.get("text"))
    ]
    return make_mcq_row(
        source_id="exams_qa",
        source_split=source_split,
        source_record_id=normalize_ws(raw.get("id")),
        language=language or "",
        language_name=language_name,
        domain="school_exam",
        subject=normalize_ws(info.get("subject")),
        grade=info.get("grade"),
        task_type="exam_mcq",
        question=normalize_ws(question_obj.get("stem")),
        options=options,
        answer=normalize_ws(raw.get("answerKey")),
        raw=raw,
    )


def load_exams(exams_repo: Path) -> list[McqRow]:
    files = {
        "train": exams_repo / "data/exams/multilingual/train.jsonl.tar.gz",
        "dev": exams_repo / "data/exams/multilingual/dev.jsonl.tar.gz",
        "test": exams_repo / "data/exams/multilingual/test.jsonl.tar.gz",
    }
    rows = []
    for source_split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing EXAMS file: {path}")
        for raw in read_tar_jsonl(path):
            row = convert_exams_row(raw, source_split)
            if row:
                rows.append(row)
    return rows


def load_llmzszl() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "test"
    print(f"Loading LLMzSzL {source_split}", flush=True)
    dataset = load_dataset("amu-cai/llmzszl-dataset", split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        options = make_options(raw.get("answers") or [])
        row = make_mcq_row(
            source_id="llmzszl",
            source_split=source_split,
            source_record_id=f"{normalize_ws(raw.get('year'))}/{normalize_ws(raw.get('type'))}/{normalize_ws(raw.get('name'))}/{index}",
            language="pl",
            language_name="Polish",
            domain="national_exam",
            subject=normalize_ws(raw.get("name")),
            grade=normalize_ws(raw.get("type")),
            task_type="national_exam_mcq",
            question=normalize_ws(raw.get("question")),
            options=options,
            answer=numeric_answer_to_letter(raw.get("correct_answer_index"), one_based=False),
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_polish_pes_medical() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "train"
    print(f"Loading Polish PES medical exams/{source_split}", flush=True)
    dataset = load_dataset("amu-cai/medical-exams-PES-PL-2007-2024", split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        question, options = parse_labeled_question_options(raw.get("question_w_options"))
        row = make_mcq_row(
            source_id="polish_pes_medical",
            source_split=source_split,
            source_record_id=(
                f"{normalize_ws(raw.get('edition'))}/{normalize_ws(raw.get('year'))}/"
                f"{normalize_ws(raw.get('season'))}/{normalize_ws(raw.get('specialty'))}/"
                f"{normalize_ws(raw.get('question_id')) or index}"
            ),
            language="pl",
            language_name="Polish",
            domain="medical_specialist_exam",
            subject=normalize_ws(raw.get("specialty")),
            grade=normalize_ws(f"{normalize_ws(raw.get('year'))} {normalize_ws(raw.get('season'))}"),
            task_type="medical_specialist_exam_mcq",
            question=question,
            options=options,
            answer=normalize_ws(raw.get("answer")),
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_polish_matura_dokato() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "train"
    print(f"Loading Polish matura dokato/{source_split}", flush=True)
    dataset = load_dataset("dokato/exam-polish-matura", split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        row = make_mcq_row(
            source_id="polish_matura_dokato",
            source_split=source_split,
            source_record_id=f"{normalize_ws(raw.get('file_name'))}/{normalize_ws(raw.get('original_question_num')) or index}",
            language="pl",
            language_name="Polish",
            domain="school_leaving_exam",
            subject=normalize_ws(raw.get("category_original_lang")) or normalize_ws(raw.get("category_en")),
            grade=normalize_ws(raw.get("level")),
            task_type="matura_exam_mcq",
            question=normalize_ws(raw.get("question")),
            options=make_options(raw.get("options") or []),
            answer=numeric_answer_to_letter(raw.get("answer"), one_based=True),
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_slovak_mathbio_dokato() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "train"
    print(f"Loading Slovak math/bio dokato/{source_split}", flush=True)
    dataset = load_dataset("dokato/exam-slovak-mathbio", split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        row = make_mcq_row(
            source_id="slovak_mathbio_dokato",
            source_split=source_split,
            source_record_id=f"{normalize_ws(raw.get('file_name'))}/{normalize_ws(raw.get('original_question_num')) or index}",
            language="sk",
            language_name="Slovak",
            domain="university_entry_exam",
            subject=normalize_ws(raw.get("category_original_lang")) or normalize_ws(raw.get("category_en")),
            grade=normalize_ws(raw.get("level")),
            task_type="university_entry_exam_mcq",
            question=normalize_ws(raw.get("question")),
            options=make_options(raw.get("options") or []),
            answer=numeric_answer_to_letter(raw.get("answer"), one_based=True),
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_slovak_financial_exam() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "test"
    print(f"Loading Slovak financial exam/{source_split}", flush=True)
    dataset = load_dataset("TUKE-KEMT/slovak-financial-exam", split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        subject = " / ".join(
            item for item in [normalize_ws(raw.get("sector")), normalize_ws(raw.get("area"))] if item
        )
        row = make_mcq_row(
            source_id="slovak_financial_exam",
            source_split=source_split,
            source_record_id=normalize_ws(raw.get("id")) or str(index),
            language="sk",
            language_name="Slovak",
            domain="financial_certification_exam",
            subject=subject,
            grade=normalize_ws(raw.get("level")),
            task_type="financial_certification_exam_mcq",
            question=normalize_ws(raw.get("prompt")),
            options=make_options(raw.get("answers") or []),
            answer=numeric_answer_to_letter(raw.get("label"), one_based=False),
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_aya_public_exam(
    dataset_name: str,
    *,
    source_id: str,
    language: str,
    language_name: str,
    require_question_terminal: bool,
) -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "train"
    print(f"Loading {dataset_name}/{source_split}", flush=True)
    dataset = load_dataset(dataset_name, split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        question = normalize_ws(raw.get("question"))
        options = make_options(raw.get("options") or [])
        answer = numeric_answer_to_letter(raw.get("answer"), one_based=True)
        if len(options) != 4 or not answer:
            continue
        if require_question_terminal and not question.endswith(("?", ":")):
            continue
        row = make_mcq_row(
            source_id=source_id,
            source_split=source_split,
            source_record_id=f"{normalize_ws(raw.get('file_name'))}/{normalize_ws(raw.get('original_question_num')) or index}",
            language=language,
            language_name=language_name,
            domain="public_service_exam",
            subject=normalize_ws(raw.get("category_original_lang")) or normalize_ws(raw.get("category_en")),
            grade=normalize_ws(raw.get("level")),
            task_type="public_service_exam_mcq",
            question=question,
            options=options,
            answer=answer,
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_basque_public_exams() -> list[McqRow]:
    return load_aya_public_exam(
        "amayuelas/aya-global-exams-basque",
        source_id="basque_public_exams",
        language="eu",
        language_name="Basque",
        require_question_terminal=False,
    )


def load_catalan_public_exams() -> list[McqRow]:
    return load_aya_public_exam(
        "amayuelas/aya-global-exams-catalan",
        source_id="catalan_public_exams",
        language="ca",
        language_name="Catalan",
        require_question_terminal=True,
    )


def load_swedish_medical_exams_hf() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    source_split = "train"
    print(f"Loading Swedish medical exams HF/{source_split}", flush=True)
    dataset = load_dataset("sarafuyu/swedish-medical-exams-mcq-1006-json", split=source_split)
    for index, raw_row in enumerate(dataset):
        raw = dict(raw_row)
        row = make_mcq_row(
            source_id="swedish_medical_exams_hf",
            source_split=source_split,
            source_record_id=f"{normalize_ws(raw.get('file_name'))}/{normalize_ws(raw.get('original_question_num')) or index}",
            language="sv",
            language_name="Swedish",
            domain="medical_licensing_exam",
            subject=normalize_ws(raw.get("category_original_lang")) or normalize_ws(raw.get("category_en")),
            grade=normalize_ws(raw.get("level")),
            task_type="medical_exam_mcq",
            question=normalize_ws(raw.get("question")),
            options=make_options(raw.get("options") or []),
            answer=numeric_answer_to_letter(raw.get("answer"), one_based=True),
            raw=raw,
        )
        if row:
            rows.append(row)
    return rows


def load_global_mmlu() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    for config, (language, language_name) in GLOBAL_MMLU_CONFIGS.items():
        for source_split in ["dev", "test"]:
            print(f"Loading Global-MMLU {config}/{source_split}", flush=True)
            dataset = load_dataset("CohereLabs/Global-MMLU", config, split=source_split)
            for index, raw_row in enumerate(dataset):
                raw = dict(raw_row)
                options = make_options([raw.get("option_a"), raw.get("option_b"), raw.get("option_c"), raw.get("option_d")])
                row = make_mcq_row(
                    source_id="global_mmlu",
                    source_split=source_split,
                    source_record_id=f"{config}/{source_split}/{normalize_ws(raw.get('sample_id')) or index}",
                    language=language,
                    language_name=language_name,
                    domain="academic_exam",
                    subject=normalize_ws(raw.get("subject")),
                    grade=None,
                    task_type="academic_exam_mcq",
                    question=normalize_ws(raw.get("question")),
                    options=options,
                    answer=normalize_ws(raw.get("answer")),
                    raw=raw,
                )
                if row:
                    rows.append(row)
    return rows


def load_mmmlu() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    for config, (language, language_name) in MMMLU_CONFIGS.items():
        source_split = "test"
        print(f"Loading MMMLU {config}/{source_split}", flush=True)
        dataset = load_dataset("openai/MMMLU", config, split=source_split)
        for index, raw_row in enumerate(dataset):
            raw = dict(raw_row)
            options = make_options([raw.get("A"), raw.get("B"), raw.get("C"), raw.get("D")])
            row = make_mcq_row(
                source_id="mmmlu",
                source_split=source_split,
                source_record_id=f"{config}/{source_split}/{normalize_ws(raw.get('Unnamed: 0')) or index}",
                language=language,
                language_name=language_name,
                domain="academic_exam_translation",
                subject=normalize_ws(raw.get("Subject")),
                grade=None,
                task_type="academic_exam_mcq",
                question=normalize_ws(raw.get("Question")),
                options=options,
                answer=normalize_ws(raw.get("Answer")),
                raw=raw,
            )
            if row:
                rows.append(row)
    return rows


def load_belebele() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    for config, (language, language_name) in BELEBELE_CONFIGS.items():
        source_split = "test"
        print(f"Loading Belebele {config}/{source_split}", flush=True)
        dataset = load_dataset("facebook/belebele", config, split=source_split)
        for index, raw_row in enumerate(dataset):
            raw = dict(raw_row)
            options = make_options([raw.get("mc_answer1"), raw.get("mc_answer2"), raw.get("mc_answer3"), raw.get("mc_answer4")])
            answer_number = normalize_ws(raw.get("correct_answer_num"))
            if answer_number not in {"1", "2", "3", "4"}:
                continue
            answer = "ABCD"[int(answer_number) - 1]
            question = f"Passage:\n{normalize_ws(raw.get('flores_passage'))}\n\nQuestion:\n{normalize_ws(raw.get('question'))}"
            row = make_mcq_row(
                source_id="belebele",
                source_split=source_split,
                source_record_id=f"{config}/{source_split}/{normalize_ws(raw.get('link'))}/{normalize_ws(raw.get('question_number')) or index}",
                language=language,
                language_name=language_name,
                domain="reading_comprehension",
                subject="general_reading_comprehension",
                grade=None,
                task_type="reading_comprehension_mcq",
                question=question,
                options=options,
                answer=answer,
                raw=raw,
            )
            if row:
                rows.append(row)
    return rows


def load_xcopa() -> list[McqRow]:
    load_dataset = require_load_dataset()
    rows = []
    for config, (language, language_name) in XCOPA_CONFIGS.items():
        for source_split in ["validation", "test"]:
            print(f"Loading XCOPA {config}/{source_split}", flush=True)
            dataset = load_dataset("cambridgeltl/xcopa", config, split=source_split)
            for index, raw_row in enumerate(dataset):
                raw = dict(raw_row)
                label = raw.get("label")
                if label not in {0, 1}:
                    continue
                relation = normalize_ws(raw.get("question"))
                if relation == "cause":
                    question = f"Premise:\n{normalize_ws(raw.get('premise'))}\n\nWhich option is the more plausible cause?"
                else:
                    question = f"Premise:\n{normalize_ws(raw.get('premise'))}\n\nWhich option is the more plausible effect?"
                row = make_mcq_row(
                    source_id="xcopa",
                    source_split=source_split,
                    source_record_id=f"{config}/{source_split}/{normalize_ws(raw.get('idx')) or index}",
                    language=language,
                    language_name=language_name,
                    domain="commonsense_reasoning",
                    subject="causal_reasoning",
                    grade=None,
                    task_type="causal_reasoning_mcq",
                    question=question,
                    options=make_options([raw.get("choice1"), raw.get("choice2")]),
                    answer="AB"[int(label)],
                    raw=raw,
                )
                if row:
                    rows.append(row)
    return rows


def load_swedish_grpo(dataset_dir: Path) -> list[McqRow]:
    rows = []
    source_meta = SOURCE_META["swedish_medical_benchmark_local"]
    for split_file, source_split in [("train.jsonl", "train"), ("validation.jsonl", "validation"), ("test.jsonl", "test")]:
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
                correct_text = normalize_ws(raw.get("answer_text")) or answer_text(options, answer)
                if not correct_text:
                    continue
                provenance = provenance_json(raw)
                source_license = normalize_ws(raw.get("source_license")) or source_meta["source_license"]
                license_id = canonical_license_id(source_license)
                source_record_id = normalize_ws(raw.get("id"))
                row_id = f"smlb_{source_split}_{stable_hash(source_record_id + provenance, 12)}"
                rows.append(
                    McqRow(
                        id=row_id,
                        dataset_id=DATASET_ID,
                        version=VERSION,
                        split=output_split("swedish_medical_benchmark_local", source_split),
                        source_id="swedish_medical_benchmark_local",
                        source_url=source_meta["source_url"],
                        source_license=source_license,
                        license_id=license_id,
                        license_category=license_category_for(license_id),
                        license_filter_tags=license_tags_for(license_id),
                        redistribution_status=source_meta["redistribution_status"],
                        source_split=source_split,
                        source_record_id=source_record_id,
                        language="sv",
                        language_name="Swedish",
                        domain=normalize_ws(raw.get("domain")) or "medical_exam",
                        subject=normalize_ws(raw.get("benchmark")),
                        grade=None,
                        task_type="medical_exam_mcq",
                        question=normalize_ws(raw.get("question")),
                        options=options,
                        answer=answer,
                        answer_text=correct_text,
                        prompt=normalize_ws(raw.get("prompt")) or prompt_for_mcq(normalize_ws(raw.get("question")), options, "sv"),
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


def dpo_pair_count(rows: Iterable[McqRow]) -> int:
    return sum(max(0, len(row.options) - 1) for row in rows)


def build_manifest(rows: list[McqRow], selected_sources: list[str], include_swedish_medical: bool) -> dict:
    by_lang = Counter(row.language for row in rows)
    by_source = Counter(row.source_id for row in rows)
    by_split = Counter(row.split for row in rows)
    by_license_id = Counter(row.license_id for row in rows)
    by_license_category = Counter(row.license_category for row in rows)
    by_redistribution_status = Counter(row.redistribution_status for row in rows)
    by_subject = Counter(f"{row.language}:{row.subject}" for row in rows)
    return {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "created": "2026-06-23",
        "rows_grpo": len(rows),
        "rows_dpo": dpo_pair_count(rows),
        "selected_sources": selected_sources,
        "include_swedish_medical": include_swedish_medical,
        "license": "Mixed; filter rows by license_id, license_category, source_id, and redistribution_status.",
        "by_language": dict(sorted(by_lang.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_license_id": dict(sorted(by_license_id.items())),
        "by_license_category": dict(sorted(by_license_category.items())),
        "by_redistribution_status": dict(sorted(by_redistribution_status.items())),
        "source_licenses": {
            source_id: {
                "rows": by_source[source_id],
                "source_license": SOURCE_META.get(source_id, {}).get("source_license", "unknown"),
                "license_id": SOURCE_META.get(source_id, {}).get("license_id", "unknown"),
                "redistribution_status": SOURCE_META.get(source_id, {}).get("redistribution_status", "unknown"),
                "source_url": SOURCE_META.get(source_id, {}).get("source_url", ""),
            }
            for source_id in sorted(by_source)
        },
        "top_language_subjects": dict(by_subject.most_common(50)),
    }


def write_readme(out_dir: Path, manifest: dict) -> None:
    license_lines = "\n".join(f"- `{key}`: {value} rows" for key, value in manifest["by_license_id"].items())
    source_lines = "\n".join(
        f"- `{source_id}`: {meta['rows']} rows, `{meta['license_id']}`, `{meta['redistribution_status']}`"
        for source_id, meta in manifest["source_licenses"].items()
    )
    text = f"""---
license: other
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
- license-filterable
---

# {DATASET_ID}

Real-source multiple-choice data for European-language GRPO/RLVR and DPO
training.

This build contains:

- GRPO/RLVR rows: {manifest['rows_grpo']}
- DPO pairs: {manifest['rows_dpo']}
- Languages: {', '.join(manifest['by_language'].keys())}

## License Filtering

This is a mixed-license dataset. Filter rows before training or redistribution by
`license_id`, `license_category`, `source_id`, and `redistribution_status`.

Licenses in this build:

{license_lines}

Sources:

{source_lines}

## Files

- `grpo/train.jsonl`, `grpo/validation.jsonl`, `grpo/test.jsonl`
- `dpo/train.jsonl`, `dpo/validation.jsonl`, `dpo/test.jsonl`
- `manifest.json`
- `source_registry.json`

GRPO rows use `reward_type=mcq_letter_exact`: reward a response whose first
answer letter matches `answer`.

DPO rows are generated as correct-letter responses preferred over each incorrect
letter for the same prompt.

## Source Registry

See `source_registry.json` for official archives and source-level license notes.
Some official exam archives are link-only until redistribution rights are
cleared.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_sources(value: str) -> list[str]:
    sources = []
    for source in parse_csv(value):
        canonical = SOURCE_ALIASES.get(source)
        if not canonical:
            raise SystemExit(f"Unknown source '{source}'. Known sources: {', '.join(sorted(SOURCE_ALIASES))}")
        if canonical not in sources:
            sources.append(canonical)
    return sources


def apply_filters(
    rows: list[McqRow],
    license_allowlist: list[str],
    redistribution_status_allowlist: list[str],
) -> list[McqRow]:
    license_allow = {canonical_license_id(item) for item in license_allowlist}
    redistribution_allow = set(redistribution_status_allowlist)
    if license_allow:
        rows = [row for row in rows if row.license_id in license_allow]
    if redistribution_allow:
        rows = [row for row in rows if row.redistribution_status in redistribution_allow]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--exams-repo", type=Path, default=DEFAULT_EXAMS_REPO)
    parser.add_argument("--hogskoleprovet-manifest", type=Path, default=DEFAULT_HOGSKOLEPROVET_MANIFEST)
    parser.add_argument("--pdf-cache-dir", type=Path, default=DEFAULT_PDF_CACHE)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="Comma-separated source list or aliases.")
    parser.add_argument("--license-allowlist", default="", help="Comma-separated license IDs, e.g. apache-2.0,mit.")
    parser.add_argument("--redistribution-status-allowlist", default="", help="Comma-separated redistribution status values.")
    parser.add_argument("--include-swedish-medical", action="store_true")
    parser.add_argument("--swedish-grpo-dir", type=Path, default=DEFAULT_SWEDISH_GRPO)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    selected_sources = parse_sources(args.sources)
    rows: list[McqRow] = []
    for source in selected_sources:
        if source == "exams_qa":
            rows.extend(load_exams(args.exams_repo))
        elif source == "hogskoleprovet_ord":
            rows.extend(load_hogskoleprovet_ord(args.hogskoleprovet_manifest, args.pdf_cache_dir))
        elif source == "llmzszl":
            rows.extend(load_llmzszl())
        elif source == "polish_pes_medical":
            rows.extend(load_polish_pes_medical())
        elif source == "swedish_medical_exams_hf":
            rows.extend(load_swedish_medical_exams_hf())
        elif source == "polish_matura_dokato":
            rows.extend(load_polish_matura_dokato())
        elif source == "slovak_mathbio_dokato":
            rows.extend(load_slovak_mathbio_dokato())
        elif source == "slovak_financial_exam":
            rows.extend(load_slovak_financial_exam())
        elif source == "basque_public_exams":
            rows.extend(load_basque_public_exams())
        elif source == "catalan_public_exams":
            rows.extend(load_catalan_public_exams())
        elif source == "global_mmlu":
            rows.extend(load_global_mmlu())
        elif source == "mmmlu":
            rows.extend(load_mmmlu())
        elif source == "belebele":
            rows.extend(load_belebele())
        elif source == "xcopa":
            rows.extend(load_xcopa())
        else:
            raise AssertionError(source)

    if args.include_swedish_medical:
        rows.extend(load_swedish_grpo(args.swedish_grpo_dir))

    rows = apply_filters(rows, parse_csv(args.license_allowlist), parse_csv(args.redistribution_status_allowlist))
    random.Random(args.seed).shuffle(rows)

    by_split = split_rows(rows)
    for split, split_rows_ in by_split.items():
        write_jsonl(args.out_dir / "grpo" / f"{split}.jsonl", (asdict(row) for row in split_rows_))
        write_jsonl(args.out_dir / "dpo" / f"{split}.jsonl", (pair for row in split_rows_ for pair in row_to_dpo_pairs(row)))

    manifest = build_manifest(rows, selected_sources, args.include_swedish_medical)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.source_registry.exists():
        (args.out_dir / "source_registry.json").write_text(args.source_registry.read_text(encoding="utf-8"), encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
