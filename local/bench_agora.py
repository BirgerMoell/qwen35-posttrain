#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""Local speed and quality check of Agora builds (Ollama GGUF quants, MLX), direct mode (thinking off), greedy.

    uv run local/bench_agora.py speed --backend ollama --model agora:9b-q4 [--host http://127.0.0.1:11434]
    uv run local/bench_agora.py quality --backend ollama --model agora:9b-q4 --eval ../oellm-rlvr/data/eu-tasks/eval.jsonl --out runs/q4.jsonl
    uv run --with mlx-lm local/bench_agora.py speed --backend mlx --model <mlx dir>

speed: a fixed set of prompts (EN/SV/DE/FR chat, a 1.5K-token document, code), 256 new tokens each after a warm-up;
reports prompt-processing and generation tokens/s (Ollama: its own eval counters; MLX: mlx-lm's counters).
quality: a fixed stratified sample of the EU task suites (eu_qa + eu_doc, the RL verifiers' own scoring via
oellm_rlvr.eu_tasks) so quants can be compared on exactly the same items; `--score` prints the comparison.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

DOC = ("Protokoll från styrelsemötet. " + "Punkt {i}: Styrelsen diskuterade budgeten för kvartalet, beslutade att skjuta upp "
       "investeringen i nya servrar och gav kansliet i uppdrag att ta fram ett nytt förslag till nästa möte. ") * 1
SPEED_PROMPTS = [
    "Explain in a few paragraphs how a heat pump works.",
    "Skriv ett kort mejl till en kollega där du föreslår ett möte nästa vecka om projektplanen.",
    "Erkläre den Unterschied zwischen Wetter und Klima für ein zehnjähriges Kind.",
    "Rédige un court paragraphe sur l'histoire de la ville de Lyon.",
    "Sammanfatta protokollet nedan i fem punkter.\n\n" + "".join(DOC.format(i=i) for i in range(1, 40)),
    "Write a Python function that parses ISO 8601 dates without external libraries, with a few tests.",
]


class Ollama:
    def __init__(self, model: str, host: str):
        import httpx
        self.model, self.http = model, httpx.Client(base_url=host, timeout=1200)

    def chat(self, prompt: str, max_tokens: int) -> dict:
        r = self.http.post("/api/chat", json={"model": self.model, "stream": False, "think": False, "keep_alive": "10m",
                                              "messages": [{"role": "user", "content": prompt}],
                                              "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens}}).json()
        return {"text": r["message"]["content"], "prompt_tokens": r.get("prompt_eval_count", 0),
                "prompt_s": r.get("prompt_eval_duration", 0) / 1e9, "gen_tokens": r.get("eval_count", 0),
                "gen_s": r.get("eval_duration", 0) / 1e9, "load_s": r.get("load_duration", 0) / 1e9}

    def unload(self) -> None:
        self.http.post("/api/generate", json={"model": self.model, "keep_alive": 0})


class MLX:
    def __init__(self, model: str):
        from mlx_lm import load
        self.model, self.tok = load(model)

    def chat(self, prompt: str, max_tokens: int) -> dict:
        from mlx_lm import stream_generate
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        out, last = "", None
        for last in stream_generate(self.model, self.tok, text, max_tokens=max_tokens):  # greedy by default
            out += last.text
        return {"text": out, "prompt_tokens": last.prompt_tokens, "prompt_s": last.prompt_tokens / last.prompt_tps,
                "gen_tokens": last.generation_tokens, "gen_s": last.generation_tokens / last.generation_tps,
                "load_s": 0.0, "peak_gb": last.peak_memory}

    def unload(self) -> None:
        pass


def backend(a):
    return MLX(a.model) if a.backend == "mlx" else Ollama(a.model, a.host)


def speed(a) -> None:
    b = backend(a)
    b.chat("Hej!", 8)  # warm-up / load
    runs = [b.chat(p, 256) for p in SPEED_PROMPTS for _ in range(a.repeats)]
    gen = sum(r["gen_tokens"] for r in runs) / sum(r["gen_s"] for r in runs)
    pp = sum(r["prompt_tokens"] for r in runs) / max(1e-9, sum(r["prompt_s"] for r in runs))
    rep = {"backend": a.backend, "model": a.model, "host": a.host if a.backend == "ollama" else None,
           "gen_tok_s": round(gen, 1), "gen_tok_s_median": round(statistics.median(r["gen_tokens"] / r["gen_s"] for r in runs), 1),
           "prompt_tok_s": round(pp, 1), "long_doc_prompt_tokens": runs[4 * a.repeats]["prompt_tokens"],
           "peak_gb": max((r.get("peak_gb", 0) for r in runs), default=0)}
    print(json.dumps(rep))
    if a.unload:
        b.unload()


def sample_eval(path: Path, n_qa: int, n_doc: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rng = random.Random(20261028)
    qa = [r for r in rows if r["suite"] == "qa"]
    doc = [r for r in rows if r["suite"] != "qa"]
    rng.shuffle(qa), rng.shuffle(doc)
    return qa[:n_qa] + doc[:n_doc]


def quality(a) -> None:
    b = backend(a)
    items = sample_eval(Path(a.eval), a.n_qa, a.n_doc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f:
        for k, r in enumerate(items, 1):
            if r["id"] in done:
                continue
            res = b.chat(r["prompt"], a.max_tokens)
            f.write(json.dumps({**r, "mode": "default", "raw": res["text"], "new_tokens": res["gen_tokens"]}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"{k}/{len(items)}", flush=True)
    if a.unload:
        b.unload()


def score(a) -> None:
    sys.path.insert(0, str(Path(a.rlvr) / "src"))
    from oellm_rlvr import eu_tasks as E
    for path in a.runs:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
        qa = [E.grade_qa(r["raw"], {**r["label"], "mode": "default"}) for r in rows if r["suite"] == "qa"]
        doc = [E.score_doc(r["raw"], {**r["label"], "mode": "default"}) for r in rows if r["suite"] != "qa"]
        n = max(1, len(qa))
        correct, wrong = qa.count("correct"), sum(g in ("wrong", "no_answer") for g in qa)
        print(json.dumps({"run": Path(path).stem, "qa_n": len(qa), "qa_acc": round(correct / n, 3), "qa_abstain": round(qa.count("abstain") / n, 3),
                          "qa_index": round(100 * (correct - wrong) / n, 1), "doc_n": len(doc),
                          "doc_score": round(sum(doc) / max(1, len(doc)), 3), "doc_exact": round(sum(s == 1.0 for s in doc) / max(1, len(doc)), 3)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["speed", "quality", "score"])
    ap.add_argument("--backend", choices=["ollama", "mlx"], default="ollama")
    ap.add_argument("--model", default="agora")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--eval", default=str(Path(__file__).resolve().parents[2] / "oellm-rlvr/data/eu-tasks/eval.jsonl"))
    ap.add_argument("--rlvr", default=str(Path(__file__).resolve().parents[2] / "oellm-rlvr"))
    ap.add_argument("--n-qa", type=int, default=150)
    ap.add_argument("--n-doc", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out")
    ap.add_argument("--unload", action="store_true", help="unload the model afterwards (keeps the Mac cool)")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    {"speed": speed, "quality": quality, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
