#!/usr/bin/env python3
"""Write the "At a glance" and "Evaluation" sections of the A4b model card from assets/evals.json.

Replaces the text between the markers <!-- glance:start --> / <!-- glance:end --> (inserted after the intro list if
absent) and the whole "## Evaluation" section (up to "## Usage"). Image links point at the published repo files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "birgermoell/oellm-9b-256k-stage5-a4b-tool-rl-step48"
IMG = f"https://huggingface.co/{REPO}/resolve/main/assets"


def fmt(v, bold=False):
    if v is None:
        return "–"
    if v == "pending":
        return "*running*"
    s = f"{v:.3f}"
    return f"**{s}**" if bold else s


def arrow(new, old, tol=0.01):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return ""
    d = new - old
    if abs(d) < tol:
        return "≈ same"
    return f"{'▲' if d > 0 else '▼'} {d:+.3f}"


def glance(data: dict) -> str:
    ck, this = data["checkpoints"], data["this_model"]
    t, s3 = ck[this], ck["3 Tool-use SFT"]
    rows = [
        ("Chat capabilities in 36 European languages", "EUROPA Eval", "europa_eval"),
        ("Safety and behaviour", "EUROPA Alignment", "europa_alignment"),
        ("Math, step by step (thinking mode)", "quick reasoning eval", "thinking_acc"),
        ("Math, direct answers", "quick reasoning eval", "direct_acc"),
        ("Thinking traces that finish", "share of traces closed within 6K tokens", "thinking_finished"),
        ("Tool calling: right call or no call", "held-out tool check", "tool_decision"),
        ("Tool calling: correct tool names", "held-out tool check", "tool_names"),
        ("Tool calling: valid arguments", "held-out tool check", "tool_args"),
        ("Code", "HumanEval+ (direct mode)", "humaneval_plus_direct"),
        ("Reading comprehension", "Belebele", "belebele"),
        ("Knowledge", "Global-MMLU", "global_mmlu"),
        ("Regional knowledge", "INCLUDE", "include"),
        ("Grammar", "MultiBLiMP", "multiblimp"),
    ]
    lines = [
        "<!-- glance:start -->",
        "## At a glance",
        "",
        "> **This model** is step 6 of the pipeline below (tool-call RL). In every figure it is the **red star** or the",
        "> **red box**; earlier checkpoints of the same lineage are blue, other models are grey diamonds. All scores are",
        "> on a 0–1 scale, higher is better.",
        "",
        f"![Per-domain scores of this model and its earlier checkpoints]({IMG}/overview.png)",
        "",
        "| Domain | Measured by | This model | Tool-use SFT (step 3) | Change |",
        "|---|---|---|---|---|",
    ]
    for domain, bench, key in rows:
        lines.append(f"| {domain} | {bench} | {fmt(t.get(key), True)} | {fmt(s3.get(key))} | {arrow(t.get(key), s3.get(key))} |")
    lines += [
        "",
        "**In short.**",
        "- **Strongest:** tool calling (the best call decisions and tool names in the lineage, valid JSON 99.5 %);",
        "  reading comprehension and grammar in European languages (Belebele 0.82, MultiBLiMP 0.93, above the project's",
        "  other SFT and DPO models); safe and well-formed behaviour overall (EUROPA Alignment 0.67, best in the lineage).",
        "- **Knowledge kept:** DPO, merging and RL left Belebele, Global-MMLU, INCLUDE and MultiBLiMP within ±0.005 of",
        "  the tool-use SFT, +4 to +12 points above the base.",
        "- **Improved by DPO, merge and RL:** EUROPA Eval 0.62 → 0.67, mostly from quantitative answers (0.04 → 0.31),",
        "  agentic control and fewer malformed answers (122 → 100).",
        "- **Weakest:** thinking mode often does not finish (about half the traces hit the 6K-token limit), so thinking",
        "  mode is less accurate than direct mode; code is weak (HumanEval+ 0.35 / MBPP+ 0.36 direct, lower in thinking",
        "  mode where many traces do not finish); grounding & safety (0.70 →",
        "  0.58), refusal calibration (0.44 → 0.35) and direct-mode EU-language math (0.45 → 0.31) regressed after DPO.",
        "- **Compared with other models:** above Apertus-8B-Instruct on EUROPA Eval (0.57) but below the project's",
        "  instruct SFT (0.76) and DPO (0.84) models, which come from a different SFT lineage.",
        "<!-- glance:end -->",
    ]
    return "\n".join(lines)


def evaluation(data: dict) -> str:
    ck, this, ref = data["checkpoints"], data["this_model"], data["references"]
    order = ["Base", "1 Instruct SFT", "2 Reasoning SFT", "3 Tool-use SFT", "4 DPO v2", "5 Merge", this]
    label = lambda c: "**6 Tool-call RL (this model)**" if c == this else c  # noqa: E731
    out = ["## Evaluation", "",
           "Every number below was produced with the scripts in the repositories linked above, with greedy decoding.",
           "The checkpoints are the stages of this model's own lineage:", "",
           "| Checkpoint | What it is |", "|---|---|",
           "| Base | the pretrained OpenEuroLLM 9B 256K base (no chat ability) |",
           "| 1 Instruct SFT | instruction tuning (direct mode only) |",
           "| 2 Reasoning SFT | adds the opt-in thinking mode |",
           "| 3 Tool-use SFT | adds tool calling (v1.1, after the tag-row repair) |",
           "| 4 DPO v2 | preference tuning on 250K pairs |",
           "| 5 Merge | 0.5 × step 3 + 0.5 × step 4 (restores tool use) |",
           "| **6 Tool-call RL (this model)** | 48 updates of multi-turn tool-call RL from the merge |", "",
           "**What each evaluation measures:**", "",
           "| Evaluation | What it measures | Size | Languages | Notes |", "|---|---|---|---|---|",
           "| EUROPA Eval | practical capabilities: digital agency, grounding & safety, language understanding, practical communication, quantitative reasoning | 432 items | 36 European | direct mode; judged by gemma4:31b; [BirgerMoell/europa-eval](https://github.com/BirgerMoell/europa-eval) |",
           "| EUROPA Alignment | behaviour: agentic control, dangerous capabilities, fairness, wellbeing, privacy, refusal calibration | 576 items | 36 European | direct mode; judged by gemma4:31b; [BirgerMoell/europa-alignment](https://github.com/BirgerMoell/europa-alignment) |",
           "| Quick reasoning eval | math word problems and competition math, final answer checked | 570 items (GSM8K 150, MGSM 200, MATH-500 100, EU-language math 120) | MGSM en/de/es/fr; EU math in 15 languages | both modes; thinking budget 6,144 tokens |",
           "| Held-out tool check | call or no call, tool names, JSON and argument validity, finishing | 361 conversations | 17 | direct mode; never trained on |",
           "| EvalPlus | code correctness on extended hidden tests (pass@1, plus) | HumanEval+ 164, MBPP+ 378 | en | code runs in a contained sandbox |",
           "| oellm-eval v0.3.0 | knowledge and language competence, few-shot without chat template | Belebele 20, Global-MMLU 11, INCLUDE 11, MultiBLiMP 28 tasks | many European | project harness [OpenEuroLLM/oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) |",
           "",
           f"![Every evaluation on every checkpoint]({IMG}/scorecard.png)", "",
           "'–' means the evaluation was not run at that stage.", "",
           "### EUROPA Eval and Alignment", "",
           f"![EUROPA scores along the pipeline]({IMG}/europa_progress.png)", "",
           "| Model | EUROPA Eval | malformed answers | EUROPA Alignment | malformed answers |", "|---|---|---|---|---|"]
    for c in order[1:]:
        r = ck[c]
        if "europa_eval" in r or "europa_alignment" in r:
            b = c == this
            out.append(f"| {label(c)} | {fmt(r.get('europa_eval'), b)} | {r.get('europa_eval_malformed', '–')} | "
                       f"{fmt(r.get('europa_alignment'), b)} | {r.get('europa_alignment_malformed', '–')} |")
    out += ["| *Reference: OpenEuroLLM instruct SFT (Neonkraft)* | 0.762 | 38 | 0.773 | 41 |",
            "| *Reference: OpenEuroLLM threeway DPO* | 0.840 | 9 | – | – |",
            "| *Reference: Apertus-8B-Instruct-2509* | 0.573 | 116 | – | – |", "",
            f"![EUROPA domains]({IMG}/europa_domains.png)", ""]
    for key, title in (("europa_eval_domains", "EUROPA Eval by domain"), ("europa_alignment_domains", "EUROPA Alignment by domain")):
        cols = [c for c in order if key in ck[c]]
        domains = list(ck[this][key])
        out += [f"**{title}**", "", "| Domain | " + " | ".join(label(c) for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for dmn in domains:
            out.append(f"| {dmn.replace('_', ' ')} | " + " | ".join(fmt(ck[c][key].get(dmn), c == this) for c in cols) + " |")
        out.append("")
    out += ["### Reasoning (quick eval)", "",
            "| Model | direct: all | thinking: all | thinking traces finished | GSM8K (d / t) | MGSM (d / t) | MATH-500 (d / t) | EU-language math (d / t) |",
            "|---|---|---|---|---|---|---|---|"]
    for c in order:
        r = ck[c]
        if "quick_subsets" not in r:
            continue
        q = r["quick_subsets"]
        cell = lambda s: f"{q[s]['default']:.2f} / {q[s]['thinking']:.2f}"  # noqa: E731
        out.append(f"| {label(c)} | {fmt(r['direct_acc'], c == this)} | {fmt(r['thinking_acc'], c == this)} | "
                   f"{r['thinking_finished']:.0%} | {cell('gsm8k')} | {cell('mgsm')} | {cell('math500')} | {cell('oellm_math_eu')} |")
    out += ["", "d = direct mode, t = thinking mode. Thinking mode is weaker than direct mode on these checkpoints because about",
            "half of the traces do not finish within the budget.", "",
            "### Tool calling (held-out check, 361 rows)", "",
            "| Model | call decision | tool names | valid JSON | valid arguments | finished |", "|---|---|---|---|---|---|",
            "| 3 Tool-use SFT v1 (before the tag-row repair) | 0.496 | 0.496 | 0.000 | 0.000 | 98.1% |"]
    for c in order:
        r = ck[c]
        if "tool_decision" in r:
            b = c == this
            out.append(f"| {label(c)} | {fmt(r['tool_decision'], b)} | {fmt(r['tool_names'], b)} | {fmt(r['tool_json'], b)} | "
                       f"{fmt(r['tool_args'], b)} | {r['tool_finished']:.1%} |")
    s3, t = ck["3 Tool-use SFT"], ck[this]
    out += ["", "### Code (EvalPlus, pass@1 on the extended tests)", "",
            "| Model | HumanEval+ direct | MBPP+ direct | HumanEval+ thinking | MBPP+ thinking |", "|---|---|---|---|---|",
            f"| 3 Tool-use SFT | {fmt(s3.get('humaneval_plus_direct'))} | {fmt(s3.get('mbpp_plus_direct'))} | {fmt(s3.get('humaneval_plus_thinking'))} | {fmt(s3.get('mbpp_plus_thinking'))} |",
            f"| **6 Tool-call RL (this model)** | {fmt(t.get('humaneval_plus_direct'), True)} | {fmt(t.get('mbpp_plus_direct'), True)} | {fmt(t.get('humaneval_plus_thinking'), True)} | {fmt(t.get('mbpp_plus_thinking'), True)} |",
            "", "Thinking mode scores lower than direct mode because many traces do not finish within 8,192 tokens (this model:",
            "34 % of HumanEval+ and 55 % of MBPP+ traces close). Code is read only after a closed trace.",
            "", "### Knowledge and language (oellm-eval v0.3.0)", "",
            "| Model | Belebele | Global-MMLU | INCLUDE | MultiBLiMP |", "|---|---|---|---|---|"]
    for c in ("Base", "3 Tool-use SFT", this):
        r, b = ck[c], c == this
        out.append(f"| {label(c)} | {fmt(r.get('belebele'), b)} | {fmt(r.get('global_mmlu'), b)} | {fmt(r.get('include'), b)} | {fmt(r.get('multiblimp'), b)} |")
    for name in ("OpenEuroLLM instruct SFT (Neonkraft)", "Apertus-8B-Instruct-2509", "Qwen3-8B"):
        r = ref[name]
        out.append(f"| *Reference: {name}* | {fmt(r.get('belebele'))} | {fmt(r.get('global_mmlu'))} | {fmt(r.get('include'))} | {fmt(r.get('multiblimp'))} |")
    out += ["", "Means over the tasks every compared model completed. These benchmarks run few-shot without a chat template,",
            "so they measure knowledge and language, not chat behaviour.", "",
            "**Measurement notes.**",
            "- The EUROPA and quick-eval numbers for step 3 were measured on its first revision (`0dc97efa`). Revision v1.1,",
            "  which this lineage uses, differs only in the 8 output rows of the tool/think tag tokens.",
            "- The EUROPA judge is gemma4:31b, run locally with a pinned digest. Scores have 95 % confidence intervals of about",
            "  ±0.03–0.04. The numbers behind every figure are in [`assets/evals.json`](" + IMG.replace("/resolve/", "/blob/") + "/evals.json).",
            ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=Path, required=True)
    ap.add_argument("--evals", type=Path, required=True)
    a = ap.parse_args()
    data = json.loads(a.evals.read_text())
    s = a.card.read_text()
    g = glance(data)
    if "<!-- glance:start -->" in s:
        s = s[: s.index("<!-- glance:start -->")] + g + s[s.index("<!-- glance:end -->") + len("<!-- glance:end -->"):]
    else:
        s = s.replace("\n## How it was made", "\n" + g + "\n\n## How it was made", 1)
    start, end = s.index("## Evaluation"), s.index("## Usage")
    s = s[:start] + evaluation(data) + "\n" + s[end:]
    a.card.write_text(s)
    print("card updated:", len(s), "chars")


if __name__ == "__main__":
    main()
