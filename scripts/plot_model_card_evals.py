#!/usr/bin/env python3
"""Figures and a curated JSON for the model cards of the public releases (--release a4b | w2).

Input: the raw eval collection (quick reasoning, held-out tool, EUROPA) gathered from the cluster, plus the
oellm-eval and EvalPlus numbers recorded in the playbook. Output (docs/model_cards/assets/):
  evals.json            - the curated numbers behind every figure and table (one source of truth)
  overview.png          - per-domain scores: every checkpoint of the lineage as a dot, this model as a star
  europa_progress.png   - EUROPA Eval / Alignment across the pipeline stages with reference models
  europa_domains.png    - EUROPA per-domain scores: tool SFT -> merge -> this model
  scorecard.png         - checkpoints x evaluations grid (not run / pending marked)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A4B = "6 Tool-call RL\n(this model)"
W2 = "7 RL window W2\n(this model)"
# oellm-eval v0.3.0 means over the tasks all compared models completed (playbook 8.1) and EvalPlus pass@1 (plus).
OELLM_STAGE = {"Base": {"belebele": 0.737, "global_mmlu": 0.547, "include": 0.422, "multiblimp": 0.935},
               "3 Tool-use SFT": {"belebele": 0.821, "global_mmlu": 0.584, "include": 0.539, "multiblimp": 0.930}}
CODE_STAGE3 = {"3 Tool-use SFT": {"humaneval_plus_direct": 0.335, "mbpp_plus_direct": 0.357,
                                  "humaneval_plus_thinking": 0.189, "mbpp_plus_thinking": 0.243}}
EARLY_KEYS = {  # checkpoint -> result names in the raw collection
    "1 Instruct SFT": {"europa": "birgermoell-oellm-9b-256k-stage1-instruct-v1-huggingface"},
    "2 Reasoning SFT": {"europa": "birgermoell-oellm-9b-256k-stage2-reasoning-v1-huggingface", "quick": "stage2-reasoning-v1"},
    "3 Tool-use SFT": {"europa": "birgermoell-oellm-9b-256k-stage3-tooluse-v1-huggingface", "quick": "stage3-tooluse-v1",
                       "tool": "stage3-tooluse-v1.1"},
    "4 DPO v2": {"quick": "stage4-dpo-v2-250k", "tool": "stage4-dpo-v2-250k"},
    "5 Merge": {"europa": "birgermoell-oellm-9b-256k-stage4-dpo-v2-merge-a05-huggingface", "quick": "stage4-dpo-v2-merge-a0.5",
                "tool": "stage4-dpo-v2-merge-a0.5"},
}
A4B_KEYS = {"europa": "local-oellm-9b-256k-stage5-a4b-tool-rl-step48-huggingface", "quick": "merge-tool-step_48", "tool": "merge-tool-step_48"}
A4B_OELLM = {"belebele": 0.824, "global_mmlu": 0.587, "include": 0.542, "multiblimp": 0.932}  # run 2026-10-26-16-27-08
A4B_CODE = {"humaneval_plus_direct": 0.354, "mbpp_plus_direct": 0.362,  # EvalPlus job 62360803
            "humaneval_plus_thinking": 0.165, "mbpp_plus_thinking": 0.254}
# EU task suites (birgermoell/oellm-eu-tasks), default mode, strict ANSWER line: factual-QA accuracy and document score.
EU_SUITES = {"a4b": {"eu_qa_acc": 0.265, "eu_qa_index": -43.7, "eu_doc": 0.360},
             "w1a": {"eu_qa_acc": 0.273, "eu_qa_index": -42.0, "eu_doc": 0.378},
             "w2": {"eu_qa_acc": 0.280, "eu_qa_index": -39.3, "eu_doc": 0.376}}  # jobs 62423356 / 62423355 / 62430244
RELEASES = {
    "a4b": {
        "this": A4B,
        "lineage": ["Base", "1 Instruct SFT", "2 Reasoning SFT", "3 Tool-use SFT", "4 DPO v2", "5 Merge", A4B],
        "keys": {**EARLY_KEYS, A4B: A4B_KEYS},
        "oellm": {**OELLM_STAGE, A4B: A4B_OELLM},
        "code": {**CODE_STAGE3, A4B: A4B_CODE},
        "extra": {},
        "siblings": [],
        "domain_models": ["3 Tool-use SFT", "5 Merge"],
    },
    "w2": {
        "this": W2,
        "lineage": ["Base", "1 Instruct SFT", "2 Reasoning SFT", "3 Tool-use SFT", "4 DPO v2", "5 Merge", "6 RL window W1a", W2],
        "keys": {**EARLY_KEYS, "6 RL window W1a": {"europa": "local-oellm-9b-w1a-final-huggingface", "quick": "w1a-step_128", "tool": "w1a-step_128"},
                 W2: {"europa": "w2-step_160", "quick": "w2-step_160", "tool": "w2-step_160"},
                 "A4b tool RL (previous release)": A4B_KEYS},
        "oellm": {**OELLM_STAGE, "6 RL window W1a": {"belebele": 0.826, "global_mmlu": 0.589, "include": 0.539, "multiblimp": 0.932},
                  "A4b tool RL (previous release)": A4B_OELLM},
        "code": {**CODE_STAGE3, "6 RL window W1a": {"humaneval_plus_direct": 0.372, "mbpp_plus_direct": 0.362,
                                                    "humaneval_plus_thinking": 0.348, "mbpp_plus_thinking": 0.328},  # job 62278871
                 W2: {"humaneval_plus_direct": 0.390, "mbpp_plus_direct": 0.365,  # job 62354510
                      "humaneval_plus_thinking": 0.299, "mbpp_plus_thinking": 0.333},
                 "A4b tool RL (previous release)": A4B_CODE},
        "extra": {"6 RL window W1a": EU_SUITES["w1a"], W2: EU_SUITES["w2"], "A4b tool RL (previous release)": EU_SUITES["a4b"]},
        "siblings": ["A4b tool RL (previous release)"],
        "domain_models": ["3 Tool-use SFT", "5 Merge", "6 RL window W1a"],
    },
}
CFG = RELEASES["a4b"]
THIS = CFG["this"]
LINEAGE = CFG["lineage"]
REFERENCES = {  # other models on the same EUROPA protocol / oellm-eval tasks
    "OpenEuroLLM instruct SFT (Neonkraft)": {"europa_eval": 0.762, "europa_alignment": 0.773,
                                              "belebele": 0.808, "global_mmlu": 0.573, "include": 0.546, "multiblimp": 0.931},
    "OpenEuroLLM threeway DPO": {"europa_eval": 0.840},
    "Apertus-8B-Instruct-2509": {"europa_eval": 0.573, "belebele": 0.792, "global_mmlu": 0.560, "include": 0.578, "multiblimp": 0.924},
    "Qwen3-8B": {"belebele": 0.865, "global_mmlu": 0.663, "include": 0.587, "multiblimp": 0.866, "eu_qa_acc": 0.157, "eu_doc": 0.742},
}
METRICS = [  # (key, short label, domain)
    ("europa_eval", "EUROPA Eval", "Chat capabilities"),
    ("europa_alignment", "EUROPA Alignment", "Alignment & safety"),
    ("thinking_acc", "Math, thinking mode", "Reasoning"),
    ("direct_acc", "Math, direct mode", "Reasoning"),
    ("thinking_finished", "Thinking traces finished", "Reasoning"),
    ("tool_decision", "Tool: call decision", "Tool use"),
    ("tool_args", "Tool: valid arguments", "Tool use"),
    ("humaneval_plus_direct", "HumanEval+ (direct)", "Code"),
    ("mbpp_plus_direct", "MBPP+ (direct)", "Code"),
    ("belebele", "Belebele (reading)", "Knowledge & language"),
    ("global_mmlu", "Global-MMLU", "Knowledge & language"),
    ("include", "INCLUDE (regional)", "Knowledge & language"),
    ("multiblimp", "MultiBLiMP (grammar)", "Knowledge & language"),
]
EU_METRICS = [("eu_qa_acc", "EU factual QA (accuracy)", "Knowledge & language"),
              ("eu_doc", "EU document tasks", "Practical tasks")]


def metrics() -> list:
    return METRICS + (EU_METRICS if CFG["extra"] else [])


def curate(raw: dict, pending: dict) -> dict:
    out = {}
    for ckpt in LINEAGE + CFG["siblings"]:
        row, keys = {}, CFG["keys"].get(ckpt, {})
        e = raw["europa"].get(keys.get("europa", ""), {})
        if "europa-eval" in e:
            row["europa_eval"] = e["europa-eval"]["score"]
            row["europa_eval_malformed"] = e["europa-eval"]["malformed"]
            row["europa_eval_domains"] = e["europa-eval"]["domains"]
        if "europa-alignment" in e:
            row["europa_alignment"] = e["europa-alignment"]["score"]
            row["europa_alignment_malformed"] = e["europa-alignment"]["malformed"]
            row["europa_alignment_domains"] = e["europa-alignment"]["domains"]
        q = raw["quick"].get(keys.get("quick", ""))
        if q:
            row["thinking_acc"] = q["ALL"]["thinking"]["accuracy"]
            row["thinking_finished"] = 1 - q["ALL"]["thinking"]["unfinished_rate"]
            row["direct_acc"] = q["ALL"]["default"]["accuracy"]
            row["quick_subsets"] = {s: {m: q[s][m]["accuracy"] for m in q[s]} for s in q if s != "ALL" and not s.startswith("lang:")}
        t = raw["tool"].get(keys.get("tool", ""))
        if t:
            row["tool_decision"], row["tool_names"] = t["decision"], t["names"]
            row["tool_json"], row["tool_args"], row["tool_finished"] = t["json_ok"], t["args_ok"], t["terminated"]
        row.update(CFG["oellm"].get(ckpt, {}))
        row.update(CFG["code"].get(ckpt, {}))
        row.update(CFG["extra"].get(ckpt, {}))
        row.update(pending.get(ckpt, {}))
        out[ckpt] = row
    return {"checkpoints": out, "references": REFERENCES, "this_model": THIS, "lineage": LINEAGE, "siblings": CFG["siblings"]}


def plot_overview(data: dict, path: Path) -> None:
    ck = data["checkpoints"]
    rows = metrics()
    fig, ax = plt.subplots(figsize=(10, 7.2 + 0.45 * (len(rows) - len(METRICS))))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(LINEAGE) - 1))
    y = np.arange(len(rows))[::-1]
    for i, (key, label, domain) in enumerate(rows):
        ax.axhline(y[i], color="#eeeeee", lw=8, zorder=0)
        for j, c in enumerate(LINEAGE[:-1]):
            v = ck[c].get(key)
            if isinstance(v, (int, float)):
                ax.scatter(v, y[i], s=55, color=colors[j], edgecolor="white", zorder=2,
                           label=c if i == 0 or key in ("belebele",) and c == "Base" else None)
        for r, ref in data["references"].items():
            v = ref.get(key)
            if v is not None:
                ax.scatter(v, y[i], marker="D", s=40, facecolor="none", edgecolor="#888888", zorder=2)
        for sib in CFG["siblings"]:
            v = ck[sib].get(key)
            if isinstance(v, (int, float)):
                ax.scatter(v, y[i], marker="s", s=42, facecolor="none", edgecolor="#ff7f0e", lw=1.4, zorder=2)
        v = ck[THIS].get(key)
        if isinstance(v, (int, float)):
            ax.scatter(v, y[i], marker="*", s=360, color="#d62728", edgecolor="black", lw=0.6, zorder=3)
            ax.annotate(f"{v:.2f}", (v, y[i]), xytext=(9, -4), textcoords="offset points", fontsize=9, color="#d62728", weight="bold")
        elif v == "pending":
            ax.annotate("this model: running", (0.02, y[i]), xytext=(0, -4), textcoords="offset points", fontsize=8, color="#d62728", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label}\n[{domain}]" for key, label, domain in rows], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score (higher is better; all metrics on a 0-1 scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[j], label=c) for j, c in enumerate(LINEAGE[:-1])]
    handles += [plt.Line2D([], [], marker="*", ls="", markersize=15, color="#d62728", markeredgecolor="black",
                           label="THIS MODEL (" + THIS.split("\n")[0].split(" ", 1)[1] + ")"),
                plt.Line2D([], [], marker="D", ls="", markerfacecolor="none", markeredgecolor="#888888", label="reference models")]
    handles += [plt.Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor="#ff7f0e", label=sib) for sib in CFG["siblings"]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8, frameon=False)
    ax.set_title("How this model and its earlier checkpoints score, by domain", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_europa_progress(data: dict, path: Path) -> None:
    ck = data["checkpoints"]
    stages = [c for c in LINEAGE if "europa_eval" in ck[c]]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for key, label, color in (("europa_eval", "EUROPA Eval (capabilities, 432 items)", "#1f77b4"),
                              ("europa_alignment", "EUROPA Alignment (safety & behaviour, 576 items)", "#2ca02c")):
        vals = [ck[c].get(key, np.nan) for c in stages]
        ax.plot(x, vals, marker="o", color=color, lw=2, label=label)
        ax.scatter([x[-1]], [vals[-1]], marker="*", s=400, color="#d62728", edgecolor="black", zorder=5)
        other = [ck[c].get("europa_eval" if key == "europa_alignment" else "europa_alignment", np.nan) for c in stages]
        for xi, v, o in zip(x, vals, other):
            dy = 9 if (v >= o or np.isnan(o)) else -15  # the higher of the two series is labelled above, the lower below
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8, color=color)
    for name, ref, style in (("OpenEuroLLM threeway DPO (Eval)", data["references"]["OpenEuroLLM threeway DPO"]["europa_eval"], ":"),
                             ("OpenEuroLLM instruct SFT (Eval)", data["references"]["OpenEuroLLM instruct SFT (Neonkraft)"]["europa_eval"], "--"),
                             ("Apertus-8B-Instruct (Eval)", data["references"]["Apertus-8B-Instruct-2509"]["europa_eval"], "-.")):
        ax.axhline(ref, color="#888888", ls=style, lw=1)
        ax.text(x[-1] + 0.35, ref + 0.004, f"{name}: {ref:.3f}", fontsize=7, color="#555555", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" (this model)", "\n(this model)") for c in stages], fontsize=9)
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("score")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("EUROPA scores along the pipeline (red star = this model)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_europa_domains(data: dict, path: Path) -> None:
    ck = data["checkpoints"]
    blues = ["#c6dbef", "#9ecae1", "#4292c6", "#08519c"][-len(CFG["domain_models"]):]
    models = list(zip(CFG["domain_models"], blues)) + [(THIS, "#d62728")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, title in ((axes[0], "europa_eval_domains", "EUROPA Eval domains"),
                           (axes[1], "europa_alignment_domains", "EUROPA Alignment domains")):
        domains = list(ck[THIS][key])
        y = np.arange(len(domains))
        h = 0.78 / len(models)
        for k, (m, color) in enumerate(models):
            vals = [ck[m][key].get(d, np.nan) for d in domains]
            label = "This model" if m == THIS else m
            ax.barh(y + ((len(models) - 1) / 2 - k) * h, vals, height=h, color=color, label=label)
        ax.set_yticks(y)
        ax.set_yticklabels([d.replace("_", " ") for d in domains], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Where the model improved and where it regressed (EUROPA, 36 European languages)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scorecard(data: dict, path: Path) -> None:
    ck = data["checkpoints"]
    cols = metrics()
    rows = LINEAGE[:-1] + CFG["siblings"] + [THIS]
    grid = np.full((len(rows), len(cols)), np.nan)
    text = [["" for _ in cols] for _ in rows]
    for i, c in enumerate(rows):
        for j, (key, label, domain) in enumerate(cols):
            v = ck[c].get(key)
            if isinstance(v, (int, float)):
                grid[i, j] = v
                text[i][j] = f"{v:.2f}"
            elif v == "pending":
                text[i][j] = "running"
            else:
                text[i][j] = "–"
    fig, ax = plt.subplots(figsize=(13 + 0.8 * (len(cols) - len(METRICS)), 4.8 + 0.4 * (len(rows) - 7)))
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad("#f2f2f2")
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            weight = "bold" if rows[i] == THIS else "normal"
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8, weight=weight,
                    color="#555555" if text[i][j] in ("–", "running") else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{label}\n[{domain}]" for key, label, domain in cols], fontsize=7, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.replace("\n", " ") for r in rows], fontsize=9)
    ax.get_yticklabels()[-1].set_color("#d62728")
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.add_patch(plt.Rectangle((-0.5, len(rows) - 1.5), len(cols), 1, fill=False, edgecolor="#d62728", lw=2.5))
    ax.set_title("Every evaluation on every checkpoint ('–' = not run at that stage; red box = this model)", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pending", default="{}", help='JSON {checkpoint: {metric: value or "pending"}}')
    ap.add_argument("--release", choices=sorted(RELEASES), default="a4b")
    a = ap.parse_args()
    global CFG, THIS, LINEAGE
    CFG = RELEASES[a.release]
    THIS, LINEAGE = CFG["this"], CFG["lineage"]
    a.out.mkdir(parents=True, exist_ok=True)
    data = curate(json.loads(a.raw.read_text()), json.loads(a.pending))
    (a.out / "evals.json").write_text(json.dumps(data, indent=1) + "\n")
    plot_overview(data, a.out / "overview.png")
    plot_europa_progress(data, a.out / "europa_progress.png")
    plot_europa_domains(data, a.out / "europa_domains.png")
    plot_scorecard(data, a.out / "scorecard.png")
    print("wrote", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
