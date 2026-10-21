#!/usr/bin/env python3
"""Render model-card figures from the OELLM 9B long-context NIAH reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "niah"
OUTPUT = ROOT / "model_cards" / "assets"

CONTEXTS = [512, 1024, 2048, 4096, 8192, 32768, 131072, 262144]
LABELS = ["512", "1K", "2K", "4K", "8K", "32K", "128K", "~262K"]

WORD_REPORTS = [
    RESULTS / "oellm9b-short-words-max64.json",
    RESULTS / "oellm9b-long-words-max64.json",
]
HEX_REPORTS = [
    RESULTS / "oellm9b-short-context-max64.json",
    RESULTS / "oellm9b-4k-control-max64.json",
    RESULTS / "oellm9b-256k-sft-max64.json",
]

BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#16202A"
GRID = "#D7DEE7"


def load_rates(paths: list[Path]) -> tuple[list[float], list[float]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        report = json.loads(path.read_text())
        for result in report["results"]:
            grouped[result["requested_length"]].append(result)

    missing = [length for length in CONTEXTS if length not in grouped]
    if missing:
        raise ValueError(f"Missing contexts in reports: {missing}")

    retrieval, exact = [], []
    for length in CONTEXTS:
        rows = grouped[length]
        if len(rows) != 5:
            raise ValueError(f"Expected five depths at {length}, got {len(rows)}")
        retrieval.append(100 * sum(row["contains_match"] for row in rows) / len(rows))
        exact.append(100 * sum(row["strict_exact_match"] for row in rows) / len(rows))
    return retrieval, exact


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8B98A8")
    ax.grid(axis="y", color=GRID, linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(colors=INK)
    ax.set_xlabel("Requested context length (tokens)", color=INK, labelpad=10)
    ax.set_ylabel("Cases with full correct answer", color=INK, labelpad=10)


def save_retrieval_comparison(word_retrieval: list[float], hex_retrieval: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=180)
    x = list(range(len(CONTEXTS)))
    style_axis(ax)
    ax.plot(
        x,
        word_retrieval,
        color=BLUE,
        marker="o",
        markersize=7,
        linewidth=2.7,
        label="Natural word passphrase",
    )
    ax.plot(
        x,
        hex_retrieval,
        color=ORANGE,
        marker="s",
        markersize=6.5,
        linewidth=2.7,
        label="Random hexadecimal code",
    )
    ax.set_xticks(x, LABELS)
    ax.set_title(
        "Full-passphrase retrieval by context length",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
        pad=20,
    )
    ax.text(
        0,
        1.03,
        "Five needle depths per length · exact published BF16 checkpoint",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#526170",
    )
    for idx, value in enumerate(hex_retrieval):
        ax.annotate(
            f"{value:.0f}%",
            (idx, value),
            xytext=(0, -18 if value == 100 else 10),
            textcoords="offset points",
            ha="center",
            color=ORANGE,
            fontsize=9,
            fontweight="bold",
        )
    ax.annotate(
        "100% at every tested length",
        (3.5, 100),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        color=BLUE,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.text(
        0.99,
        0.012,
        "Metric: complete expected passphrase appears anywhere in the decoded answer",
        ha="right",
        fontsize=8.5,
        color="#657585",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.99, 0.98))
    fig.savefig(OUTPUT / "oellm9b-retrieval-by-context.png", bbox_inches="tight")
    plt.close(fig)


def save_retrieval_vs_format(word_retrieval: list[float], word_exact: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=180)
    x = list(range(len(CONTEXTS)))
    width = 0.36
    style_axis(ax)
    ax.bar(
        [i - width / 2 for i in x],
        word_retrieval,
        width,
        color=BLUE,
        label="Full retrieval",
    )
    exact_bars = ax.bar(
        [i + width / 2 for i in x],
        word_exact,
        width,
        color=ORANGE,
        label="Exact requested format",
    )
    ax.set_xticks(x, LABELS)
    ax.set_title(
        "Retrieval remains intact as format compliance drops",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
        pad=20,
    )
    ax.text(
        0,
        1.03,
        "Natural word passphrases · five needle depths per length",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#526170",
    )
    for bar, value in zip(exact_bars, word_exact):
        ax.annotate(
            f"{value:.0f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=ORANGE,
            fontsize=9,
            fontweight="bold",
        )
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    fig.text(
        0.99,
        0.012,
        "A verbose answer can pass retrieval while failing the exact-output instruction",
        ha="right",
        fontsize=8.5,
        color="#657585",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.99, 0.98))
    fig.savefig(OUTPUT / "oellm9b-retrieval-vs-format.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    word_retrieval, word_exact = load_rates(WORD_REPORTS)
    hex_retrieval, _ = load_rates(HEX_REPORTS)
    save_retrieval_comparison(word_retrieval, hex_retrieval)
    save_retrieval_vs_format(word_retrieval, word_exact)
    print(f"Wrote figures to {OUTPUT}")


if __name__ == "__main__":
    main()
