#!/usr/bin/env python3
"""Plot pretraining/evaluation overlap per benchmark dataset.

Reads analysis/pretraining_eval_overlap.csv (from pretraining_eval_overlap.py) and
writes a single dot plot comparing full-dataset and test-split overlap.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "analysis" / "pretraining_eval_overlap.csv"
OUT_PATH = ROOT / "analysis" / "plots" / "pretraining_eval_overlap.png"

FULL_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"
LINK = "#c3c2b7"


def main() -> None:
    df = pd.read_csv(CSV_PATH).dropna(subset=["overlap_pct"])
    df = df.sort_values("overlap_pct").reset_index(drop=True)
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(df) + 1.4))
    ax.hlines(
        y,
        df[["overlap_pct", "test_overlap_pct"]].min(axis=1),
        df[["overlap_pct", "test_overlap_pct"]].max(axis=1),
        color=LINK,
        linewidth=2,
        zorder=1,
    )
    ax.scatter(
        df["overlap_pct"],
        y,
        s=48,
        color=FULL_COLOR,
        edgecolors="white",
        linewidths=1.5,
        label="Full dataset",
        zorder=3,
    )
    ax.scatter(
        df["test_overlap_pct"],
        y,
        s=48,
        color=TEST_COLOR,
        marker="D",
        edgecolors="white",
        linewidths=1.5,
        label="Test split",
        zorder=3,
    )

    ax.set_yticks(list(y), df["dataset"], fontsize=8, color=INK)
    ax.set_xlim(0, 102)
    ax.set_xlabel("Molecules seen in ChEMBL pretraining set (%)", color=MUTED_INK)
    ax.set_title("Benchmark overlap with pretraining data", color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED_INK, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINK)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=MUTED_INK)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
