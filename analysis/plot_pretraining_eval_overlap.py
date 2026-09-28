#!/usr/bin/env python3
"""Create plots for the pretraining/evaluation overlap analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "analysis" / "pretraining_eval_overlap.csv"
OUT_DIR = ROOT / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _short_label(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("prepared/"):
        stem = stem[len("prepared/") :]
    if stem.endswith(".joblib"):
        stem = stem[: -len(".joblib")]
    return stem


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("overlap_pct", ascending=True).reset_index(drop=True)

    df["dataset_label"] = df["dataset"].map(_short_label)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df["dataset_label"], df["overlap_pct"], color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Pretraining/evaluation overlap by benchmark dataset")
    ax.set_xlabel("Overlap (%)")
    ax.set_ylabel("Dataset")
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(False)
    for bar, value in zip(bars, df["overlap_pct"]):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pretraining_eval_overlap_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["n_eval"], df["overlap_pct"], s=60, c="darkorange", edgecolors="black")
    for _, row in df.iterrows():
        ax.annotate(
            _short_label(row["dataset"]),
            (row["n_eval"], row["overlap_pct"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Overlap percentage vs dataset size")
    ax.set_xlabel("Evaluation dataset size (n_eval)")
    ax.set_ylabel("Overlap (%)")
    ax.set_xlim(0, max(df["n_eval"]) * 1.08)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pretraining_eval_overlap_scatter.png", dpi=200)
    plt.close(fig)

    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
