#!/usr/bin/env python3
"""Rebuild the descriptive pretraining sweep figure from archived run metrics."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMALL_LEARNING_RATES = {"span": 0.0002, "hetero_span": 0.0004, "standard": 0.0004}
BASE_LEARNING_RATES = (0.0002, 0.0004, 0.0008)
FIELDS = (
    "run",
    "size",
    "strategy",
    "mlm_prob",
    "learning_rate",
    "final_eval_masked_accuracy",
    "final_eval_loss",
)


def selected_rows(source: Path) -> list[dict[str, str]]:
    with source.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 36:
        raise ValueError("Expected all 27 small and 9 base archived sweep runs")
    rows = []
    keys = set()
    for row in raw:
        key = (row["size"], row["strategy"], float(row["mlm_prob"]), float(row["learning_rate"]))
        if key in keys:
            raise ValueError(f"Duplicate sweep condition: {key}")
        keys.add(key)
        if row["size"] == "small":
            include = float(row["learning_rate"]) == SMALL_LEARNING_RATES[row["strategy"]]
        else:
            include = row["size"] == "base" and row["strategy"] == "standard"
        if include:
            for field in ("final_eval_masked_accuracy", "final_eval_loss"):
                if not 0 < float(row[field]) < 2:
                    raise ValueError(f"Invalid archived metric in {row['run']}: {field}")
            rows.append({field: row[field] for field in FIELDS})
    expected = {
        ("small", strategy, probability, learning_rate)
        for strategy, learning_rate in SMALL_LEARNING_RATES.items()
        for probability in (0.15, 0.2, 0.25)
    } | {
        ("base", "standard", probability, learning_rate)
        for learning_rate in BASE_LEARNING_RATES
        for probability in (0.15, 0.2, 0.25)
    }
    actual = {
        (row["size"], row["strategy"], float(row["mlm_prob"]), float(row["learning_rate"]))
        for row in rows
    }
    if actual != expected:
        raise ValueError(
            f"Wrong plotted conditions: missing={expected - actual}, extra={actual - expected}"
        )
    return sorted(
        rows,
        key=lambda row: (
            row["size"],
            row["strategy"],
            float(row["learning_rate"]),
            float(row["mlm_prob"]),
        ),
    )


def plot(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex="col")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.21, hspace=0.10, wspace=0.18)
    colors = {
        "span": "#009e73",
        "hetero_span": "#d55e00",
        "standard": "#0072b2",
        0.0002: "#0072b2",
        0.0004: "#009e73",
        0.0008: "#cc79a7",
    }
    for column, size in enumerate(("small", "base")):
        if size == "small":
            groups = [(strategy, rate) for strategy, rate in SMALL_LEARNING_RATES.items()]
        else:
            groups = [("standard", rate) for rate in BASE_LEARNING_RATES]
        for strategy, rate in groups:
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["size"] == size
                    and row["strategy"] == strategy
                    and float(row["learning_rate"]) == rate
                ),
                key=lambda row: float(row["mlm_prob"]),
            )
            label = (
                f"{strategy.replace('_', ' ').title()}, {rate:g}"
                if size == "small"
                else f"{rate:g}"
            )
            color = colors[strategy if size == "small" else rate]
            for row_index, field in enumerate(("final_eval_masked_accuracy", "final_eval_loss")):
                axes[row_index, column].plot(
                    [float(row["mlm_prob"]) for row in subset],
                    [float(row[field]) for row in subset],
                    marker="o",
                    linewidth=1.7,
                    markersize=4.5,
                    color=color,
                    label=label,
                )
        axes[0, column].set_title(f"{size.title()} model")
        axes[1, column].set_xlabel("MLM masking probability")
        axes[1, column].set_xticks((0.15, 0.2, 0.25))
        for row_index in (0, 1):
            axes[row_index, column].grid(axis="y", color="0.85", linewidth=0.6)
            axes[row_index, column].spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, column].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Small strategy / learning rate" if size == "small" else "Base learning rate",
            frameon=False,
            fontsize=7.3,
            title_fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.29 if size == "small" else 0.77, 0.01),
            ncol=3,
        )
    axes[0, 0].set_ylabel("Validation masked accuracy")
    axes[1, 0].set_ylabel("Validation MLM loss")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    args = parser.parse_args()
    rows = selected_rows(args.source)
    plot(rows, args.figure)
    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    with args.source_data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
