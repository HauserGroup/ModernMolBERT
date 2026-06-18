#!/usr/bin/env python3
"""Regenerate the task-group distribution figure from bundled source data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from modernmolbert.common.paths import project_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_FILENAME = "Fig_task_group_distributions.csv"
DEFAULT_CSV_RELATIVE = Path("paper/source_data") / DATA_FILENAME
REQUIRED_COLUMNS = ("task_group", "task", "model", "roc_auc_x100")
GROUP_ORDER = ["TDC-ADME", "TDC-Tox", "TDC-HTS", "MoleculeNet"]
GROUP_TASK_COUNTS = {
    "TDC-ADME": 12,
    "TDC-Tox": 4,
    "TDC-HTS": 2,
    "MoleculeNet": 7,
}
MODELS = ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer", "MMB-small", "MMB-base"]
MODEL_LABELS = {
    "ECFP4": "ECFP4",
    "ChemBERTa-2": "ChemBERTa-2",
    "SELFormer": "SELFormer",
    "MoLFormer": "MoLFormer",
    "MMB-small": "ModernMolBERT-small",
    "MMB-base": "ModernMolBERT-base",
}
MODEL_COLORS = {
    "ECFP4": "#1B9E77",
    "ChemBERTa-2": "#D95F02",
    "SELFormer": "#7570B3",
    "MoLFormer": "#E7298A",
    "MMB-small": "#66A61E",
    "MMB-base": "#E6AB02",
}
DEFAULT_OUTPUT_RELATIVE = Path("paper/figures/Fig_task_group_distributions.pdf")


def default_csv_path() -> Path:
    """Return the repo-local default source-data path, falling back to the current CWD."""

    try:
        return project_path(DEFAULT_CSV_RELATIVE, start=__file__)
    except FileNotFoundError:
        return DEFAULT_CSV_RELATIVE.resolve()


def default_output_path() -> Path:
    """Return the repo-local default output path, falling back to the current CWD."""

    try:
        return project_path(DEFAULT_OUTPUT_RELATIVE, start=__file__)
    except FileNotFoundError:
        return DEFAULT_OUTPUT_RELATIVE.resolve()


def load_group_distribution_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate task-group ROC-AUC source data.

    Parameters
    ----------
    csv_path:
        Optional external CSV path. When omitted,
        ``paper/source_data/Fig_task_group_distributions.csv`` is used.
    """

    if csv_path is None:
        csv_path = default_csv_path()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)

    validate_group_distribution_data(df)
    df = df.copy()
    df["roc_auc_x100"] = pd.to_numeric(df["roc_auc_x100"], errors="raise")
    return df


def validate_group_distribution_data(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the group-distribution source data is incomplete."""

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    unknown_groups = sorted(set(df["task_group"]) - set(GROUP_ORDER))
    if unknown_groups:
        raise ValueError(f"Unknown task groups: {unknown_groups}")

    unknown_models = sorted(set(df["model"]) - set(MODELS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    duplicated = df.duplicated(["task_group", "task", "model"], keep=False)
    if duplicated.any():
        duplicates = df.loc[duplicated, ["task_group", "task", "model"]]
        raise ValueError(f"Duplicate task/model rows:\n{duplicates.to_string(index=False)}")

    numeric_auc = pd.to_numeric(df["roc_auc_x100"], errors="coerce")
    if numeric_auc.isna().any():
        bad_rows = df.loc[numeric_auc.isna(), ["task_group", "task", "model", "roc_auc_x100"]]
        raise ValueError(f"Non-numeric ROC-AUC values:\n{bad_rows.to_string(index=False)}")

    if not numeric_auc.between(0, 100).all():
        bad_rows = df.loc[
            ~numeric_auc.between(0, 100), ["task_group", "task", "model", "roc_auc_x100"]
        ]
        raise ValueError(f"ROC-AUC values outside [0, 100]:\n{bad_rows.to_string(index=False)}")

    expected_counts = pd.Series(GROUP_TASK_COUNTS, name="expected")
    coverage = (
        df.groupby(["task_group", "model"])["task"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=GROUP_ORDER, columns=MODELS, fill_value=0)
    )
    expected = pd.DataFrame(
        {model: expected_counts for model in MODELS},
        index=GROUP_ORDER,
        dtype="int64",
    )
    if not coverage.equals(expected):
        raise ValueError(
            "Unexpected task coverage by group/model:\n"
            f"{coverage.to_string()}\n\nExpected:\n{expected.to_string()}"
        )


def group_means(df: pd.DataFrame) -> pd.DataFrame:
    """Return group mean ROC-AUC x100 values with canonical ordering."""

    return (
        df.groupby(["task_group", "model"])["roc_auc_x100"]
        .mean()
        .round(1)
        .unstack()
        .reindex(index=GROUP_ORDER, columns=MODELS)
    )


def plot_group_distribution(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Plot per-task ROC-AUC distributions and write the figure to ``output_path``."""

    validate_group_distribution_data(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(13)
    fig, axes = plt.subplots(
        1,
        len(GROUP_ORDER),
        figsize=(13, 4.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1] * len(GROUP_ORDER)},
    )

    for ax, group in zip(axes, GROUP_ORDER, strict=True):
        sub = df[df.task_group == group]
        for xi, model in enumerate(MODELS):
            vals = sub[sub.model == model]["roc_auc_x100"].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                np.full(len(vals), xi) + jitter,
                vals,
                s=42,
                color=MODEL_COLORS[model],
                alpha=0.75,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            if len(vals):
                ax.hlines(
                    vals.mean(),
                    xi - 0.32,
                    xi + 0.32,
                    color="black",
                    linewidth=2.6,
                    zorder=4,
                )

        ax.set_title(
            f"{group} ({GROUP_TASK_COUNTS[group]})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=40, ha="right", fontsize=8.5)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(r"ROC-AUC ($\times$100)", fontsize=11)
    axes[0].set_ylim(0, 100)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(MODELS),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return group_means(df)


def generate_group_distribution_figure(
    *,
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load source data, generate the figure, and return corrected group means."""

    if output_path is None:
        output_path = default_output_path()

    df = load_group_distribution_data(csv_path)
    means = plot_group_distribution(df, output_path)
    if verbose:
        print(f"Wrote {Path(output_path)}")
        print("Group means:")
        print(means)
    return means


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Fig_task_group_distributions from its source CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Source CSV. Default: {default_csv_path()}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output figure path. Default: {default_output_path()}",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the printed group-means table.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_group_distribution_figure(
        csv_path=args.csv,
        output_path=args.out,
        verbose=not args.no_summary,
    )


if __name__ == "__main__":
    main()
