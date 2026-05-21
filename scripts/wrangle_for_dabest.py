# %%
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = "../data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv"
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CORR_METHOD = "spearman"
N_BINS = 25


# =============================================================================
# Plotting utilities
# =============================================================================


def clean_axis(ax):
    """Apply a simple, consistent axis style."""
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_shared_bins(values, n_bins=25):
    """Return shared histogram bins for comparable overlaid histograms."""
    values = pd.Series(values).dropna()

    if values.empty:
        raise ValueError("Cannot create bins from empty values.")

    if values.min() == values.max():
        center = values.iloc[0]
        return np.linspace(center - 0.5, center + 0.5, n_bins + 1)

    return np.linspace(values.min(), values.max(), n_bins + 1)


def plot_overlay_histograms(
    df,
    group_cols,
    value_col="metric_value",
    n_bins=25,
    density_options=(False, True),
    figsize_per_panel=(7, 4.5),
):
    """
    Plot related overlaid histograms on the same figure.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the value column and grouping columns.
    group_cols : list[str]
        Columns to use for grouping in separate panels.
    value_col : str
        Numeric column to histogram.
    n_bins : int
        Number of histogram bins.
    density_options : tuple[bool]
        Whether to plot count, density, or both.
    figsize_per_panel : tuple[float, float]
        Approximate size of each subplot panel.
    """
    plot_df = df.dropna(subset=[value_col]).copy()
    bins = make_shared_bins(plot_df[value_col], n_bins=n_bins)

    n_rows = len(density_options)
    n_cols = len(group_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for row_idx, density in enumerate(density_options):
        for col_idx, group_col in enumerate(group_cols):
            ax = axes[row_idx, col_idx]

            sub_df = plot_df.dropna(subset=[group_col])

            for group, group_df in sub_df.groupby(group_col, sort=True):
                ax.hist(
                    group_df[value_col],
                    bins=bins,
                    alpha=0.45,
                    label=str(group),
                    density=density,
                    edgecolor="white",
                    linewidth=0.8,
                )

            y_label = "Density" if density else "Count"
            title_suffix = "density" if density else "counts"

            ax.set_title(
                f"{value_col} by {group_col} ({title_suffix})",
                fontsize=13,
                weight="bold",
            )
            ax.set_xlabel(value_col)
            ax.set_ylabel(y_label)
            ax.legend(title=group_col, frameon=False)
            clean_axis(ax)

    fig.tight_layout()
    return fig, axes


def plot_correlation_heatmap(
    wide_df,
    method="spearman",
    title="Correlation heatmap",
    min_periods=2,
    ax=None,
):
    """
    Plot a correlation heatmap from a wide data frame.

    Columns are correlated against each other.
    Rows are paired observations.
    """
    corr = wide_df.corr(method=method, min_periods=min_periods)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(corr, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.iloc[i, j]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10)

    ax.set_title(title, fontsize=13, weight="bold")
    clean_axis(ax)

    return fig, ax, corr, im


def plot_metric_pair_scatters(
    metric_wide,
    method="pearson",
    max_cols=3,
    figsize_per_panel=(5.5, 4.5),
):
    """
    Plot all pairwise scatter plots between metric columns on one figure.

    Parameters
    ----------
    metric_wide : pd.DataFrame
        Wide data frame with one metric per column.
    method : str
        Correlation method used for displayed r.
    max_cols : int
        Maximum subplot columns.
    """
    metric_cols = list(metric_wide.columns)
    pairs = list(combinations(metric_cols, 2))

    if not pairs:
        raise ValueError("Need at least two metric columns for pairwise scatter plots.")

    n_cols = min(max_cols, len(pairs))
    n_rows = math.ceil(len(pairs) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for ax, (x_metric, y_metric) in zip(flat_axes, pairs, strict=False):
        sub = metric_wide[[x_metric, y_metric]].dropna()

        ax.scatter(
            sub[x_metric],
            sub[y_metric],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.6,
        )

        if len(sub) >= 2:
            r = sub[[x_metric, y_metric]].corr(method=method).iloc[0, 1]
            r_label = "NA" if pd.isna(r) else f"{r:.2f}"
        else:
            r_label = "NA"

        ax.set_title(
            f"{x_metric} vs {y_metric}\nr = {r_label}",
            fontsize=12,
            weight="bold",
        )
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        clean_axis(ax)

    for ax in flat_axes[len(pairs) :]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


# =============================================================================
# Load raw data
# =============================================================================

df = pd.read_csv(INPUT_CSV)

# `df` is the original benchmark table.
# Important raw columns:
# - dataset: benchmark dataset name; this is the paired unit going forward
# - embedder: representation method to compare
# - model: nuisance model type
# - *_metric: numeric performance columns
# - *_metric_name: optional human-readable metric names, if present


# =============================================================================
# Create strict long data frame
# =============================================================================

metric_value_cols = [c for c in df.columns if c.endswith("_metric")]
metric_name_cols = [f"{c}_name" for c in metric_value_cols]

# `long_df` is the strict long-format data frame.
# It preserves model-level results before any max-collapse.
# Columns:
# - dataset
# - embedder
# - model
# - metric_name: source metric column, e.g. cv_metric or test_metric
# - metric_value: numeric value
long_df = df.melt(
    id_vars=["dataset", "embedder", "model"],
    value_vars=metric_value_cols,
    var_name="metric_name",
    value_name="metric_value",
)

long_df = long_df.dropna(subset=["metric_value"])[
    ["dataset", "embedder", "model", "metric_name", "metric_value"]
].reset_index(drop=True)

long_df.to_csv(OUTPUT_DIR / "long_metrics.csv", index=False)


# =============================================================================
# Collapse nuisance model dimension
# =============================================================================

# `best_df` is the main embedder-comparison data frame.
# For each dataset x embedder x metric_name, it keeps the maximum metric_value
# and preserves the model that produced that maximum.
#
# Columns:
# - dataset
# - embedder
# - metric_name
# - metric_value
# - model: winning nuisance model
best_idx = (
    long_df.sort_values(
        ["dataset", "embedder", "metric_name", "metric_value", "model"],
        ascending=[True, True, True, False, True],
    )
    .drop_duplicates(
        subset=["dataset", "embedder", "metric_name"],
        keep="first",
    )
    .index
)

best_df = (
    long_df.loc[best_idx, ["dataset", "embedder", "metric_name", "metric_value", "model"]]
    .sort_values(["dataset", "embedder", "metric_name"])
    .reset_index(drop=True)
)

best_df.to_csv(OUTPUT_DIR / "best_metric_by_dataset_embedder.csv", index=False)

# =============================================================================

# Save wide embedder tables split by metric_name

# =============================================================================

OUTPUT_DIR = Path("processed_outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

# `best_df` is the input here:

# dataset, embedder, metric_name, metric_value, model

#

# The nuisance model dimension has already been collapsed:

# for each dataset x embedder x metric_name, metric_value is the maximum

# across models, and model records which model produced that maximum.

wide_tables_by_metric = {}

for metric_name, metric_df in best_df.groupby("metric_name", sort=True):
    wide_df = metric_df.pivot_table(
        index="dataset",
        columns="embedder",
        values="metric_value",
        aggfunc="max",
    ).sort_index()

    wide_df.columns.name = None

    wide_tables_by_metric[metric_name] = wide_df

    safe_metric_name = (
        str(metric_name).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    )

    wide_df.to_csv(
        OUTPUT_DIR / f"embedder_metrics_wide_{safe_metric_name}.csv",
        index=True,
    )

print("Saved wide embedder tables:")

for metric_name, wide_df in wide_tables_by_metric.items():
    print(f"- {metric_name}: {wide_df.shape[0]} datasets x {wide_df.shape[1]} embedders")

# %%
