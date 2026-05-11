from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


MetricDirection = Literal["higher", "lower"]


DEFAULT_PRIMARY_METRICS: dict[str, str] = {
    "classification": "roc_auc",
    "regression": "rmse",
}


DEFAULT_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "roc_auc": "higher",
    "average_precision": "higher",
    "mcc": "higher",
    "f1": "higher",
    "rmse": "lower",
    "mae": "lower",
    "r2": "higher",
}


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a benchmark results CSV."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    frame = pd.read_csv(path)

    required = {
        "dataset",
        "task",
        "task_type",
        "featurizer",
        "downstream_name",
        "seed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Results file is missing required columns: {missing}")

    return frame


def available_metric_columns(results: pd.DataFrame) -> list[str]:
    """Return known metric columns present in a results frame."""

    return [metric for metric in DEFAULT_METRIC_DIRECTIONS if metric in results.columns]


def primary_metric_for_task_type(task_type: str) -> str:
    """Return the default primary metric for a task type."""

    if task_type not in DEFAULT_PRIMARY_METRICS:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    return DEFAULT_PRIMARY_METRICS[task_type]


def summarize_results(
    results: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize benchmark results by dataset/task/featurizer/downstream model.

    The output contains mean/std/count across seeds for each metric.
    """

    if metrics is None:
        metrics = available_metric_columns(results)

    if not metrics:
        raise ValueError("No known metric columns found in results.")

    group_cols = [
        "dataset",
        "task",
        "task_type",
        "featurizer",
        "downstream_name",
        "downstream_model",
    ]
    group_cols = [col for col in group_cols if col in results.columns]

    summary = (
        results.groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]

    return summary


def add_primary_metric_column(results: pd.DataFrame) -> pd.DataFrame:
    """Add primary_metric and primary_metric_value columns per row."""

    out = results.copy()

    primary_metrics: list[str] = []
    values: list[float] = []

    for _, row in out.iterrows():
        metric = primary_metric_for_task_type(str(row["task_type"]))
        primary_metrics.append(metric)
        values.append(
            float(row[metric]) if metric in out.columns and pd.notna(row[metric]) else np.nan
        )

    out["primary_metric"] = primary_metrics
    out["primary_metric_value"] = values

    return out


def metric_matrix(
    results: pd.DataFrame,
    *,
    metric: str,
    index: str = "dataset",
    columns: str = "featurizer",
    values: str | None = None,
) -> pd.DataFrame:
    """Create a pivot table of mean metric values.

    By default, rows are datasets and columns are featurizers.
    """

    if metric not in results.columns:
        raise ValueError(f"Metric {metric!r} not found in results.")

    if values is not None:
        metric = values

    return results.pivot_table(
        index=index,
        columns=columns,
        values=metric,
        aggfunc="mean",
    )


def average_rank(
    results: pd.DataFrame,
    *,
    metric: str,
    group_cols: list[str] | None = None,
    candidate_col: str = "featurizer",
    direction: MetricDirection | None = None,
) -> pd.DataFrame:
    """Compute average rank of candidates across tasks/datasets.

    Lower rank is better. For metrics like RMSE, lower values rank better.
    """

    if metric not in results.columns:
        raise ValueError(f"Metric {metric!r} not found in results.")

    if direction is None:
        direction = DEFAULT_METRIC_DIRECTIONS.get(metric)
        if direction is None:
            raise ValueError(
                f"No default direction for metric {metric!r}; pass direction explicitly."
            )

    if group_cols is None:
        group_cols = ["dataset", "task", "downstream_name"]

    ascending = direction == "lower"

    grouped = (
        results.groupby(group_cols + [candidate_col], dropna=False)[metric].mean().reset_index()
    )

    grouped["rank"] = grouped.groupby(group_cols, dropna=False)[metric].rank(
        method="average",
        ascending=ascending,
    )

    return (
        grouped.groupby(candidate_col, dropna=False)["rank"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "average_rank",
                "std": "rank_std",
                "count": "n_comparisons",
            }
        )
        .sort_values("average_rank")
    )


def write_summary_tables(
    *,
    results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write standard summary tables for a benchmark results CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    summary = summarize_results(results)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    out: dict[str, Path] = {"summary": summary_path}

    for metric in available_metric_columns(results):
        matrix = metric_matrix(results, metric=metric)
        matrix_path = output_dir / f"{metric}_matrix.csv"
        matrix.to_csv(matrix_path)
        out[f"{metric}_matrix"] = matrix_path

        rank = average_rank(results, metric=metric)
        rank_path = output_dir / f"{metric}_average_rank.csv"
        rank.to_csv(rank_path, index=False)
        out[f"{metric}_average_rank"] = rank_path

    return out


def plot_metric_by_dataset(
    results: pd.DataFrame,
    *,
    metric: str,
    output_path: str | Path,
    hue: str = "featurizer",
) -> None:
    """Create a bar plot of one metric by dataset and candidate."""

    if metric not in results.columns:
        raise ValueError(f"Metric {metric!r} not found in results.")

    import matplotlib.pyplot as plt

    plot_data = results.groupby(["dataset", hue], dropna=False)[metric].mean().reset_index()

    pivot = plot_data.pivot(index="dataset", columns=hue, values=metric)

    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 1.2), 5))
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by dataset")
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pairwise_metric_scatter(
    results: pd.DataFrame,
    *,
    metric: str,
    candidate_a: str,
    candidate_b: str,
    output_path: str | Path,
    candidate_col: str = "featurizer",
) -> None:
    """Scatter plot comparing two candidates across dataset/task rows."""

    if metric not in results.columns:
        raise ValueError(f"Metric {metric!r} not found in results.")

    import matplotlib.pyplot as plt

    grouped = (
        results.groupby(["dataset", "task", candidate_col], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["dataset", "task"],
        columns=candidate_col,
        values=metric,
        aggfunc="mean",
    )

    if candidate_a not in pivot.columns:
        raise ValueError(f"Candidate {candidate_a!r} not found in {candidate_col}.")
    if candidate_b not in pivot.columns:
        raise ValueError(f"Candidate {candidate_b!r} not found in {candidate_col}.")

    x = pivot[candidate_a]
    y = pivot[candidate_b]

    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y)

    if len(x) > 0:
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(candidate_a)
    ax.set_ylabel(candidate_b)
    ax.set_title(f"{metric}: {candidate_b} vs {candidate_a}")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_standard_plots(
    *,
    results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a small standard plot set for available metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    outputs: dict[str, Path] = {}

    for metric in available_metric_columns(results):
        path = output_dir / f"{metric}_by_dataset.png"
        plot_metric_by_dataset(
            results,
            metric=metric,
            output_path=path,
        )
        outputs[f"{metric}_by_dataset"] = path

    return outputs
