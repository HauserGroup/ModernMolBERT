import json
from pathlib import Path
import re
from typing import Any, Literal

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
    """Return known metric columns present with at least one finite value."""

    out: list[str] = []
    for metric in DEFAULT_METRIC_DIRECTIONS:
        if metric not in results.columns:
            continue
        values = pd.to_numeric(results[metric], errors="coerce")
        if values.notna().any():
            out.append(metric)
    return out


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


def primary_metric_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize each task by its default primary metric."""

    with_primary = add_primary_metric_column(results)
    group_cols = [
        "dataset",
        "task",
        "task_type",
        "primary_metric",
        "featurizer",
        "downstream_name",
        "downstream_model",
    ]
    group_cols = [col for col in group_cols if col in with_primary.columns]

    return (
        with_primary.groupby(group_cols, dropna=False)["primary_metric_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "primary_metric_mean",
                "std": "primary_metric_std",
                "count": "n_runs",
            }
        )
        .sort_values(group_cols)
    )


def best_by_dataset_task(results: pd.DataFrame) -> pd.DataFrame:
    """Select the best candidate per dataset/task using the task primary metric."""

    summary = primary_metric_summary(results)
    if summary.empty:
        return summary

    rows: list[pd.Series] = []
    for _, group in summary.groupby(["dataset", "task"], dropna=False):
        metric = str(group["primary_metric"].iloc[0])
        direction = DEFAULT_METRIC_DIRECTIONS.get(metric, "higher")
        values = pd.to_numeric(group["primary_metric_mean"], errors="coerce")
        valid = group[values.notna()]
        if valid.empty:
            continue
        best_pos = (
            valid["primary_metric_mean"].argmin()
            if direction == "lower"
            else valid["primary_metric_mean"].argmax()
        )
        rows.append(valid.iloc[int(best_pos)])

    if not rows:
        return summary.iloc[0:0].copy()

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out.sort_values(["dataset", "task"]).reset_index(drop=True)


def skipped_task_summary(skipped: pd.DataFrame) -> pd.DataFrame:
    """Summarize skipped benchmark tasks by reason and available run dimensions."""

    if skipped.empty:
        return pd.DataFrame(columns=["reason", "n_skipped"])

    group_cols = [
        col
        for col in ["reason", "dataset", "task", "featurizer", "downstream_name", "seed"]
        if col in skipped.columns
    ]
    return (
        skipped.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_skipped")
        .sort_values(group_cols)
    )


def manifest_runs_table(manifest: dict[str, Any]) -> pd.DataFrame:
    """Flatten the suite manifest run records into a table."""

    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return pd.DataFrame()

    table = pd.json_normalize(runs, sep=".")
    for column in table.columns:
        if table[column].map(lambda value: isinstance(value, (dict, list))).any():
            table[column] = table[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                )
            )
    return table


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


def write_deliverable_tables(
    *,
    results: pd.DataFrame,
    output_dir: str | Path,
    skipped: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the static table package for benchmark deliverables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    summary_path = output_dir / "summary.csv"
    summarize_results(results).to_csv(summary_path, index=False)
    outputs["summary"] = summary_path

    primary_path = output_dir / "primary_metric_summary.csv"
    primary_metric_summary(results).to_csv(primary_path, index=False)
    outputs["primary_metric_summary"] = primary_path

    best_path = output_dir / "best_by_dataset_task.csv"
    best_by_dataset_task(results).to_csv(best_path, index=False)
    outputs["best_by_dataset_task"] = best_path

    for metric in available_metric_columns(results):
        matrix_path = output_dir / f"metric_matrix_{_safe_slug(metric)}.csv"
        metric_matrix(results, metric=metric).to_csv(matrix_path)
        outputs[f"metric_matrix_{metric}"] = matrix_path

        rank_path = output_dir / f"average_rank_{_safe_slug(metric)}.csv"
        average_rank(results, metric=metric).to_csv(rank_path, index=False)
        outputs[f"average_rank_{metric}"] = rank_path

    if skipped is not None:
        skipped_path = output_dir / "skipped_tasks_summary.csv"
        skipped_task_summary(skipped).to_csv(skipped_path, index=False)
        outputs["skipped_tasks_summary"] = skipped_path

    if manifest is not None:
        manifest_path = output_dir / "manifest_runs.csv"
        manifest_runs_table(manifest).to_csv(manifest_path, index=False)
        outputs["manifest_runs"] = manifest_path

    return outputs


def plot_metric_heatmap(
    results: pd.DataFrame,
    *,
    metric: str,
    output_path: str | Path,
) -> None:
    """Plot a dataset-by-featurizer heatmap for a metric."""

    matrix = metric_matrix(results, metric=metric)
    if matrix.empty:
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 1.3), max(4, matrix.shape[0] * 0.5)))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(
        np.arange(matrix.shape[1]), labels=[str(x) for x in matrix.columns], rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(matrix.shape[0]), labels=[str(x) for x in matrix.index])
    ax.set_title(f"{metric} heatmap")
    ax.set_xlabel("Featurizer")
    ax.set_ylabel("Dataset")
    fig.colorbar(image, ax=ax, label=metric)

    plt.tight_layout()
    _save_figure(output_path)


def plot_average_rank(
    rank: pd.DataFrame,
    *,
    metric: str,
    output_path: str | Path,
    candidate_col: str = "featurizer",
) -> None:
    """Plot average rank values for one metric."""

    if rank.empty:
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(rank) * 1.0), 4))
    ax.bar(rank[candidate_col].astype(str), rank["average_rank"])
    ax.set_xlabel(candidate_col)
    ax.set_ylabel("Average rank")
    ax.set_title(f"Average rank: {metric}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_figure(output_path)


def plot_primary_metric_by_dataset(results: pd.DataFrame, *, output_path: str | Path) -> None:
    """Plot each candidate's mean primary metric by dataset."""

    with_primary = add_primary_metric_column(results)
    plot_data = (
        with_primary.groupby(["dataset", "featurizer"], dropna=False)["primary_metric_value"]
        .mean()
        .reset_index()
    )
    _plot_grouped_bar(
        plot_data,
        index="dataset",
        columns="featurizer",
        values="primary_metric_value",
        output_path=output_path,
        title="Primary metric by dataset",
        ylabel="Primary metric",
    )


def plot_invalid_feature_rate(results: pd.DataFrame, *, output_path: str | Path) -> bool:
    """Plot invalid feature rate by dataset when the metric is available."""

    if "eval_feature_invalid_rate" not in results.columns:
        return False

    plot_data = (
        results.groupby(["dataset", "featurizer"], dropna=False)["eval_feature_invalid_rate"]
        .mean()
        .reset_index()
    )
    _plot_grouped_bar(
        plot_data,
        index="dataset",
        columns="featurizer",
        values="eval_feature_invalid_rate",
        output_path=output_path,
        title="Invalid feature rate by dataset",
        ylabel="Invalid feature rate",
    )
    return True


def find_prediction_artifacts(predictions_dir: str | Path) -> list[Path]:
    """Find prediction .npz files below a predictions directory."""

    predictions_dir = Path(predictions_dir)
    if not predictions_dir.exists():
        return []
    return sorted(predictions_dir.rglob("*.npz"))


def load_prediction_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one prediction artifact and its JSON sidecar when available."""

    path = Path(path)
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    metadata_path = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return arrays, metadata


def write_deliverable_figures(
    *,
    results: pd.DataFrame,
    output_dir: str | Path,
    predictions_dir: str | Path | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """Write aggregate and prediction-level benchmark figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    warnings: list[str] = []

    for metric in available_metric_columns(results):
        heatmap_path = output_dir / f"metric_heatmap_{_safe_slug(metric)}.png"
        plot_metric_heatmap(results, metric=metric, output_path=heatmap_path)
        if heatmap_path.exists():
            outputs[f"metric_heatmap_{metric}"] = heatmap_path

        rank = average_rank(results, metric=metric)
        rank_path = output_dir / f"average_rank_{_safe_slug(metric)}.png"
        plot_average_rank(rank, metric=metric, output_path=rank_path)
        if rank_path.exists():
            outputs[f"average_rank_{metric}"] = rank_path

    primary_path = output_dir / "primary_metric_by_dataset.png"
    plot_primary_metric_by_dataset(results, output_path=primary_path)
    if primary_path.exists():
        outputs["primary_metric_by_dataset"] = primary_path

    invalid_path = output_dir / "invalid_feature_rate_by_dataset.png"
    if plot_invalid_feature_rate(results, output_path=invalid_path):
        outputs["invalid_feature_rate_by_dataset"] = invalid_path

    if predictions_dir is None or not Path(predictions_dir).exists():
        warnings.append(
            "Prediction artifacts not found; skipped ROC/PR/calibration/confusion/"
            "regression/residual figures."
        )
        return outputs, warnings

    prediction_paths = find_prediction_artifacts(predictions_dir)
    if not prediction_paths:
        warnings.append(
            "Prediction artifacts directory is empty; skipped prediction-level figures."
        )
        return outputs, warnings

    prediction_outputs, prediction_warnings = write_prediction_figures(
        prediction_paths=prediction_paths,
        output_dir=output_dir,
    )
    outputs.update(prediction_outputs)
    warnings.extend(prediction_warnings)
    return outputs, warnings


def write_prediction_figures(
    *,
    prediction_paths: list[Path],
    output_dir: str | Path,
) -> tuple[dict[str, Path], list[str]]:
    """Write ROC/PR/calibration/confusion and regression diagnostic figures."""

    output_dir = Path(output_dir)
    grouped: dict[
        tuple[str, str, str], list[tuple[Path, dict[str, np.ndarray], dict[str, Any]]]
    ] = {}
    warnings: list[str] = []

    for path in prediction_paths:
        arrays, metadata = load_prediction_artifact(path)
        dataset = str(
            metadata.get("dataset", path.parts[-5] if len(path.parts) >= 5 else "dataset")
        )
        task = str(metadata.get("task", path.parts[-4] if len(path.parts) >= 4 else "task"))
        task_type = str(metadata.get("task_type") or _infer_prediction_task_type(arrays))
        grouped.setdefault((task_type, dataset, task), []).append((path, arrays, metadata))

    outputs: dict[str, Path] = {}
    for (task_type, dataset, task), entries in grouped.items():
        slug = f"{_safe_slug(dataset)}_{_safe_slug(task)}"
        if task_type == "classification":
            outputs.update(
                _write_classification_prediction_figures(
                    entries=entries,
                    output_dir=output_dir,
                    slug=slug,
                    title=f"{dataset}: {task}",
                    warnings=warnings,
                )
            )
        elif task_type == "regression":
            outputs.update(
                _write_regression_prediction_figures(
                    entries=entries,
                    output_dir=output_dir,
                    slug=slug,
                    title=f"{dataset}: {task}",
                    warnings=warnings,
                )
            )

    return outputs, warnings


def create_benchmark_deliverables(
    *,
    sweep_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create static benchmark deliverables from a full sweep directory."""

    sweep_dir = Path(sweep_dir)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = sweep_dir / "results.csv"
    results = load_results(results_path)

    skipped_path = sweep_dir / "skipped_tasks.csv"
    skipped = pd.read_csv(skipped_path) if skipped_path.exists() else None

    manifest_path = sweep_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )

    prediction_dir = sweep_dir / "predictions"
    table_outputs = write_deliverable_tables(
        results=results,
        output_dir=tables_dir,
        skipped=skipped,
        manifest=manifest,
    )
    figure_outputs, warnings = write_deliverable_figures(
        results=results,
        output_dir=figures_dir,
        predictions_dir=prediction_dir,
    )

    prediction_artifacts = find_prediction_artifacts(prediction_dir)
    deliverable_manifest: dict[str, Any] = {
        "sweep_dir": str(sweep_dir),
        "results_csv": str(results_path),
        "manifest_json": str(manifest_path) if manifest_path.exists() else None,
        "skipped_tasks_csv": str(skipped_path) if skipped_path.exists() else None,
        "predictions_dir": str(prediction_dir) if prediction_dir.exists() else None,
        "n_result_rows": int(len(results)),
        "n_prediction_artifacts": int(len(prediction_artifacts)),
        "tables": {key: str(path) for key, path in table_outputs.items()},
        "figures": {key: str(path) for key, path in figure_outputs.items()},
        "warnings": warnings,
    }

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _deliverables_readme(
            table_outputs=table_outputs,
            figure_outputs=figure_outputs,
            warnings=warnings,
        ),
        encoding="utf-8",
    )

    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(
        json.dumps(deliverable_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return deliverable_manifest


def _write_classification_prediction_figures(
    *,
    entries: list[tuple[Path, dict[str, np.ndarray], dict[str, Any]]],
    output_dir: Path,
    slug: str,
    title: str,
    warnings: list[str],
) -> dict[str, Path]:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix
    from sklearn.metrics import precision_recall_curve, roc_curve

    import matplotlib.pyplot as plt

    outputs: dict[str, Path] = {}
    score_entries = [
        (path, arrays, metadata)
        for path, arrays, metadata in entries
        if "y_score" in arrays and len(np.unique(arrays["y_true"])) == 2
    ]

    if score_entries:
        roc_path = output_dir / f"roc_{slug}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        for _, arrays, metadata in score_entries:
            y_true = arrays["y_true"].astype(int)
            y_score = arrays["y_score"].astype(float)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{_prediction_label(metadata)} AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC: {title}")
        ax.legend(fontsize="small")
        plt.tight_layout()
        _save_figure(roc_path)
        outputs[f"roc_{slug}"] = roc_path

        pr_path = output_dir / f"pr_{slug}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        for _, arrays, metadata in score_entries:
            y_true = arrays["y_true"].astype(int)
            y_score = arrays["y_score"].astype(float)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax.plot(recall, precision, label=_prediction_label(metadata))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-recall: {title}")
        ax.legend(fontsize="small")
        plt.tight_layout()
        _save_figure(pr_path)
        outputs[f"pr_{slug}"] = pr_path

        calibration_path = output_dir / f"calibration_{slug}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        for _, arrays, metadata in score_entries:
            y_true = arrays["y_true"].astype(int)
            y_score = arrays["y_score"].astype(float)
            n_bins = min(10, max(2, len(y_true) // 2))
            prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="o", label=_prediction_label(metadata))
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"Calibration: {title}")
        ax.legend(fontsize="small")
        plt.tight_layout()
        _save_figure(calibration_path)
        outputs[f"calibration_{slug}"] = calibration_path
    else:
        warnings.append(f"No usable y_score arrays for classification figures: {title}")

    confusion_path = output_dir / f"confusion_{slug}.png"
    y_true_all = np.concatenate([arrays["y_true"].astype(int) for _, arrays, _ in entries])
    y_pred_all = np.concatenate([arrays["y_pred"].astype(int) for _, arrays, _ in entries])
    matrix = confusion_matrix(y_true_all, y_pred_all)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(values_format="d")
    plt.title(f"Confusion matrix: {title}")
    plt.tight_layout()
    _save_figure(confusion_path)
    outputs[f"confusion_{slug}"] = confusion_path

    return outputs


def _write_regression_prediction_figures(
    *,
    entries: list[tuple[Path, dict[str, np.ndarray], dict[str, Any]]],
    output_dir: Path,
    slug: str,
    title: str,
    warnings: list[str],
) -> dict[str, Path]:
    del warnings
    import matplotlib.pyplot as plt

    outputs: dict[str, Path] = {}

    regression_path = output_dir / f"regression_{slug}.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    mins: list[float] = []
    maxs: list[float] = []
    for _, arrays, metadata in entries:
        y_true = arrays["y_true"].astype(float)
        y_pred = arrays["y_pred"].astype(float)
        ax.scatter(y_true, y_pred, alpha=0.75, label=_prediction_label(metadata))
        mins.extend([float(np.min(y_true)), float(np.min(y_pred))])
        maxs.extend([float(np.max(y_true)), float(np.max(y_pred))])
    if mins and maxs:
        lo = min(mins)
        hi = max(maxs)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.5")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs true: {title}")
    ax.legend(fontsize="small")
    plt.tight_layout()
    _save_figure(regression_path)
    outputs[f"regression_{slug}"] = regression_path

    residual_path = output_dir / f"residuals_{slug}.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for _, arrays, metadata in entries:
        y_true = arrays["y_true"].astype(float)
        y_pred = arrays["y_pred"].astype(float)
        residual = y_pred - y_true
        axes[0].scatter(y_pred, residual, alpha=0.75, label=_prediction_label(metadata))
        axes[1].hist(residual, bins=min(20, max(5, len(residual))), alpha=0.45)
    axes[0].axhline(0, linestyle="--", color="0.5")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[0].legend(fontsize="small")
    fig.suptitle(f"Residuals: {title}")
    plt.tight_layout()
    _save_figure(residual_path)
    outputs[f"residuals_{slug}"] = residual_path

    return outputs


def _plot_grouped_bar(
    data: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    if data.empty:
        return

    pivot = data.pivot(index=index, columns=columns, values=values)
    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 1.2), 5))
    ax.set_xlabel(index)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=columns, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    _save_figure(output_path)


def _save_figure(output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _infer_prediction_task_type(arrays: dict[str, np.ndarray]) -> str:
    if "y_score" in arrays:
        return "classification"
    y_true = arrays.get("y_true")
    if y_true is not None and np.all(np.isin(y_true, [0, 1])):
        return "classification"
    return "regression"


def _prediction_label(metadata: dict[str, Any]) -> str:
    pieces = [
        str(metadata.get("featurizer", "")),
        str(metadata.get("downstream_name", metadata.get("downstream_model", ""))),
        f"seed={metadata.get('seed')}" if metadata.get("seed") is not None else "",
    ]
    label = " / ".join(piece for piece in pieces if piece)
    return label or "prediction"


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return slug or "value"


def _deliverables_readme(
    *,
    table_outputs: dict[str, Path],
    figure_outputs: dict[str, Path],
    warnings: list[str],
) -> str:
    lines = [
        "# Benchmark Deliverables",
        "",
        "Static tables and figures generated from a benchmark sweep directory.",
        "",
        "## Tables",
        "",
    ]
    lines.extend(f"- `{path.relative_to(path.parents[1])}`" for path in table_outputs.values())
    lines.extend(["", "## Figures", ""])
    if figure_outputs:
        lines.extend(f"- `{path.relative_to(path.parents[1])}`" for path in figure_outputs.values())
    else:
        lines.append("- None generated.")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)
