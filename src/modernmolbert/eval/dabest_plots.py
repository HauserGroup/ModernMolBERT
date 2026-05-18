"""DABEST effect-size plots for benchmark comparisons.

Requires: dabest (not a hard dependency — imported lazily).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    PRASKI_COLUMNS,
    read_results_csv,
    to_praski_schema,
)

if TYPE_CHECKING:
    import dabest  # type: ignore[import] # noqa
    import matplotlib.figure


def _load_praski_data(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return to_praski_schema(data)
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_results_csv(path)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")


def _drop_incomplete_pairs(
    df: pd.DataFrame,
    pair_col: str,
    group_col: str,
    required_groups: list[str],
) -> pd.DataFrame:
    """Remove pair IDs that don't have all required groups."""
    counts = df[df[group_col].isin(required_groups)].groupby(pair_col)[group_col].nunique()
    complete = counts[counts == len(required_groups)].index
    n_dropped = df[pair_col].nunique() - len(complete)
    if n_dropped:
        import warnings

        warnings.warn(
            f"Dropped {n_dropped} incomplete pair(s) (missing one or more models).",
            stacklevel=3,
        )
    return df[df[pair_col].isin(complete)]


def _filter_values(
    df: pd.DataFrame,
    *,
    column: str,
    values: list[str] | None,
    label: str,
) -> pd.DataFrame:
    if values is None:
        return df
    _require_columns(df, [column])
    out = df[df[column].isin(values)].copy()
    if out.empty:
        raise ValueError(f"No rows after filtering {label} to {values}")
    return out


def _drop_duplicate_pair_groups(
    df: pd.DataFrame,
    *,
    pair_id_col: str,
    group_col: str,
) -> pd.DataFrame:
    duplicate_mask = df.duplicated(subset=[pair_id_col, group_col], keep=False)
    if not duplicate_mask.any():
        return df

    import warnings

    warnings.warn(
        "Duplicate rows found for one or more paired observations; keeping the last row by id.",
        stacklevel=3,
    )
    return df.sort_values("id").drop_duplicates(subset=[pair_id_col, group_col], keep="last")


def _dabest_paired_arg(dabest_module, groups: list[str]) -> bool | str:
    version = getattr(dabest_module, "__version__", None)
    if isinstance(version, str) and version.startswith("0."):
        if len(groups) != 2:
            raise ValueError(
                f"dabest {version} only supports paired plots with one comparison. "
                "Pass a single comparison or install a newer dabest release."
            )
        return True
    return "baseline"


def _dabest_load_compat(dabest_module, **kwargs):
    """Call dabest.load with a narrow compatibility patch for dabest 0.2.x."""
    version = getattr(dabest_module, "__version__", None)
    if not (isinstance(version, str) and version.startswith("0.")):
        return dabest_module.load(**kwargs)

    original_unique = pd.unique

    def unique_compat(values):
        if isinstance(values, list):
            return original_unique(pd.Series(values, dtype="object"))
        return original_unique(values)

    pd.unique = unique_compat
    try:
        return dabest_module.load(**kwargs)
    finally:
        pd.unique = original_unique


def dabest_paired_comparison(
    data: pd.DataFrame | str | Path,
    *,
    control: str,
    comparisons: list[str],
    group_col: str,
    pair_on: list[str],
    metric_col: str = "test_metric",
    metric_name_col: str = "test_metric_name",
    metric_name: str | None = None,
    embedders: list[str] | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    tasks: list[str] | None = None,
    output_path: str | Path | None = None,
    **plot_kwargs,
) -> tuple[Any, matplotlib.figure.Figure]:
    """DABEST shared-control paired plot from Praski-schema benchmark results.

    ``data`` may be a path to the benchmark CSV or a DataFrame. Inputs are
    normalized through the Praski schema before filtering and pairing.

    Parameters
    ----------
    data:
        DataFrame or path to CSV with columns compatible with ``PRASKI_COLUMNS``.
    control:
        The baseline model name (first in the DABEST idx tuple).
    comparisons:
        One or more group names to compare against the control.
    group_col:
        Column that identifies the group being compared, usually ``model`` or ``embedder``.
    pair_on:
        Columns whose combination identifies a paired observation.
    metric_col:
        Column holding the numeric metric value.
    metric_name_col:
        Column holding the metric name string (used for filtering).
    metric_name:
        If given, filter rows where ``metric_name_col == metric_name``.
    embedders:
        If given, restrict to these embedder values.
    models:
        If given, restrict to these supervised head values.
    datasets:
        If given, restrict to these dataset values.
    tasks:
        If given, restrict to these task values.
    output_path:
        If given, save the figure to this path.
    **plot_kwargs:
        Passed to ``analysis.mean_diff.plot()``.

    Returns
    -------
    (dabest_object, figure)
    """
    try:
        import dabest as _dabest  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "dabest is required for this function. Install it with: pip install dabest"
        ) from exc

    df = _load_praski_data(data)
    _require_columns(df, PRASKI_COLUMNS)
    _require_columns(df, [group_col, metric_col, *pair_on])

    all_groups = [control, *comparisons]

    if metric_name is not None:
        _require_columns(df, [metric_name_col])
        df = df[df[metric_name_col] == metric_name]
        if df.empty:
            raise ValueError(f"No rows with {metric_name_col}={metric_name!r}")

    df = _filter_values(df, column="embedder", values=embedders, label="embedders")
    df = _filter_values(df, column="model", values=models, label="models")
    df = _filter_values(df, column="dataset", values=datasets, label="datasets")
    df = _filter_values(df, column="task", values=tasks, label="tasks")

    df = df[df[group_col].isin(all_groups)].copy()
    if df.empty:
        raise ValueError(f"No rows matching groups {all_groups} in column {group_col!r}")

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[group_col, metric_col, *pair_on])
    if df.empty:
        raise ValueError("No rows with complete pairing columns and numeric metric values.")

    pair_id_col = "__pair_id__"
    pair_labels = df[pair_on].astype(str).agg("__".join, axis=1)
    df[pair_id_col] = pd.factorize(pair_labels, sort=False)[0]
    df = _drop_duplicate_pair_groups(df, pair_id_col=pair_id_col, group_col=group_col)

    df = _drop_incomplete_pairs(df, pair_id_col, group_col, all_groups)
    if df.empty:
        raise ValueError("No complete pairs remaining after filtering.")

    dabest_df = df.loc[:, [group_col, metric_col, pair_id_col]].copy()
    analysis = _dabest_load_compat(
        _dabest,
        data=dabest_df,
        x=group_col,
        y=metric_col,
        idx=tuple(all_groups),
        paired=_dabest_paired_arg(_dabest, all_groups),
        id_col=pair_id_col,
    )

    fig = analysis.mean_diff.plot(**plot_kwargs)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return analysis, fig


def dabest_model_comparison(
    data: pd.DataFrame | str | Path,
    *,
    control_model: str,
    comparison_models: list[str],
    metric_col: str = "test_metric",
    metric_name_col: str = "test_metric_name",
    metric_name: str | None = None,
    pair_on: list[str] | None = None,
    embedders: list[str] | None = None,
    output_path: str | Path | None = None,
    **plot_kwargs,
) -> tuple[Any, matplotlib.figure.Figure]:
    """Compare supervised heads using paired (dataset, embedder) observations."""
    return dabest_paired_comparison(
        data,
        control=control_model,
        comparisons=comparison_models,
        group_col="model",
        pair_on=pair_on or ["dataset", "embedder"],
        metric_col=metric_col,
        metric_name_col=metric_name_col,
        metric_name=metric_name,
        embedders=embedders,
        output_path=output_path,
        **plot_kwargs,
    )


def dabest_embedder_comparison(
    data: pd.DataFrame | str | Path,
    *,
    control_embedder: str,
    comparison_embedders: list[str],
    metric_col: str = "test_metric",
    metric_name_col: str = "test_metric_name",
    metric_name: str | None = None,
    pair_on: list[str] | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    tasks: list[str] | None = None,
    output_path: str | Path | None = None,
    **plot_kwargs,
) -> tuple[Any, matplotlib.figure.Figure]:
    """Compare embedders using paired (dataset, model/head) observations."""
    return dabest_paired_comparison(
        data,
        control=control_embedder,
        comparisons=comparison_embedders,
        group_col="embedder",
        pair_on=pair_on or ["dataset", "model"],
        metric_col=metric_col,
        metric_name_col=metric_name_col,
        metric_name=metric_name,
        embedders=[control_embedder, *comparison_embedders],
        models=models,
        datasets=datasets,
        tasks=tasks,
        output_path=output_path,
        **plot_kwargs,
    )
