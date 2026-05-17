"""DABEST effect-size plots for benchmark comparisons.

Requires: dabest (not a hard dependency — imported lazily).
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import dabest  # type: ignore[import]
    import matplotlib.figure


def _load_data(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


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


def dabest_model_comparison(
    data: pd.DataFrame | str | Path,
    *,
    control_model: str,
    comparison_models: list[str],
    metric_col: str = "test_metric",
    metric_name_col: str = "test_metric_name",
    metric_name: str | None = None,
    group_col: str = "model",
    pair_on: list[str] | None = None,
    embedders: list[str] | None = None,
    embedder_col: str = "embedder",
    output_path: str | Path | None = None,
    **plot_kwargs,
) -> tuple[dabest.Dabest, matplotlib.figure.Figure]:
    """DABEST shared-control paired plot comparing downstream models.

    Each (dataset, embedder) combination is one paired observation. The plot
    shows mean difference ± confidence interval relative to the control model.

    Parameters
    ----------
    data:
        DataFrame or path to CSV in arxiv-preprint format
        (columns: dataset, embedder, model, test_metric, ...).
    control_model:
        The baseline model name (first in the DABEST idx tuple).
    comparison_models:
        One or more model names to compare against the control.
    metric_col:
        Column holding the numeric metric value.
    metric_name_col:
        Column holding the metric name string (used for filtering).
    metric_name:
        If given, filter rows where ``metric_name_col == metric_name``.
    group_col:
        Column that identifies the model/group being compared (default "model").
    pair_on:
        Columns whose combination identifies a paired observation.
        Defaults to ["dataset", embedder_col].
    embedders:
        If given, restrict to these embedder values before pairing.
    embedder_col:
        Column holding embedder names (default "embedder").
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

    if pair_on is None:
        pair_on = ["dataset", embedder_col]

    df = _load_data(data)
    _require_columns(df, [group_col, metric_col, *pair_on])

    all_models = [control_model, *comparison_models]

    if metric_name is not None:
        _require_columns(df, [metric_name_col])
        df = df[df[metric_name_col] == metric_name]
        if df.empty:
            raise ValueError(f"No rows with {metric_name_col}={metric_name!r}")

    if embedders is not None:
        df = df[df[embedder_col].isin(embedders)]
        if df.empty:
            raise ValueError(f"No rows after filtering embedders to {embedders}")

    df = df[df[group_col].isin(all_models)].copy()
    if df.empty:
        raise ValueError(f"No rows matching models {all_models}")

    pair_id_col = "__pair_id__"
    df[pair_id_col] = df[pair_on].astype(str).agg("__".join, axis=1)

    df = _drop_incomplete_pairs(df, pair_id_col, group_col, all_models)
    if df.empty:
        raise ValueError("No complete pairs remaining after filtering.")

    analysis = _dabest.load(
        data=df,
        x=group_col,
        y=metric_col,
        idx=tuple(all_models),
        paired="baseline",
        id_col=pair_id_col,
    )

    fig = analysis.mean_diff.plot(**plot_kwargs)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return analysis, fig
