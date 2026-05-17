"""
# One CSV

uv run python src/modernmolbert/eval/benchmarking_molecular_models/compare_praski_tables.py \
  --baseline outputs/benchmark/all_results.csv \
  --output-dir outputs/benchmark/comparison \
  --our-embedder ModernMolBERT_SELFIES_ChEMBL36_2M

# Multiple CSVs
uv run python src/modernmolbert/eval/benchmarking_molecular_models/compare_praski_tables.py \
  --baseline data/benchmarks/arxiv_preprint_2025_08.csv \
  --ours outputs/molecular_eval/final/results_praski.csv \
  --output-dir outputs/molecular_eval/final/comparison \
  --our-embedder ModernMolBERT_SELFIES_ChEMBL36_2M

File | Meaning
table6_like.csv | Closest to Table 6: mean rank and mean metric for best head, kNN, RF, and linear/ridge.
table1_like.csv | Compact model summary: mean rank and mean metric after best-head selection.
dataset_winners.csv | Best embedder/head per dataset.
pairwise_vs_ours.csv | Win/loss table comparing our embedder to every other embedder on shared datasets.

To add model-family annotations, join on a model_lookup.csv via the Model/competitor column.

"""

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "task",
    "embedder",
    "model",
    "cv_metric_name",
    "cv_metric",
    "test_metric_name",
    "test_metric",
}


HEAD_NAME_MAP = {
    "ridge": "linear",
    "linear": "linear",
    "logistic": "linear",
    "rf": "rf",
    "random_forest": "rf",
    "knn": "knn",
}


HIGHER_IS_BETTER = {
    "roc_auc": True,
    "auroc": True,
    "average_precision": True,
    "ap": True,
    "rmse": False,
    "mae": False,
    "mse": False,
}

SCALE_TO_PERCENT: frozenset[str] = frozenset({"roc_auc", "auroc", "average_precision", "ap"})

SUMMARY_TABLE_SENTINEL_COLS: frozenset[str] = frozenset(
    {"Mean_rank", "Mean_AUROC", "rank_best", "metric_best", "Mean_rank_↓", "Mean_AUROC_↑"}
)


def _metric_scale(metric: str) -> float:
    """Display scale factor: 100 for bounded probability metrics, 1 otherwise."""
    return 100.0 if metric.strip().lower() in SCALE_TO_PERCENT else 1.0


def _add_display_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add display_metric = test_metric * per-row scale based on metric type."""
    df = df.copy()
    df["display_metric"] = [
        row["test_metric"] * _metric_scale(str(row["metric"])) for _, row in df.iterrows()
    ]
    return df


def load_praski_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)

    sentinel_found = SUMMARY_TABLE_SENTINEL_COLS & set(df.columns)
    if sentinel_found:
        raise ValueError(
            f"{path.name} looks like a pre-computed summary table "
            f"(has columns: {sorted(sentinel_found)}). "
            "load_praski_csv expects raw per-dataset benchmark results. "
            "Bundled Praski TSVs (Praski_table_1.tsv etc.) are reference artifacts, "
            f"not input to this script. Required columns: {sorted(REQUIRED_COLUMNS)}."
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["test_metric"] = pd.to_numeric(df["test_metric"], errors="coerce")
    df["cv_metric"] = pd.to_numeric(df["cv_metric"], errors="coerce")
    df["head"] = df["model"].astype(str).map(normalize_head_name)
    df["metric"] = df["test_metric_name"].astype(str)

    return df.dropna(subset=["test_metric"])


def normalize_head_name(name: str) -> str:
    key = name.strip().lower()
    return HEAD_NAME_MAP.get(key, key)


def metric_higher_is_better(metric: str) -> bool:
    metric = metric.strip().lower()
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"Unknown metric direction for {metric!r}")
    return HIGHER_IS_BETTER[metric]


def rank_within_dataset(
    df: pd.DataFrame,
    *,
    score_col: str = "test_metric",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["dataset", "metric"]

    ranked_parts = []

    for (_, metric), group in df.groupby(["dataset", "metric"], dropna=False):
        higher = metric_higher_is_better(str(metric))
        group = group.copy()
        group["rank"] = group[score_col].rank(
            ascending=not higher,
            method="average",
        )
        ranked_parts.append(group)

    return pd.concat(ranked_parts, ignore_index=True)


def select_best_head_per_dataset_embedder(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset × embedder, selecting the best downstream head by cv_metric.

    Head is chosen by cross-validation score (cv_metric) to keep model selection fair.
    The test_metric of the selected row is then reported.
    """
    parts = []

    for (_, metric), group in df.groupby(["dataset", "metric"], dropna=False):
        higher = metric_higher_is_better(str(metric))
        group = group.sort_values(
            "cv_metric",
            ascending=not higher,
        )
        best = group.groupby(["dataset", "embedder"], as_index=False).first()
        parts.append(best)

    return pd.concat(parts, ignore_index=True)


def best_head_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for select_best_head_per_dataset_embedder for external callers."""
    return select_best_head_per_dataset_embedder(df)


def summarize_head_specific(df: pd.DataFrame, *, head: str) -> pd.DataFrame:
    """Mean rank and mean metric for a fixed head: knn, rf, or linear."""
    sub = df[df["head"] == head].copy()
    if sub.empty:
        return pd.DataFrame(columns=["embedder", f"rank_{head}", f"score_{head}"])

    ranked = _add_display_metric(rank_within_dataset(sub))

    out = ranked.groupby("embedder", as_index=False).agg(
        **{
            f"rank_{head}": ("rank", "mean"),
            f"score_{head}": ("display_metric", "mean"),
            f"n_{head}": ("dataset", "nunique"),
        }
    )
    return out


def make_table6_like(df: pd.DataFrame) -> pd.DataFrame:
    """Praski Table 6-like: per embedder ranks/scores for best, knn, rf, and linear."""
    best = select_best_head_per_dataset_embedder(df)
    best_ranked = _add_display_metric(rank_within_dataset(best))

    table = best_ranked.groupby("embedder", as_index=False).agg(
        rank_best=("rank", "mean"),
        score_best=("display_metric", "mean"),
        n_best=("dataset", "nunique"),
    )

    for head in ["knn", "rf", "linear"]:
        head_summary = summarize_head_specific(df, head=head)
        table = table.merge(head_summary, on="embedder", how="left")

    table = table.rename(
        columns={
            "embedder": "Model",
            "score_best": "metric_best",
            "score_knn": "metric_knn",
            "score_rf": "metric_rf",
            "score_linear": "metric_linear",
        }
    )

    ordered = [
        "Model",
        "rank_best",
        "rank_knn",
        "rank_rf",
        "rank_linear",
        "metric_best",
        "metric_knn",
        "metric_rf",
        "metric_linear",
        "n_best",
        "n_knn",
        "n_rf",
        "n_linear",
    ]
    table = table[[col for col in ordered if col in table.columns]]

    return table.sort_values("rank_best", ascending=True)


def collapse_model_name(embedder: str) -> str:
    """Optional compact naming for Table 1-like summary.

    Keeps this conservative: strip bracketed training-size/objective suffixes.
    Examples:
      ChemBERTa_[10M][MTR] -> ChemBERTa
      ECFP_[Count] -> ECFP
      R-MAT_[4M] -> R-MAT
    """
    name = str(embedder)
    name = re.sub(r"_?\[[^\]]+\]", "", name)
    return name


def make_table1_like(df: pd.DataFrame, *, collapse_names: bool = True) -> pd.DataFrame:
    """Compact Table 1-like summary.

    Uses best downstream head per dataset/embedder first.
    Then optionally collapses model variants by taking the best variant per dataset.
    """
    best = select_best_head_per_dataset_embedder(df)

    if collapse_names:
        best = best.copy()
        best["Model"] = best["embedder"].map(collapse_model_name)

        collapsed_parts = []
        for (_, metric), group in best.groupby(["dataset", "metric"], dropna=False):
            higher = metric_higher_is_better(str(metric))
            group = group.sort_values("test_metric", ascending=not higher)
            collapsed_parts.append(group.groupby(["dataset", "Model"], as_index=False).first())

        best = pd.concat(collapsed_parts, ignore_index=True)
    else:
        best = best.rename(columns={"embedder": "Model"})

    ranked = _add_display_metric(rank_within_dataset(best))

    out = ranked.groupby("Model", as_index=False).agg(
        Mean_rank=("rank", "mean"),
        Mean_metric=("display_metric", "mean"),
        N_datasets=("dataset", "nunique"),
    )

    return out.sort_values(["Mean_rank", "Mean_metric"], ascending=[True, False])


def make_dataset_winners(df: pd.DataFrame) -> pd.DataFrame:
    best = select_best_head_per_dataset_embedder(df)

    rows = []
    for (dataset, metric), group in best.groupby(["dataset", "metric"], dropna=False):
        higher = metric_higher_is_better(str(metric))
        group = group.sort_values("test_metric", ascending=not higher)
        winner = group.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "best_embedder": winner["embedder"],
                "best_head": winner["head"],
                "best_test_metric": winner["test_metric"],
                "n_embedders": group["embedder"].nunique(),
            }
        )

    return pd.DataFrame(rows).sort_values("dataset")


def make_pairwise_vs_ours(df: pd.DataFrame, *, ours: str) -> pd.DataFrame:
    best = select_best_head_per_dataset_embedder(df)

    rows = []

    for metric, group in best.groupby("metric", dropna=False):
        higher = metric_higher_is_better(str(metric))
        wide = group.pivot(index="dataset", columns="embedder", values="test_metric")

        if ours not in wide.columns:
            raise ValueError(f"Our embedder {ours!r} not found. Available: {list(wide.columns)}")

        for competitor in wide.columns:
            if competitor == ours:
                continue

            sub = wide[[ours, competitor]].dropna()
            if sub.empty:
                continue

            raw_delta = sub[ours] - sub[competitor]
            win_delta = raw_delta if higher else -raw_delta

            rows.append(
                {
                    "metric": metric,
                    "competitor": competitor,
                    "shared_datasets": int(len(sub)),
                    "wins": int((win_delta > 0).sum()),
                    "losses": int((win_delta < 0).sum()),
                    "ties": int((win_delta == 0).sum()),
                    "win_rate": float((win_delta > 0).mean()),
                    "mean_raw_delta": float(raw_delta.mean()),
                    "median_raw_delta": float(raw_delta.median()),
                    "higher_is_better": higher,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["win_rate", "shared_datasets", "mean_raw_delta"],
        ascending=[False, False, False],
    )


def write_outputs(
    *,
    baseline_path: Path,
    ours_path: Path | None,
    output_dir: Path,
    our_embedder: str | None,
    collapse_names: bool,
    write_debug_tables: bool = False,
) -> None:
    baseline = load_praski_csv(baseline_path)

    if ours_path is not None:
        ours = load_praski_csv(ours_path)
        df = pd.concat([baseline, ours], ignore_index=True)
    else:
        df = baseline

    output_dir.mkdir(parents=True, exist_ok=True)

    table6 = make_table6_like(df)
    table1 = make_table1_like(df, collapse_names=collapse_names)

    table6.to_csv(output_dir / "table6_like.csv", index=False)
    table1.to_csv(output_dir / "table1_like.csv", index=False)

    if our_embedder is not None:
        pairwise = make_pairwise_vs_ours(df, ours=our_embedder)
        pairwise.to_csv(output_dir / "pairwise_vs_ours.csv", index=False)

    if write_debug_tables:
        make_dataset_winners(df).to_csv(
            output_dir / "dataset_winners.csv",
            index=False,
        )
        make_table6_like(df).to_csv(
            output_dir / "table6_like_unannotated.csv",
            index=False,
        )
        make_table1_like(df, collapse_names=collapse_names).to_csv(
            output_dir / "table1_like_unannotated.csv",
            index=False,
        )

    manifest = pd.DataFrame(
        [
            {
                "baseline_path": str(baseline_path),
                "ours_path": str(ours_path) if ours_path is not None else None,
                "our_embedder": our_embedder,
                "n_rows": len(df),
                "n_datasets": df["dataset"].nunique(),
                "n_embedders": df["embedder"].nunique(),
            }
        ]
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ours", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--our-embedder", type=str, default=None)
    parser.add_argument(
        "--no-collapse-names",
        action="store_true",
        help="Do not collapse ChemBERTa_[10M][MTR] style names for table1_like.csv.",
    )
    parser.add_argument(
        "--write-debug-tables",
        action="store_true",
        help="Also write unannotated and dataset-winner debug tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(
        baseline_path=args.baseline,
        ours_path=args.ours,
        output_dir=args.output_dir,
        our_embedder=args.our_embedder,
        collapse_names=not args.no_collapse_names,
        write_debug_tables=args.write_debug_tables,
    )


if __name__ == "__main__":
    main()
