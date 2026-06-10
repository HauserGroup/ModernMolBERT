#!/usr/bin/env python3

"""
build_benchmark_result_frames.py

Build unified benchmark result frames from:

1. Praski benchmark:
   data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv

2. Own ModernMolBERT results:
   outputs/eval/praski_best_*/results.csv

Special case:
   outputs/eval/praski_best_base_standard/results.csv
   is forced to embedder = modernmolbert_best_base

Outputs:
   outputs/eval/combined_benchmark_results.csv
   outputs/eval/best_metric_by_dataset_embedder.csv
   outputs/eval/dabest/dabest_test_metric__roc_auc.csv

This script only wrangles data. It does not subset models for plotting.
"""

from pathlib import Path
import re
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(".").resolve()

PRASKI_CSV = PROJECT_ROOT / "data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv"
OWN_RESULTS_ROOT = PROJECT_ROOT / "outputs/eval"

OUTPUT_DIR = PROJECT_ROOT / "outputs/eval"
DABEST_DIR = OUTPUT_DIR / "dabest"

COMBINED_OUT = OUTPUT_DIR / "combined_benchmark_results.csv"
BEST_OUT = OUTPUT_DIR / "best_metric_by_dataset_embedder.csv"


# =============================================================================
# Name handling
# =============================================================================


def clean_embedder_name(name: object) -> str:
    """
    Normalize embedder names.

    Removes run-specific suffixes like:
        modernmolbert_best_standard__subsample_train8000_seed42
    ->  modernmolbert_best_standard
    """
    s = str(name)
    s = re.sub(r"__subsample.*$", "", s)
    return s


def infer_embedder_from_result_path(path: Path) -> str:
    """
    Infer embedder name from paths like:

        outputs/eval/praski_best_standard/results.csv
        -> modernmolbert_best_standard

        outputs/eval/praski_best_span/results.csv
        -> modernmolbert_best_span

        outputs/eval/praski_best_base_standard/results.csv
        -> modernmolbert_best_base

    This is intentionally explicit and simple.
    """
    result_dir = path.parent.name

    if result_dir == "praski_best_base_standard":
        return "modernmolbert_best_base"

    if result_dir.startswith("praski_best_"):
        suffix = result_dir.removeprefix("praski_best_")
        return clean_embedder_name(f"modernmolbert_best_{suffix}")

    return clean_embedder_name(result_dir)


def safe_name(x: object) -> str:
    """Filesystem-safe name for metric-specific exports."""
    s = str(x)
    s = re.sub(r"[^\w.-]+", "_", s)
    s = s.strip("_")
    return s or "all"


# =============================================================================
# File discovery
# =============================================================================


def find_own_result_files(root: Path) -> list[Path]:
    """
    Find own result files.

    Expected layout:
        outputs/eval/praski_best_*/results.csv

    This intentionally only looks for results.csv to avoid accidentally loading
    summary outputs or unrelated CSV files.
    """
    paths = sorted(root.glob("praski_best_*/results.csv"))
    return [p for p in paths if p.is_file()]


# =============================================================================
# Schema normalization
# =============================================================================


def ensure_test_metric_name(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every frame has test_metric_name."""
    df = df.copy()

    if "test_metric_name" not in df.columns:
        df["test_metric_name"] = "roc_auc"

    return df


def normalize_praski(path: Path) -> pd.DataFrame:
    """Load Praski benchmark CSV and normalize schema."""
    if not path.exists():
        raise FileNotFoundError(f"Praski benchmark CSV not found: {path}")

    df = pd.read_csv(path)
    df = ensure_test_metric_name(df)

    required = {"dataset", "embedder", "model", "test_metric"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Praski CSV is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["embedder"] = df["embedder"].map(clean_embedder_name)
    df["test_metric"] = pd.to_numeric(df["test_metric"], errors="coerce")

    keep = [
        "dataset",
        "embedder",
        "model",
        "test_metric_name",
        "test_metric",
    ]

    optional = [
        "cv_metric",
        "cv_metric_name",
        "task",
        "task_type",
        "split",
    ]

    keep += [c for c in optional if c in df.columns]

    out = df[keep].copy()
    out["result_source"] = "praski"

    return out


def normalize_own_result(path: Path) -> pd.DataFrame:
    """
    Load one own result CSV and normalize schema.

    The embedder name is always inferred from the directory name.
    This deliberately overrides any embedder column inside the CSV.

    This ensures:
        outputs/eval/praski_best_base_standard/results.csv
    becomes:
        modernmolbert_best_base
    """
    df = pd.read_csv(path)

    rename_map = {
        "downstream": "model",
        "model_kind": "model",
        "test_roc_auc": "test_metric",
        "roc_auc": "test_metric",
        "cv_roc_auc": "cv_metric",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = ensure_test_metric_name(df)

    # Critical behavior: path determines own embedder identity.
    df["embedder"] = infer_embedder_from_result_path(path)

    if "model" not in df.columns:
        df["model"] = "best"

    df["embedder"] = df["embedder"].map(clean_embedder_name)
    df["test_metric"] = pd.to_numeric(df["test_metric"], errors="coerce")

    required = {"dataset", "embedder", "model", "test_metric", "test_metric_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Own result file missing required columns after normalization: {path}\n"
            f"Missing: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    keep = [
        "dataset",
        "embedder",
        "model",
        "test_metric_name",
        "test_metric",
    ]

    optional = [
        "cv_metric",
        "cv_metric_name",
        "task",
        "task_type",
        "checkpoint",
        "embedding",
        "run_dir",
    ]

    keep += [c for c in optional if c in df.columns]

    out = df[keep].copy()
    out["result_source"] = str(path.relative_to(PROJECT_ROOT))

    return out


# =============================================================================
# Build frames
# =============================================================================


def collapse_best_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse downstream model/head dimension.

    For each:
        dataset × embedder × test_metric_name

    keep the row with the highest test_metric.

    This also handles repeated rows introduced by stripped __subsample suffixes.
    """
    required = {"dataset", "embedder", "test_metric_name", "test_metric", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot collapse; missing columns: {sorted(missing)}")

    out = df.copy()
    out["embedder"] = out["embedder"].map(clean_embedder_name)
    out["test_metric"] = pd.to_numeric(out["test_metric"], errors="coerce")
    out = out.dropna(subset=["test_metric"])

    group_keys = ["dataset", "embedder", "test_metric_name"]

    out = (
        out.sort_values(
            [*group_keys, "test_metric", "model"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(subset=group_keys, keep="first")
        .reset_index(drop=True)
    )

    keep = [
        "dataset",
        "embedder",
        "test_metric_name",
        "test_metric",
        "model",
    ]

    optional = [
        "cv_metric",
        "result_source",
    ]

    keep += [c for c in optional if c in out.columns]

    return out[keep]


def write_dabest_exports(best_df: pd.DataFrame) -> None:
    """
    Write DABEST-ready long-format CSVs.

    DABEST format:
        dataset, embedder, test_metric
    """
    DABEST_DIR.mkdir(exist_ok=True, parents=True)

    for metric_name, metric_df in best_df.groupby("test_metric_name", sort=True):
        out = metric_df[["dataset", "embedder", "test_metric"]].copy()

        metric_safe = safe_name(metric_name)
        out_path = DABEST_DIR / f"dabest_test_metric__{metric_safe}.csv"

        out.to_csv(out_path, index=False)

        n_datasets = out["dataset"].nunique()
        n_embedders = out["embedder"].nunique()
        coverage = out.groupby("dataset")["embedder"].nunique()
        complete = int((coverage == n_embedders).sum())

        print(f"[{metric_name}] DABEST export")
        print(f"  datasets: {n_datasets}")
        print(f"  embedders: {n_embedders}")
        print(f"  complete datasets: {complete}/{n_datasets}")
        print(f"  wrote: {out_path}")
        print()


def print_summary(best_df: pd.DataFrame) -> None:
    """Print sanity summary."""
    print("Final best_df summary")
    print("---------------------")
    print(f"rows: {len(best_df)}")
    print(f"datasets: {best_df['dataset'].nunique()}")
    print(f"embedders: {best_df['embedder'].nunique()}")
    print(f"metrics: {sorted(best_df['test_metric_name'].unique())}")
    print()

    print("ModernMolBERT rows")
    print("------------------")
    modern = (
        best_df[best_df["embedder"].str.contains("modernmolbert", case=False, na=False)]
        .groupby("embedder")
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_metric=("test_metric", "mean"),
            median_metric=("test_metric", "median"),
        )
        .sort_values("mean_metric", ascending=False)
    )

    if len(modern) == 0:
        print("No ModernMolBERT embedders found.")
    else:
        print(modern.to_string())
    print()

    print("All embedders")
    print("-------------")
    for e in sorted(best_df["embedder"].unique()):
        print(f"  {e}")
    print()

    summary = (
        best_df.groupby(["test_metric_name", "embedder"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_metric=("test_metric", "mean"),
            median_metric=("test_metric", "median"),
        )
        .sort_values(["test_metric_name", "mean_metric"], ascending=[True, False])
    )

    print("Top rows by mean metric")
    print("-----------------------")
    print(summary.head(40).to_string(index=False))
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    DABEST_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Praski CSV: {PRASKI_CSV}")
    print(f"Own results root: {OWN_RESULTS_ROOT}")
    print()

    praski = normalize_praski(PRASKI_CSV)

    own_paths = find_own_result_files(OWN_RESULTS_ROOT)

    print(f"Found {len(own_paths)} own results.csv file(s):")
    for p in own_paths:
        inferred = infer_embedder_from_result_path(p)
        print(f"  {p}  ->  {inferred}")
    print()

    own_frames = [normalize_own_result(p) for p in own_paths]

    if own_frames:
        own = pd.concat(own_frames, ignore_index=True, sort=False)
        combined = pd.concat([praski, own], ignore_index=True, sort=False)
    else:
        combined = praski.copy()

    combined["embedder"] = combined["embedder"].map(clean_embedder_name)

    combined.to_csv(COMBINED_OUT, index=False)
    print(f"Wrote combined raw results: {COMBINED_OUT}")

    best_df = collapse_best_head(combined)
    best_df.to_csv(BEST_OUT, index=False)
    print(f"Wrote best dataset/embedder results: {BEST_OUT}")
    print()

    print_summary(best_df)
    write_dabest_exports(best_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
