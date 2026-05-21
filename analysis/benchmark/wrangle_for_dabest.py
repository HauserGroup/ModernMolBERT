# %%
from pathlib import Path
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = Path("../data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv")
OWN_RESULTS_CSV = Path("../data/modernmolbert_benchmark_results.csv")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Load and align data
# =============================================================================


def load_own_results(path: Path) -> pd.DataFrame:
    """Load ModernMolBERT results aligned to Praski schema columns."""
    own = pd.read_csv(path)
    own = own.rename(
        columns={
            "downstream": "model",
            "test_roc_auc": "test_metric",
            "cv_roc_auc": "cv_metric",
        }
    )
    keep = ["dataset", "embedder", "model", "test_metric"]
    if "cv_metric" in own.columns:
        keep.append("cv_metric")
    if "test_metric_name" in own.columns:
        keep.append("test_metric_name")
    return own[keep]


praski = pd.read_csv(INPUT_CSV)
own = load_own_results(OWN_RESULTS_CSV)
df = pd.concat([praski, own], ignore_index=True)


# =============================================================================
# Collapse nuisance model dimension
#
# For each (dataset, embedder, test_metric_name), keep the downstream model
# that achieved the highest test_metric. This is the "best head" per embedder
# per dataset — the number we actually care about for comparison.
# =============================================================================

group_keys = ["dataset", "embedder"]
if "test_metric_name" in df.columns:
    group_keys.append("test_metric_name")

best_df = (
    df.sort_values(
        [*group_keys, "test_metric", "model"],
        ascending=[*(True for _ in group_keys), False, True],
    )
    .drop_duplicates(subset=group_keys, keep="first")
    .reset_index(drop=True)
)[
    [
        "dataset",
        "embedder",
        *([c] if (c := "test_metric_name") in df.columns else []),
        "test_metric",
        "model",
    ]
]

best_df.to_csv(OUTPUT_DIR / "best_metric_by_dataset_embedder.csv", index=False)


# =============================================================================
# DABEST repeated-measures export
#
# DABEST repeated measures (paired="baseline") expects long format:
#
#   x      = embedder       — the condition / group being compared
#   y      = test_metric    — the measurement value
#   id_col = dataset        — the paired unit (one "subject" per dataset)
#
# Each dataset appears once per embedder, forming a paired structure.
# Datasets missing any embedder are retained; DABEST handles incomplete pairs.
#
# If multiple metric types are present (e.g. ROC-AUC vs RMSE), datasets using
# different metrics are not comparable and must be analysed separately.
# We export one CSV per metric name so each file is ready to pass directly
# to dabest.load().
#
# Example Python usage:
#
#   import dabest, pandas as pd
#   df = pd.read_csv("dabest_test_metric__roc_auc.csv")
#   analysis = dabest.load(
#       df, x="embedder", y="test_metric", id_col="dataset",
#       idx=("MolBERT", "ModernMolBERT"),
#       paired="baseline",
#   )
#   analysis.mean_diff.plot()
# =============================================================================

dabest_cols = ["dataset", "embedder", "test_metric"]

if "test_metric_name" in best_df.columns:
    groups = best_df.groupby("test_metric_name", sort=True)
else:
    groups = [("all", best_df)]

for metric_name, metric_df in groups:
    out = metric_df[dabest_cols].copy()

    safe = str(metric_name).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    path = OUTPUT_DIR / f"dabest_test_metric__{safe}.csv"
    out.to_csv(path, index=False)

    n_datasets = out["dataset"].nunique()
    n_embedders = out["embedder"].nunique()
    coverage = out.groupby("dataset")["embedder"].count()
    complete = (coverage == n_embedders).sum()

    print(f"[{metric_name}] {n_datasets} datasets × {n_embedders} embedders")
    print(f"  complete pairs: {complete}/{n_datasets} datasets")
    print(f"  embedders: {sorted(out['embedder'].unique())}")
    print(f"  → {path}")
    print()

# %%
