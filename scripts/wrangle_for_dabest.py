# %%
from pathlib import Path
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = "../data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv"
# Path to your benchmark results CSV (or glob pattern if one file per model)
OWN_RESULTS_CSV = "../data/modernmolbert_benchmark_results.csv"
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CORR_METHOD = "spearman"
N_BINS = 25


# =============================================================================
# Load and normalise own benchmark results
# =============================================================================


def load_own_results(path: str | Path) -> pd.DataFrame:
    """
    Load ModernMolBERT benchmark results and return a dataframe that
    matches the Praski schema:

        dataset | embedder | model | test_metric [| cv_metric]

    Adjust column renaming below to match your actual output format.

    Expected own-results format (adapt as needed):
        dataset       : benchmark dataset name  (e.g. "BBBP")
        embedder      : model name              (e.g. "ModernMolBERT-small")
        downstream    : downstream model type   (e.g. "rf", "knn", "logistic")
        test_roc_auc  : held-out ROC-AUC
        cv_roc_auc    : cross-val ROC-AUC       (optional)
    """
    own = pd.read_csv(path)

    # ── Rename to match Praski schema ─────────────────────────────────────
    # TODO: adjust these keys to match your actual column names
    own = own.rename(
        columns={
            "downstream": "model",  # nuisance downstream model
            "test_roc_auc": "test_metric",  # primary metric
            "cv_roc_auc": "cv_metric",  # optional; drop line if absent
        }
    )

    # Keep only the columns the rest of the pipeline expects
    keep = ["dataset", "embedder", "model", "test_metric"]
    if "cv_metric" in own.columns:
        keep.append("cv_metric")
    return own[keep]


# =============================================================================
# Load raw data
# =============================================================================

praski = pd.read_csv(INPUT_CSV)
own = load_own_results(OWN_RESULTS_CSV)

# Concatenate; own results appear as additional embedders
df = pd.concat([praski, own], ignore_index=True)

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
