import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.compare_praski_tables import (
    best_head_per_dataset,
    make_pairwise_vs_ours,
    make_table1_like,
    make_table6_like,
    metric_higher_is_better,
    normalize_head_name,
    rank_within_dataset,
)


def make_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Dataset A: classification / ROC-AUC, higher is better
            {
                "dataset": "AMES",
                "task": "classification",
                "embedder": "ECFP",
                "model": "ridge",
                "cv_metric_name": "roc_auc",
                "cv_metric": 0.70,
                "test_metric_name": "roc_auc",
                "test_metric": 0.75,
            },
            {
                "dataset": "AMES",
                "task": "classification",
                "embedder": "ECFP",
                "model": "rf",
                "cv_metric_name": "roc_auc",
                "cv_metric": 0.71,
                "test_metric_name": "roc_auc",
                "test_metric": 0.80,
            },
            {
                "dataset": "AMES",
                "task": "classification",
                "embedder": "ModernMolBERT_SELFIES_ChEMBL36_2M",
                "model": "ridge",
                "cv_metric_name": "roc_auc",
                "cv_metric": 0.72,
                "test_metric_name": "roc_auc",
                "test_metric": 0.82,
            },
            {
                "dataset": "AMES",
                "task": "classification",
                "embedder": "ModernMolBERT_SELFIES_ChEMBL36_2M",
                "model": "knn",
                "cv_metric_name": "roc_auc",
                "cv_metric": 0.71,
                "test_metric_name": "roc_auc",
                "test_metric": 0.78,
            },
            # Dataset B: classification / ROC-AUC
            {
                "dataset": "MoleculeNet_BBBP",
                "task": "classification",
                "embedder": "ECFP",
                "model": "ridge",
                "cv_metric_name": "roc_auc",
                "cv_metric": 0.65,
                "test_metric_name": "roc_auc",
                "test_metric": 0.70,
            },
            {
                "dataset": "MoleculeNet_BBBP",
                "task": "classification",
                "embedder": "ModernMolBERT_SELFIES_ChEMBL36_2M",
                "model": "ridge",
                "cv_metric_name": "roc_auc",
                "cv_metric": 0.66,
                "test_metric_name": "roc_auc",
                "test_metric": 0.68,
            },
            # Dataset C: regression / RMSE, lower is better
            {
                "dataset": "MoleculeNet_ESOL",
                "task": "regression",
                "embedder": "ECFP",
                "model": "ridge",
                "cv_metric_name": "rmse",
                "cv_metric": 0.90,
                "test_metric_name": "rmse",
                "test_metric": 0.85,
            },
            {
                "dataset": "MoleculeNet_ESOL",
                "task": "regression",
                "embedder": "ModernMolBERT_SELFIES_ChEMBL36_2M",
                "model": "ridge",
                "cv_metric_name": "rmse",
                "cv_metric": 0.88,
                "test_metric_name": "rmse",
                "test_metric": 0.80,
            },
        ]
    )


def test_normalize_head_name_maps_ridge_to_linear() -> None:
    assert normalize_head_name("ridge") == "linear"
    assert normalize_head_name("logistic") == "linear"
    assert normalize_head_name("rf") == "rf"
    assert normalize_head_name("knn") == "knn"


def test_metric_direction() -> None:
    assert metric_higher_is_better("roc_auc") is True
    assert metric_higher_is_better("average_precision") is True
    assert metric_higher_is_better("rmse") is False
    assert metric_higher_is_better("mae") is False


def test_unknown_metric_direction_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric direction"):
        metric_higher_is_better("not_a_metric")


def test_best_head_per_dataset_selects_highest_roc_auc() -> None:
    df = make_rows()
    df["head"] = df["model"].map(normalize_head_name)
    df["metric"] = df["test_metric_name"]

    best = best_head_per_dataset(df)

    row = best[(best["dataset"] == "AMES") & (best["embedder"] == "ECFP")].iloc[0]

    assert row["model"] == "rf"
    assert row["test_metric"] == 0.80


def test_best_head_per_dataset_selects_lowest_rmse() -> None:
    df = make_rows()
    df["head"] = df["model"].map(normalize_head_name)
    df["metric"] = df["test_metric_name"]

    best = best_head_per_dataset(df)

    row = best[
        (best["dataset"] == "MoleculeNet_ESOL")
        & (best["embedder"] == "ModernMolBERT_SELFIES_ChEMBL36_2M")
    ].iloc[0]

    assert row["test_metric_name"] == "rmse"
    assert row["test_metric"] == 0.80


def test_rank_within_dataset_handles_roc_auc_and_rmse_direction() -> None:
    df = make_rows()
    df["head"] = df["model"].map(normalize_head_name)
    df["metric"] = df["test_metric_name"]

    best = best_head_per_dataset(df)
    ranked = rank_within_dataset(best)

    ames_ours = ranked[
        (ranked["dataset"] == "AMES") & (ranked["embedder"] == "ModernMolBERT_SELFIES_ChEMBL36_2M")
    ].iloc[0]
    esol_ours = ranked[
        (ranked["dataset"] == "MoleculeNet_ESOL")
        & (ranked["embedder"] == "ModernMolBERT_SELFIES_ChEMBL36_2M")
    ].iloc[0]

    assert ames_ours["rank"] == 1
    assert esol_ours["rank"] == 1


def test_make_table6_like_contains_annotation_ready_model_column() -> None:
    df = make_rows()
    df["head"] = df["model"].map(normalize_head_name)
    df["metric"] = df["test_metric_name"]

    table = make_table6_like(df)

    assert "Model" in table.columns
    assert "rank_best" in table.columns
    assert "metric_best" in table.columns
    assert "ModernMolBERT_SELFIES_ChEMBL36_2M" in set(table["Model"])


def test_make_table1_like_collapses_and_summarizes() -> None:
    df = make_rows()
    df["head"] = df["model"].map(normalize_head_name)
    df["metric"] = df["test_metric_name"]

    table = make_table1_like(df, collapse_names=True)

    assert "Model" in table.columns
    assert "Mean_rank" in table.columns
    assert "Mean_metric" in table.columns
    assert "ModernMolBERT_SELFIES_ChEMBL36_2M" in set(table["Model"])


def test_pairwise_vs_ours_counts_wins_losses_with_metric_direction() -> None:
    df = make_rows()
    df["head"] = df["model"].map(normalize_head_name)
    df["metric"] = df["test_metric_name"]

    pairwise = make_pairwise_vs_ours(
        df,
        ours="ModernMolBERT_SELFIES_ChEMBL36_2M",
    )

    ecfp_rows = pairwise[pairwise["competitor"] == "ECFP"]

    assert not ecfp_rows.empty
    assert int(ecfp_rows["wins"].sum()) == 2
    assert int(ecfp_rows["losses"].sum()) == 1
