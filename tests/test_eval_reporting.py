from pathlib import Path

import pandas as pd

from modernmolbert.eval.reporting import (
    average_rank,
    available_metric_columns,
    load_results,
    metric_matrix,
    summarize_results,
    write_summary_tables,
)


from modernmolbert.eval.reporting import write_standard_plots


def _write_results(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["bbbp", "bbbp", "esol", "esol"],
            "task": ["p_np", "p_np", "measured log solubility", "measured log solubility"],
            "task_type": ["classification", "classification", "regression", "regression"],
            "featurizer": ["ecfp4", "modernmolbert", "ecfp4", "modernmolbert"],
            "downstream_name": ["logistic", "logistic", "ridge", "ridge"],
            "downstream_model": ["logistic_regression", "logistic_regression", "ridge", "ridge"],
            "seed": [13, 13, 13, 13],
            "roc_auc": [0.80, 0.85, None, None],
            "rmse": [None, None, 1.20, 1.10],
        }
    )
    frame.to_csv(path, index=False)


def test_load_results_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)

    assert len(results) == 4


def test_available_metric_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)

    assert "roc_auc" in available_metric_columns(results)
    assert "rmse" in available_metric_columns(results)


def test_summarize_results(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)
    summary = summarize_results(results)

    assert len(summary) == 4
    assert "roc_auc_mean" in summary.columns
    assert "rmse_mean" in summary.columns


def test_metric_matrix(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)
    matrix = metric_matrix(results, metric="roc_auc")

    assert "ecfp4" in matrix.columns
    assert "modernmolbert" in matrix.columns


def test_average_rank_higher_metric(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)
    rank = average_rank(
        results.dropna(subset=["roc_auc"]),
        metric="roc_auc",
        candidate_col="featurizer",
        direction="higher",
    )

    assert rank.iloc[0]["featurizer"] == "modernmolbert"


def test_write_summary_tables(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    outputs = write_summary_tables(
        results_path=path,
        output_dir=tmp_path / "report",
    )

    assert outputs["summary"].exists()


def test_write_standard_plots(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    outputs = write_standard_plots(
        results_path=path,
        output_dir=tmp_path / "plots",
    )

    assert outputs
    assert all(path.exists() for path in outputs.values())
