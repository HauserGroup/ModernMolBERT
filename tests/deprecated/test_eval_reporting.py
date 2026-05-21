import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from modernmolbert.eval.reporting import (
    average_rank,
    available_metric_columns,
    best_by_dataset_task,
    create_benchmark_deliverables,
    load_results,
    metric_matrix,
    primary_metric_summary,
    summarize_results,
    skipped_task_summary,
    write_summary_tables,
    write_standard_plots,
)


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
            "eval_feature_invalid_rate": [0.0, 0.0, 0.1, 0.2],
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


def test_available_metric_columns_skips_all_nan_columns() -> None:
    results = pd.DataFrame(
        {
            "dataset": ["bbbp"],
            "task": ["p_np"],
            "task_type": ["classification"],
            "featurizer": ["ecfp4"],
            "downstream_name": ["logistic"],
            "downstream_model": ["logistic_regression"],
            "seed": [13],
            "roc_auc": [np.nan],
        }
    )

    assert "roc_auc" not in available_metric_columns(results)


def test_summarize_results(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)
    summary = summarize_results(results)

    assert len(summary) == 4
    assert "roc_auc_mean" in summary.columns
    assert "rmse_mean" in summary.columns


def test_primary_metric_summary_and_best_by_dataset_task(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results(path)

    results = load_results(path)
    primary = primary_metric_summary(results)
    best = best_by_dataset_task(results)

    assert "primary_metric_mean" in primary.columns
    bbbp_best = best[best["dataset"] == "bbbp"].iloc[0]
    esol_best = best[best["dataset"] == "esol"].iloc[0]
    assert bbbp_best["featurizer"] == "modernmolbert"
    assert esol_best["featurizer"] == "modernmolbert"


def test_skipped_task_summary() -> None:
    skipped = pd.DataFrame(
        {
            "dataset": ["bbbp", "bbbp"],
            "task": ["p_np", "p_np"],
            "featurizer": ["ecfp4", "ecfp4"],
            "downstream_name": ["logistic", "logistic"],
            "seed": [13, 17],
            "reason": ["classification_train_has_single_class"] * 2,
        }
    )

    summary = skipped_task_summary(skipped)

    assert summary["n_skipped"].sum() == 2


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


def test_create_benchmark_deliverables_with_prediction_figures(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    _write_results(sweep_dir / "results.csv")
    pd.DataFrame({"dataset": ["tox21"], "task": ["NR"], "reason": ["empty"]}).to_csv(
        sweep_dir / "skipped_tasks.csv",
        index=False,
    )
    (sweep_dir / "manifest.json").write_text(
        json.dumps(
            {
                "suite_name": "toy",
                "runs": [
                    {
                        "dataset": "bbbp",
                        "featurizer": "ecfp4",
                        "downstream_name": "logistic",
                        "seed": 13,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_prediction_artifact(
        sweep_dir / "predictions" / "bbbp" / "p_np" / "ecfp4" / "logistic" / "seed_13.npz",
        metadata={
            "dataset": "bbbp",
            "task": "p_np",
            "task_type": "classification",
            "split": "test",
            "featurizer": "ecfp4",
            "downstream_name": "logistic",
            "downstream_model": "logistic_regression",
            "seed": 13,
        },
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        y_score=np.array([0.1, 0.9, 0.6, 0.8]),
    )
    _write_prediction_artifact(
        sweep_dir
        / "predictions"
        / "esol"
        / "measured_log_solubility"
        / "ecfp4"
        / "ridge"
        / "seed_13.npz",
        metadata={
            "dataset": "esol",
            "task": "measured log solubility",
            "task_type": "regression",
            "split": "test",
            "featurizer": "ecfp4",
            "downstream_name": "ridge",
            "downstream_model": "ridge",
            "seed": 13,
        },
        y_true=np.array([0.0, 1.0, 2.0, 3.0]),
        y_pred=np.array([0.1, 0.9, 2.2, 2.8]),
    )

    manifest = create_benchmark_deliverables(
        sweep_dir=sweep_dir,
        output_dir=sweep_dir / "deliverables",
    )

    assert (sweep_dir / "deliverables" / "README.md").exists()
    assert (sweep_dir / "deliverables" / "manifest.json").exists()
    assert (sweep_dir / "deliverables" / "tables" / "summary.csv").exists()
    assert (sweep_dir / "deliverables" / "tables" / "primary_metric_summary.csv").exists()
    assert (sweep_dir / "deliverables" / "tables" / "best_by_dataset_task.csv").exists()
    assert (sweep_dir / "deliverables" / "figures" / "roc_bbbp_p_np.png").exists()
    assert (sweep_dir / "deliverables" / "figures" / "pr_bbbp_p_np.png").exists()
    assert (sweep_dir / "deliverables" / "figures" / "calibration_bbbp_p_np.png").exists()
    assert (sweep_dir / "deliverables" / "figures" / "confusion_bbbp_p_np.png").exists()
    assert (
        sweep_dir / "deliverables" / "figures" / "regression_esol_measured_log_solubility.png"
    ).exists()
    assert (
        sweep_dir / "deliverables" / "figures" / "residuals_esol_measured_log_solubility.png"
    ).exists()
    assert manifest["n_prediction_artifacts"] == 2
    assert not manifest["warnings"]


def test_create_benchmark_deliverables_without_predictions_warns(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    _write_results(sweep_dir / "results.csv")

    manifest = create_benchmark_deliverables(
        sweep_dir=sweep_dir,
        output_dir=sweep_dir / "deliverables",
    )

    assert (sweep_dir / "deliverables" / "tables" / "summary.csv").exists()
    assert (sweep_dir / "deliverables" / "figures" / "primary_metric_by_dataset.png").exists()
    assert manifest["warnings"]


def test_create_benchmark_deliverables_cli_smoke(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    _write_results(sweep_dir / "results.csv")

    output_dir = sweep_dir / "cli_deliverables"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.eval.cli.create_benchmark_deliverables",
            "--sweep_dir",
            str(sweep_dir),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "tables" / "summary.csv").exists()
    assert (output_dir / "figures" / "primary_metric_by_dataset.png").exists()
    assert (output_dir / "manifest.json").exists()


def _write_prediction_artifact(
    path: Path,
    *,
    metadata: dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "y_true": y_true,
        "y_pred": y_pred,
        "eval_original_index": np.arange(len(y_true)),
    }
    if y_score is not None:
        arrays["y_score"] = y_score
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
