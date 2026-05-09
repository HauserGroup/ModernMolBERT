import json
import subprocess
import sys
import pytest
from pathlib import Path

import numpy as np
import pandas as pd

from modernmolbert.eval.datasets import EvalDataset, load_csv_eval_dataset
from modernmolbert.eval.downstream import FrozenDownstreamConfig, fit_predict_downstream
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.metrics import compute_classification_metrics
from modernmolbert.eval.registry import make_featurizer
from modernmolbert.eval.runner import FrozenBenchmarkRunner


def test_feature_batch_shape_check_passes() -> None:
    batch = FeatureBatch(
        X=np.ones((2, 4), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )
    batch.check(3)


def test_feature_batch_shape_check_fails_on_bad_row_count() -> None:
    batch = FeatureBatch(
        X=np.ones((1, 4), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )

    try:
        batch.check(3)
    except ValueError as e:
        assert "Number of rows" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_dummy_featurizer_valid_mask() -> None:
    featurizer = DummyFeaturizer(n_features=4)
    out = featurizer.featurize_smiles(["CCO", "", "N"])
    out.check(3)

    assert out.valid_mask.tolist() == [True, False, True]
    assert out.X.shape == (2, 4)


def test_registry_makes_dummy_featurizer() -> None:
    featurizer = make_featurizer("dummy", name="dummy_test", n_features=3)
    out = featurizer.featurize_smiles(["CCO"])
    assert out.X.shape == (1, 3)


def test_eval_dataset_check() -> None:
    train = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]})
    test = pd.DataFrame({"smiles": ["CO", "CN"], "label": [0, 1]})

    ds = EvalDataset(
        name="tiny",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )
    ds.check()


def test_load_csv_eval_dataset(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]}).to_csv(
        train_csv, index=False
    )
    pd.DataFrame({"smiles": ["CO", "CN"], "label": [0, 1]}).to_csv(
        test_csv, index=False
    )

    ds = load_csv_eval_dataset(
        name="csv_tiny",
        task_type="classification",
        task_names=["label"],
        train_csv=train_csv,
        test_csv=test_csv,
    )

    assert ds.name == "csv_tiny"
    assert ds.task_type == "classification"
    assert len(ds.train) == 2
    assert len(ds.test) == 2


def test_downstream_classification_fixed_model() -> None:
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=np.float64,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.float64)
    X_eval = np.array([[0.05, 0.0], [1.05, 1.0]], dtype=np.float64)

    pred = fit_predict_downstream(
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        config=FrozenDownstreamConfig(),
    )

    assert pred.y_pred.shape == (2,)
    assert pred.y_score is not None
    assert pred.y_score.shape == (2,)
    assert pred.metadata["downstream_model"] == "logistic_regression"


def test_downstream_regression_fixed_model() -> None:
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    X_eval = np.array([[1.5], [2.5]], dtype=np.float64)

    pred = fit_predict_downstream(
        task_type="regression",
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        config=FrozenDownstreamConfig(regression_alpha=1.0),
    )

    assert pred.y_pred.shape == (2,)
    assert pred.y_score is None
    assert pred.metadata["downstream_model"] == "ridge"


def test_frozen_runner_classification(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CCCl", "CCBr", "CO"],
            "label": [0, 0, 1, 1, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCBr", "CO"],
            "label": [0, 1, 1, 0],
        }
    )

    ds = EvalDataset(
        name="tiny_classification",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(
        downstream_config=FrozenDownstreamConfig(),
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    result = runner.run(
        dataset=ds,
        featurizer=DummyFeaturizer(n_features=8),
        output_dir=tmp_path / "out",
    )

    assert len(result.task_results) == 1
    assert result.skipped_tasks == []

    metrics = result.task_results[0].metrics
    assert "accuracy" in metrics
    assert "roc_auc" in metrics

    assert (tmp_path / "out" / "results.json").exists()
    assert (tmp_path / "out" / "results.csv").exists()

    csv = pd.read_csv(tmp_path / "out" / "results.csv")
    assert "train_feature_invalid_rate" in csv.columns
    assert "eval_feature_invalid_rate" in csv.columns
    assert "model_type" in csv.columns
    assert "standardize" in csv.columns


def test_frozen_runner_regression(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC", "CCCl"],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CO", "CN", "CCBr"],
            "y": [0.15, 0.25, 0.55],
        }
    )

    ds = EvalDataset(
        name="tiny_regression",
        task_type="regression",
        task_names=["y"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(
        downstream_config=FrozenDownstreamConfig(),
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    result = runner.run(
        dataset=ds,
        featurizer=DummyFeaturizer(n_features=8),
        output_dir=tmp_path / "out",
    )

    assert len(result.task_results) == 1

    metrics = result.task_results[0].metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_runner_cache_reuse(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC"],
            "label": [0, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CO", "CCBr"],
            "label": [0, 1],
        }
    )

    ds = EvalDataset(
        name="cache_test",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    featurizer = DummyFeaturizer(n_features=8)

    runner.run(dataset=ds, featurizer=featurizer)
    runner.run(dataset=ds, featurizer=featurizer)

    cache_files = list((tmp_path / "cache").rglob("features.npy"))
    assert cache_files


def test_runner_records_structured_skip_for_one_class_train(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC"],
            "label": [1, 1, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CO", "CCBr"],
            "label": [1, 1],
        }
    )

    ds = EvalDataset(
        name="skip_test",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(cache_dir=tmp_path / "cache", use_cache=False)
    result = runner.run(dataset=ds, featurizer=DummyFeaturizer(n_features=8))

    assert len(result.task_results) == 0
    assert len(result.skipped_tasks) == 1
    assert result.skipped_tasks[0].reason == "classification_train_has_single_class"


def test_classification_metrics_one_class_returns_nans_for_rank_metrics() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7])

    metrics = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, y_score=y_score
    )

    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["average_precision"])


def test_cli_run_frozen_benchmark_with_dummy_featurizer(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    config_json = tmp_path / "dummy.json"
    output_dir = tmp_path / "results"

    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CCCl", "CCBr", "CO"],
            "label": [0, 0, 1, 1, 1, 0],
        }
    ).to_csv(train_csv, index=False)

    pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCBr", "CO"],
            "label": [0, 1, 1, 0],
        }
    ).to_csv(test_csv, index=False)

    config_json.write_text(
        json.dumps(
            {
                "type": "dummy",
                "name": "dummy_cli",
                "n_features": 8,
            }
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "modernmolbert.eval.cli.run_frozen_benchmark",
        "--name",
        "tiny_cli",
        "--task_type",
        "classification",
        "--task_names",
        "label",
        "--train_csv",
        str(train_csv),
        "--test_csv",
        str(test_csv),
        "--featurizer_config",
        str(config_json),
        "--output_dir",
        str(output_dir),
        "--cache_dir",
        str(tmp_path / "cache"),
        "--batch_size",
        "2",
    ]

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()

    payload = json.loads((output_dir / "results.json").read_text())
    assert payload["dataset"] == "tiny_cli"
    assert payload["featurizer"] == "dummy_cli"
    assert len(payload["task_results"]) == 1


def test_feature_batch_shape_check_fails_on_non_numeric_features() -> None:
    batch = FeatureBatch(
        X=np.array([["a"], ["b"]], dtype=object),
        valid_mask=np.array([True, True]),
    )

    try:
        batch.check(2)
    except TypeError as e:
        assert "numeric" in str(e)
    else:
        raise AssertionError("Expected TypeError")


def test_feature_batch_check_accepts_valid_batch() -> None:
    batch = FeatureBatch(
        X=np.zeros((2, 8), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )

    batch.check(n_inputs=3)


def test_feature_batch_check_rejects_bad_row_count() -> None:
    batch = FeatureBatch(
        X=np.zeros((1, 8), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )

    with pytest.raises(ValueError, match="Number of rows"):
        batch.check(n_inputs=3)


def test_feature_batch_check_rejects_non_numeric_features() -> None:
    batch = FeatureBatch(
        X=np.array([["a"], ["b"]], dtype=object),
        valid_mask=np.array([True, True]),
    )

    with pytest.raises(TypeError, match="numeric"):
        batch.check(n_inputs=2)
