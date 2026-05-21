import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.suite import run_benchmark_suite, suite_config_from_dict
from modernmolbert.eval.cli.run_benchmark_suite import validate_output_dir


def _write_toy_classification_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

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

    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)


def test_suite_config_from_dict_parses_minimal_suite(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    _write_toy_classification_dataset(dataset_dir)

    suite = suite_config_from_dict(
        {
            "name": "toy_suite",
            "datasets": [
                {
                    "name": "toy",
                    "loader": "table_splits",
                    "task_type": "classification",
                    "task_names": "label",
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                    "smiles_column": "smiles",
                }
            ],
            "featurizers": [
                {
                    "type": "dummy",
                    "name": "dummy_8",
                    "n_features": 8,
                }
            ],
            "downstream_models": {
                "classification": [
                    {
                        "name": "logistic",
                        "model_type": "logistic_regression",
                        "standardize": True,
                        "params": {"max_iter": 100},
                    }
                ],
            },
            "seeds": [13],
            "eval_split": "test",
            "batch_size": 2,
            "use_cache": True,
        }
    )

    assert suite.name == "toy_suite"
    assert len(suite.datasets) == 1
    assert len(suite.featurizers) == 1
    assert len(suite.downstream_models) == 1
    assert suite.seeds == [13]
    assert suite.eval_split == "test"
    assert suite.batch_size == 2
    assert suite.use_cache is True


def test_suite_runner_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    _write_toy_classification_dataset(dataset_dir)

    suite = suite_config_from_dict(
        {
            "name": "toy_suite",
            "datasets": [
                {
                    "name": "toy",
                    "loader": "table_splits",
                    "task_type": "classification",
                    "task_names": ["label"],
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                    "smiles_column": "smiles",
                }
            ],
            "featurizers": [
                {
                    "type": "dummy",
                    "name": "dummy_8",
                    "n_features": 8,
                }
            ],
            "downstream_models": {
                "classification": [
                    {
                        "name": "logistic",
                        "model_type": "logistic_regression",
                        "standardize": True,
                        "params": {"max_iter": 100},
                    }
                ],
            },
            "seeds": [13],
            "eval_split": "test",
            "batch_size": 2,
            "use_cache": True,
        }
    )

    results = run_benchmark_suite(
        suite=suite,
        output_dir=tmp_path / "out",
    )

    assert len(results) == 1
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert (tmp_path / "out" / "cache").exists()

    csv = pd.read_csv(tmp_path / "out" / "results.csv")

    assert csv.loc[0, "dataset"] == "toy"
    assert csv.loc[0, "task"] == "label"
    assert csv.loc[0, "featurizer"] == "dummy_8"
    assert csv.loc[0, "downstream_name"] == "logistic"
    assert csv.loc[0, "seed"] == 13


def test_suite_runner_writes_prediction_artifacts_when_requested(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    _write_toy_classification_dataset(dataset_dir)

    suite = suite_config_from_dict(
        {
            "name": "toy_suite",
            "datasets": [
                {
                    "name": "toy",
                    "loader": "table_splits",
                    "task_type": "classification",
                    "task_names": ["label"],
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                    "smiles_column": "smiles",
                }
            ],
            "featurizers": [
                {
                    "type": "dummy",
                    "name": "dummy_8",
                    "n_features": 8,
                }
            ],
            "downstream_models": {
                "classification": [
                    {
                        "name": "logistic",
                        "model_type": "logistic_regression",
                        "standardize": True,
                        "params": {"max_iter": 100},
                    }
                ],
            },
            "seeds": [13],
        }
    )

    run_benchmark_suite(
        suite=suite,
        output_dir=tmp_path / "out",
        write_predictions=True,
    )

    prediction_path = (
        tmp_path / "out" / "predictions" / "toy" / "label" / "dummy_8" / "logistic" / "seed_13.npz"
    )
    assert prediction_path.exists()
    with np.load(prediction_path) as data:
        assert set(data.files) == {"y_true", "y_pred", "y_score", "eval_original_index"}
        assert data["eval_original_index"].tolist() == [0, 1]

    metadata = json.loads(prediction_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "toy"
    assert metadata["task"] == "label"
    assert metadata["task_type"] == "classification"
    assert metadata["downstream_name"] == "logistic"
    assert metadata["seed"] == 13

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["write_predictions"] is True


def test_suite_runner_does_not_write_prediction_artifacts_by_default(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    _write_toy_classification_dataset(dataset_dir)

    suite = suite_config_from_dict(
        {
            "name": "toy_suite",
            "datasets": [
                {
                    "name": "toy",
                    "loader": "table_splits",
                    "task_type": "classification",
                    "task_names": ["label"],
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                }
            ],
            "featurizers": [{"type": "dummy", "name": "dummy_8", "n_features": 8}],
            "downstream_models": {
                "classification": [
                    {
                        "name": "logistic",
                        "model_type": "logistic_regression",
                        "params": {"max_iter": 100},
                    }
                ],
            },
            "seeds": [13],
        }
    )

    run_benchmark_suite(suite=suite, output_dir=tmp_path / "out")

    assert not (tmp_path / "out" / "predictions").exists()


def test_suite_runner_skipped_tasks_do_not_write_prediction_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy_skip"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "label": [1, 1, 1]}).to_csv(
        dataset_dir / "train.csv",
        index=False,
    )
    pd.DataFrame({"smiles": ["CO", "CCBr"], "label": [0, 1]}).to_csv(
        dataset_dir / "test.csv",
        index=False,
    )

    suite = suite_config_from_dict(
        {
            "name": "toy_suite",
            "datasets": [
                {
                    "name": "toy_skip",
                    "loader": "table_splits",
                    "task_type": "classification",
                    "task_names": ["label"],
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                }
            ],
            "featurizers": [{"type": "dummy", "name": "dummy_8", "n_features": 8}],
            "downstream_models": {
                "classification": [
                    {
                        "name": "logistic",
                        "model_type": "logistic_regression",
                    }
                ],
            },
            "seeds": [13],
        }
    )

    run_benchmark_suite(
        suite=suite,
        output_dir=tmp_path / "out",
        write_predictions=True,
    )

    assert (tmp_path / "out" / "skipped_tasks.csv").exists()
    prediction_dir = tmp_path / "out" / "predictions"
    assert not prediction_dir.exists() or not list(prediction_dir.rglob("*.npz"))


def test_suite_runner_runs_multiple_seeds(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    _write_toy_classification_dataset(dataset_dir)

    suite = suite_config_from_dict(
        {
            "name": "toy_suite",
            "datasets": [
                {
                    "name": "toy",
                    "loader": "table_splits",
                    "task_type": "classification",
                    "task_names": "label",
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                }
            ],
            "featurizers": [
                {
                    "type": "dummy",
                    "name": "dummy_8",
                    "n_features": 8,
                }
            ],
            "downstream_models": {
                "classification": [
                    {
                        "name": "logistic",
                        "model_type": "logistic_regression",
                        "params": {"max_iter": 100},
                    }
                ],
            },
            "seeds": [13, 17],
        }
    )

    results = run_benchmark_suite(
        suite=suite,
        output_dir=tmp_path / "out",
    )

    assert sorted(results["seed"].tolist()) == [13, 17]


def test_suite_config_rejects_missing_downstream_models(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    _write_toy_classification_dataset(dataset_dir)

    with pytest.raises(ValueError, match="downstream"):
        suite_config_from_dict(
            {
                "name": "bad_suite",
                "datasets": [
                    {
                        "name": "toy",
                        "loader": "table_splits",
                        "task_type": "classification",
                        "task_names": "label",
                        "train_path": str(dataset_dir / "train.csv"),
                        "test_path": str(dataset_dir / "test.csv"),
                    }
                ],
                "featurizers": [
                    {
                        "type": "dummy",
                        "name": "dummy_8",
                        "n_features": 8,
                    }
                ],
                "downstream_models": {},
            }
        )


def test_validate_output_dir_allows_missing_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing"

    validate_output_dir(output_dir, overwrite=False)


def test_validate_output_dir_allows_empty_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty"
    output_dir.mkdir()

    validate_output_dir(output_dir, overwrite=False)


def test_validate_output_dir_rejects_nonempty_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        validate_output_dir(output_dir, overwrite=False)


def test_validate_output_dir_overwrite_removes_existing_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old\n", encoding="utf-8")

    validate_output_dir(output_dir, overwrite=True)

    assert not output_dir.exists()
