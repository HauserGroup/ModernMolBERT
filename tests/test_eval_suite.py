from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.eval.suite import run_benchmark_suite, suite_config_from_dict


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
