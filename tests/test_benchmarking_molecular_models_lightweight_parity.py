from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.lightweight_parity import (
    KNN_CLF_GRID,
    RF_CLF_GRID,
    RIDGE_CLF_GRID,
    lightweight_positive_class_scores,
    lightweight_roc_auc,
    normalize_lightweight_parity_heads,
    run_lightweight_parity_suite,
)
from modernmolbert.eval.suite import suite_config_from_dict


def test_lightweight_parity_head_expansion_and_grids() -> None:
    assert normalize_lightweight_parity_heads(["auto"]) == ["rf", "ridge", "knn"]
    assert normalize_lightweight_parity_heads(["ridge"]) == ["ridge"]
    assert RF_CLF_GRID["clf__n_estimators"] == [500]
    assert RF_CLF_GRID["clf__criterion"] == ["entropy"]
    assert RF_CLF_GRID["clf__min_samples_split"].tolist() == [2, 4, 6, 8, 10]
    assert len(RIDGE_CLF_GRID["clf__C"]) == 10
    assert KNN_CLF_GRID["clf__n_neighbors"].tolist() == [1, 3, 5, 7, 9]

    with pytest.raises(ValueError, match="Unsupported lightweight parity head"):
        normalize_lightweight_parity_heads(["logreg"])


def test_lightweight_positive_class_scores_binary() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )

    scores = lightweight_positive_class_scores(y_pred, y_true)

    assert scores.tolist() == [0.1, 0.8, 0.3, 0.9]
    assert lightweight_roc_auc(y_true=y_true, y_score=scores) == 1.0


def test_lightweight_positive_class_scores_multioutput_with_nan_labels() -> None:
    y_true = np.array(
        [
            [0, 1],
            [1, np.nan],
            [0, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y_pred = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
            [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]],
        ]
    )

    scores = lightweight_positive_class_scores(y_pred, y_true)

    assert scores.shape == (4, 2)
    assert scores[:, 0].tolist() == [0.1, 0.9, 0.2, 0.8]
    assert scores[:, 1].tolist() == [0.8, 0.6, 0.3, 0.9]
    assert lightweight_roc_auc(y_true=y_true, y_score=scores) == 1.0


def test_run_lightweight_parity_suite_binary_classification(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    train = pd.DataFrame(
        {
            "smiles": [f"C{'C' * i}" for i in range(20)],
            "label": [0, 1] * 10,
        }
    )
    test = pd.DataFrame(
        {
            "smiles": [f"N{'C' * i}" for i in range(10)],
            "label": [0, 1] * 5,
        }
    )
    train.to_csv(dataset_dir / "train.csv", index=False)
    test.to_csv(dataset_dir / "test.csv", index=False)

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
            "featurizers": [{"type": "dummy", "name": "dummy_8", "n_features": 8}],
            "downstream_models": {
                "classification": [
                    {
                        "name": "ridge",
                        "model_type": "lightweight_parity_classifier",
                    }
                ]
            },
        }
    )

    results = run_lightweight_parity_suite(
        suite=suite,
        output_dir=tmp_path / "out",
        heads=["ridge"],
    )

    assert len(results) == 1
    assert results.loc[0, "dataset"] == "toy"
    assert results.loc[0, "task"] == "__all__"
    assert results.loc[0, "downstream_name"] == "ridge"
    assert np.isfinite(results.loc[0, "roc_auc"])
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_run_lightweight_parity_suite_rejects_regression(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "reg"
    dataset_dir.mkdir()
    train = pd.DataFrame({"smiles": ["CC", "CCC", "CCCC", "CCCCC", "CCCCCC"], "y": range(5)})
    test = pd.DataFrame({"smiles": ["NN", "NNN"], "y": [1.0, 2.0]})
    train.to_csv(dataset_dir / "train.csv", index=False)
    test.to_csv(dataset_dir / "test.csv", index=False)
    suite = suite_config_from_dict(
        {
            "name": "reg_suite",
            "datasets": [
                {
                    "name": "reg",
                    "loader": "table_splits",
                    "task_type": "regression",
                    "task_names": ["y"],
                    "train_path": str(dataset_dir / "train.csv"),
                    "test_path": str(dataset_dir / "test.csv"),
                    "smiles_column": "smiles",
                }
            ],
            "featurizers": [{"type": "dummy", "name": "dummy_8", "n_features": 8}],
            "downstream_models": {
                "classification": [
                    {
                        "name": "ridge",
                        "model_type": "lightweight_parity_classifier",
                    }
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="only supports classification datasets"):
        run_lightweight_parity_suite(
            suite=suite,
            output_dir=tmp_path / "out",
            heads=["ridge"],
        )
