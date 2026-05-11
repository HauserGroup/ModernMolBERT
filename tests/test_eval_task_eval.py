from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.downstream import FrozenDownstreamConfig
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.task_eval import TaskResult, TaskSkip, align_task_data, evaluate_single_task


def test_align_task_data_filters_missing_labels_and_invalid_features() -> None:
    train = pd.DataFrame(
        {
            "smiles": ["a", "b", "c", "d"],
            "label": [0.0, np.nan, 1.0, 1.0],
        }
    )
    eval_frame = pd.DataFrame(
        {
            "smiles": ["e", "f", "g"],
            "label": [0.0, 1.0, np.nan],
        }
    )

    train_features = FeatureBatch(
        X=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        valid_mask=np.array([True, False, True, True]),
    )
    eval_features = FeatureBatch(
        X=np.array([[0.0], [1.0]], dtype=np.float32),
        valid_mask=np.array([True, True, False]),
    )

    aligned = align_task_data(
        task="label",
        train_frame=train,
        eval_frame=eval_frame,
        train_features=train_features,
        eval_features=eval_features,
    )

    assert aligned.X_train.shape == (3, 1)
    assert aligned.y_train.tolist() == [0.0, 1.0, 1.0]
    assert aligned.X_eval.shape == (2, 1)
    assert aligned.y_eval.tolist() == [0.0, 1.0]


def test_evaluate_single_task_skips_one_class_train() -> None:
    train = pd.DataFrame(
        {
            "smiles": ["a", "b", "c"],
            "label": [1.0, 1.0, 1.0],
        }
    )
    eval_frame = pd.DataFrame(
        {
            "smiles": ["d", "e"],
            "label": [0.0, 1.0],
        }
    )

    train_features = FeatureBatch(
        X=np.ones((3, 2), dtype=np.float32),
        valid_mask=np.array([True, True, True]),
    )
    eval_features = FeatureBatch(
        X=np.ones((2, 2), dtype=np.float32),
        valid_mask=np.array([True, True]),
    )

    result, skip = evaluate_single_task(
        dataset_name="toy",
        task="label",
        task_type="classification",
        eval_split="test",
        featurizer_name="dummy",
        train_frame=train,
        eval_frame=eval_frame,
        train_features=train_features,
        eval_features=eval_features,
        downstream_config=FrozenDownstreamConfig(),
    )

    assert result is None
    assert skip is not None
    assert skip.reason == "classification_train_has_single_class"


def test_evaluate_single_task_classification_success() -> None:
    train = pd.DataFrame(
        {
            "smiles": ["a", "b", "c", "d"],
            "label": [0.0, 0.0, 1.0, 1.0],
        }
    )
    eval_frame = pd.DataFrame(
        {
            "smiles": ["e", "f"],
            "label": [0.0, 1.0],
        }
    )

    train_features = FeatureBatch(
        X=np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
            ],
            dtype=np.float32,
        ),
        valid_mask=np.array([True, True, True, True]),
    )
    eval_features = FeatureBatch(
        X=np.array([[0.05, 0.0], [1.05, 1.0]], dtype=np.float32),
        valid_mask=np.array([True, True]),
    )

    result, skip = evaluate_single_task(
        dataset_name="toy",
        task="label",
        task_type="classification",
        eval_split="test",
        featurizer_name="dummy",
        train_frame=train,
        eval_frame=eval_frame,
        train_features=train_features,
        eval_features=eval_features,
        downstream_config=FrozenDownstreamConfig(),
    )

    assert skip is None
    assert result is not None
    assert result.dataset == "toy"
    assert result.task == "label"
    assert "accuracy" in result.metrics
    assert result.downstream_metadata["downstream_model"] == "logistic_regression"


# ---------------------------------------------------------------------------
# TaskResult direct construction
# ---------------------------------------------------------------------------


def _make_task_result(**overrides) -> TaskResult:
    defaults = dict(
        dataset="bbbp",
        task="p_np",
        task_type="classification",
        split="test",
        featurizer="dummy",
        metrics={"roc_auc": 0.85, "accuracy": 0.80},
        n_train=100,
        n_eval=20,
        n_train_total=110,
        n_eval_total=22,
        n_train_feature_valid=98,
        n_eval_feature_valid=20,
    )
    defaults.update(overrides)
    return TaskResult(**defaults)  # type: ignore[call-overload]


def _make_task_skip(**overrides) -> TaskSkip:
    defaults = dict(
        dataset="bbbp",
        task="p_np",
        split="test",
        reason="classification_train_has_single_class",
        n_train_label_valid_rows=10,
        n_eval_label_valid_rows=5,
        n_train_feature_valid_rows=10,
        n_eval_feature_valid_rows=5,
    )
    defaults.update(overrides)
    return TaskSkip(**defaults)  # type: ignore[call-overload]


def test_task_result_fields() -> None:
    result = _make_task_result()

    assert result.dataset == "bbbp"
    assert result.task == "p_np"
    assert result.task_type == "classification"
    assert result.split == "test"
    assert result.featurizer == "dummy"
    assert result.metrics["roc_auc"] == 0.85
    assert result.n_train == 100
    assert result.n_eval == 20
    assert result.n_train_total == 110
    assert result.n_eval_total == 22
    assert result.n_train_feature_valid == 98
    assert result.n_eval_feature_valid == 20

    d = asdict(result)
    assert d["metrics"] == {"roc_auc": 0.85, "accuracy": 0.80}
    assert "downstream_metadata" in d
    assert "feature_metadata" in d


def test_task_result_default_metadata_empty() -> None:
    result = _make_task_result()

    assert result.downstream_metadata == {}
    assert result.feature_metadata == {}


def test_task_result_stores_downstream_metadata() -> None:
    result = _make_task_result(downstream_metadata={"downstream_model": "ridge", "alpha": 1.0})

    assert result.downstream_metadata["downstream_model"] == "ridge"
    assert result.downstream_metadata["alpha"] == 1.0


def test_task_result_is_frozen() -> None:
    result = _make_task_result()

    with pytest.raises(FrozenInstanceError):
        result.task = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskSkip direct construction
# ---------------------------------------------------------------------------


def test_task_skip_fields() -> None:
    skip = _make_task_skip()

    assert skip.dataset == "bbbp"
    assert skip.task == "p_np"
    assert skip.split == "test"
    assert skip.reason == "classification_train_has_single_class"
    assert skip.n_train_label_valid_rows == 10
    assert skip.n_eval_label_valid_rows == 5
    assert skip.n_train_feature_valid_rows == 10
    assert skip.n_eval_feature_valid_rows == 5

    d = asdict(skip)
    assert d["reason"] == "classification_train_has_single_class"


def test_task_skip_is_frozen() -> None:
    skip = _make_task_skip()

    with pytest.raises(FrozenInstanceError):
        skip.reason = "other"  # type: ignore[misc]


def test_task_skip_reason_preserved() -> None:
    for reason in (
        "no_train_rows_after_label_and_feature_filtering",
        "no_eval_rows_after_label_and_feature_filtering",
        "classification_train_has_single_class",
    ):
        skip = _make_task_skip(reason=reason)
        assert skip.reason == reason
