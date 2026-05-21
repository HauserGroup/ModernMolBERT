import math

import numpy as np
import pytest

from modernmolbert.eval.metrics import (
    compute_classification_metrics,
    compute_metrics,
    compute_regression_metrics,
)


def test_classification_metrics_compute_binary_scores() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.9, 0.6, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_classification_metrics_return_nan_for_single_class_eval_split() -> None:
    metrics = compute_classification_metrics(
        y_true=np.array([1, 1, 1]),
        y_pred=np.array([1, 0, 1]),
        y_score=np.array([0.9, 0.4, 0.8]),
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert math.isnan(metrics["balanced_accuracy"])
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])


def test_regression_metrics_compute_standard_scores() -> None:
    metrics = compute_regression_metrics(
        y_true=np.array([1.0, 2.0, 4.0]),
        y_pred=np.array([1.0, 3.0, 2.0]),
    )

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx((5 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(-0.0714285714)


def test_compute_metrics_dispatches_by_task_type() -> None:
    classification = compute_metrics(
        task_type="classification",
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        y_score=np.array([0.2, 0.8]),
    )
    regression = compute_metrics(
        task_type="regression",
        y_true=np.array([1.0, 2.0]),
        y_pred=np.array([1.5, 1.5]),
    )

    assert classification["roc_auc"] == pytest.approx(1.0)
    assert regression["mae"] == pytest.approx(0.5)


def test_compute_metrics_rejects_missing_scores_and_unknown_tasks() -> None:
    with pytest.raises(ValueError, match="require y_score"):
        compute_metrics(
            task_type="classification",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
        )

    with pytest.raises(ValueError, match="Unknown task_type"):
        compute_metrics(
            task_type="ranking",  # type: ignore[arg-type]
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_score=np.array([0.2, 0.8]),
        )
