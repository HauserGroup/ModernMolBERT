from __future__ import annotations

import numpy as np

from modernmolbert.eval.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return compute_classification_metrics(
        y_true=np.asarray(y_true),
        y_pred=(np.asarray(y_score) >= 0.5).astype(int),
        y_score=np.asarray(y_score),
    )["roc_auc"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return compute_regression_metrics(np.asarray(y_true), np.asarray(y_pred))["rmse"]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return compute_regression_metrics(np.asarray(y_true), np.asarray(y_pred))["mae"]
