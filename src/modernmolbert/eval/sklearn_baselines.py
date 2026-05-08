from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalResult:
    dataset: str
    task: str
    split: str
    model_name: str
    metrics: dict[str, float]
    n_train: int
    n_eval: int


def _positive_class_scores(
    model: object, X: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return continuous scores for binary classification metrics."""

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))  # type: ignore[union-attr]

        if proba.ndim != 2:
            raise ValueError(f"predict_proba should return 2D array, got {proba.shape}")

        if proba.shape[1] == 1:
            return proba[:, 0]

        return proba[:, 1]

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)  # type: ignore[union-attr]

        score = np.asarray(score)

        if score.ndim == 2:
            if score.shape[1] == 1:
                return score[:, 0]

            return score[:, 1]

        return score

    raise TypeError(
        "Classification model must implement either predict_proba or decision_function."
    )


def fit_predict_sklearn(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_eval: NDArray[np.float64],
    task_type: str,
    model_name: str = "ridge_or_logreg",
):

    if task_type == "classification":
        if model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                class_weight="balanced",
                random_state=13,
            )

        else:
            # n_jobs is deprecated in recent sklearn LogisticRegression versions;

            # leave it out unless you are pinned to an older version.

            clf = LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=13,
            )

        clf.fit(X_train, y_train.astype(int))

        y_pred = clf.predict(X_eval)

        y_score = _positive_class_scores(clf, X_eval)

        return y_pred, y_score

    if task_type == "regression":
        if model_name == "rf":
            reg = RandomForestRegressor(
                n_estimators=300,
                n_jobs=-1,
                random_state=13,
            )

        else:
            reg = Ridge(alpha=1.0)

        reg.fit(X_train, y_train.astype(float))

        y_pred = reg.predict(X_eval)

        return y_pred, y_pred

    raise ValueError(f"Unknown task_type: {task_type}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

        # ROC-AUC/AP can fail if only one class is present.
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["average_precision"] = float(
                average_precision_score(y_true, y_score)
            )

        return metrics

    if task_type == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }

    raise ValueError(f"Unknown task_type: {task_type}")
