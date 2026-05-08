from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DownstreamPrediction:
    y_pred: np.ndarray
    y_score: np.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class FrozenDownstreamConfig:
    classification_max_iter: int = 5000
    classification_class_weight: str | None = "balanced"
    regression_alpha: float = 1.0
    use_ridge_cv: bool = False
    ridge_cv_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    random_state: int = 13
    standardize: bool = True


def fit_predict_downstream(
    *,
    task_type: TaskType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    config: FrozenDownstreamConfig,
) -> DownstreamPrediction:
    """Fit the fixed downstream learner for the frozen-representation benchmark.

    Primary benchmark policy:
      - classification: LogisticRegression
      - regression: Ridge or RidgeCV
    """

    if task_type == "classification":
        y_train_int = np.asarray(y_train).astype(int)

        classes = np.unique(y_train_int)
        if len(classes) < 2:
            raise ValueError("Classification training labels contain only one class")
        if len(classes) != 2:
            raise ValueError(
                "Only binary classification is currently supported. "
                f"Found classes: {classes.tolist()}"
            )

        estimator = LogisticRegression(
            max_iter=config.classification_max_iter,
            class_weight=config.classification_class_weight,
            random_state=config.random_state,
        )

        model = (
            make_pipeline(StandardScaler(), estimator)
            if config.standardize
            else estimator
        )

        model.fit(X_train, y_train_int)

        y_pred = np.asarray(model.predict(X_eval)).astype(int)
        y_score = np.asarray(model.predict_proba(X_eval))[:, 1].astype(float)

        return DownstreamPrediction(
            y_pred=y_pred,
            y_score=y_score,
            metadata={
                "downstream_model": "logistic_regression",
                "standardize": config.standardize,
                "class_weight": config.classification_class_weight,
            },
        )

    if task_type == "regression":
        if config.use_ridge_cv:
            estimator = RidgeCV(alphas=np.asarray(config.ridge_cv_alphas))
        else:
            estimator = Ridge(alpha=config.regression_alpha)

        model = (
            make_pipeline(StandardScaler(), estimator)
            if config.standardize
            else estimator
        )

        model.fit(X_train, np.asarray(y_train).astype(float))

        y_pred = np.asarray(model.predict(X_eval)).astype(float)

        metadata: dict[str, object] = {
            "downstream_model": "ridge_cv" if config.use_ridge_cv else "ridge",
            "standardize": config.standardize,
        }

        if config.use_ridge_cv:
            if isinstance(model, Pipeline):
                final_estimator = model[-1]
            else:
                final_estimator = model

            if isinstance(final_estimator, RidgeCV):
                metadata["alpha"] = float(final_estimator.alpha_)
            else:
                metadata["alpha"] = None
        else:
            metadata["alpha"] = float(config.regression_alpha)

        return DownstreamPrediction(
            y_pred=y_pred,
            y_score=None,
            metadata=metadata,
        )

    raise ValueError(f"Unknown task_type: {task_type!r}")
