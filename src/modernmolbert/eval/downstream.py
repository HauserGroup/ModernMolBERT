from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DownstreamPrediction:
    y_pred: np.ndarray
    y_score: np.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class FrozenDownstreamConfig:
    """Configuration for the shared downstream learner."""

    classification_max_iter: int = 5000
    classification_class_weight: str | None = "balanced"
    regression_alpha: float = 1.0
    use_ridge_cv: bool = False
    ridge_cv_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)
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

    # ... [Classification block remains largely the same] ...

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
        model.fit(X_train, y_train.astype(float))

        # FIX 3: Wrap in np.asarray to clear the "tuple" type error
        raw_preds = model.predict(X_eval)
        y_pred = np.asarray(raw_preds).astype(np.float64)

        metadata: dict[str, object] = {
            "downstream_model": "ridge_cv" if config.use_ridge_cv else "ridge",
            "standardize": config.standardize,
        }

        if config.use_ridge_cv:
            # FIX 1 & 2: Use explicit type narrowing for the estimator
            if isinstance(model, Pipeline):
                final_estimator = model[-1]
            else:
                final_estimator = model

            # Tell Pylance this definitely has alpha_
            if isinstance(final_estimator, RidgeCV):
                metadata["alpha"] = float(final_estimator.alpha_)
            else:
                # Fallback for type safety, though logic dictates this is RidgeCV
                metadata["alpha"] = None
        else:
            metadata["alpha"] = float(config.regression_alpha)

        return DownstreamPrediction(
            y_pred=y_pred,
            y_score=None,
            metadata=metadata,
        )

    raise ValueError(f"Unknown task_type: {task_type!r}")
