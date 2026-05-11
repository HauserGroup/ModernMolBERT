from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


TaskType = Literal["classification", "regression"]

ClassificationModelType = Literal[
    "auto",
    "logistic_regression",
    "random_forest_classifier",
]

RegressionModelType = Literal[
    "auto",
    "ridge",
    "ridge_cv",
    "random_forest_regressor",
]


@dataclass(frozen=True)
class DownstreamPrediction:
    y_pred: np.ndarray
    y_score: np.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class FrozenDownstreamConfig:
    """Configuration for one downstream model used on frozen features."""

    model_type: str = "auto"
    params: dict[str, object] | None = None
    random_state: int = 13
    standardize: bool = True


def _params(config: FrozenDownstreamConfig) -> dict[str, Any]:
    return {} if config.params is None else dict(config.params)


def _maybe_standardize(estimator, *, standardize: bool, with_mean: bool = True):
    if not standardize:
        return estimator

    return make_pipeline(StandardScaler(with_mean=with_mean), estimator)


def _final_estimator(model):
    if isinstance(model, Pipeline):
        return model[-1]
    return model


def make_classification_estimator(config: FrozenDownstreamConfig):
    """Build a sklearn classifier from config."""

    params = _params(config)
    model_type = config.model_type
    if model_type == "auto":
        model_type = "logistic_regression"

    if model_type == "logistic_regression":
        max_iter = int(params.get("max_iter", 5000))
        class_weight = params.get("class_weight", "balanced")
        C = float(params.get("C", 1.0))
        solver = str(params.get("solver", "lbfgs"))

        estimator = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,  # type: ignore[arg-type]
            C=C,
            solver=solver,  # type: ignore[arg-type]
            random_state=config.random_state,
        )

        model = _maybe_standardize(
            estimator,
            standardize=config.standardize,
            with_mean=True,
        )

        metadata = {
            "downstream_model": "logistic_regression",
            "standardize": config.standardize,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "C": C,
            "solver": solver,
            "random_state": config.random_state,
        }

        return model, metadata

    if model_type == "random_forest_classifier":
        from sklearn.ensemble import RandomForestClassifier

        n_estimators = int(params.get("n_estimators", 500))
        max_depth = params.get("max_depth", None)
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        class_weight = params.get("class_weight", "balanced")
        n_jobs = int(params.get("n_jobs", -1))

        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,  # type: ignore[arg-type]
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,  # type: ignore[arg-type]
            random_state=config.random_state,
            n_jobs=n_jobs,
        )

        metadata = {
            "downstream_model": "random_forest_classifier",
            "standardize": False,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "n_jobs": n_jobs,
            "random_state": config.random_state,
        }

        return estimator, metadata

    raise ValueError(f"Unsupported classification model_type: {config.model_type!r}")


def make_regression_estimator(config: FrozenDownstreamConfig):
    """Build a sklearn regressor from config."""

    params = _params(config)
    model_type = config.model_type
    if model_type == "auto":
        model_type = "ridge"

    if model_type == "ridge":
        alpha = float(params.get("alpha", 1.0))

        estimator = Ridge(alpha=alpha)

        model = _maybe_standardize(
            estimator,
            standardize=config.standardize,
            with_mean=True,
        )

        metadata = {
            "downstream_model": "ridge",
            "standardize": config.standardize,
            "alpha": alpha,
        }

        return model, metadata

    if model_type == "ridge_cv":
        raw_alphas = params.get("alphas", (0.01, 0.1, 1.0, 10.0, 100.0))
        alphas = np.asarray(raw_alphas, dtype=float)

        estimator = RidgeCV(alphas=alphas)

        model = _maybe_standardize(
            estimator,
            standardize=config.standardize,
            with_mean=True,
        )

        metadata = {
            "downstream_model": "ridge_cv",
            "standardize": config.standardize,
            "candidate_alphas": [float(x) for x in alphas],
        }

        return model, metadata

    if model_type == "random_forest_regressor":
        from sklearn.ensemble import RandomForestRegressor

        n_estimators = int(params.get("n_estimators", 500))
        max_depth = params.get("max_depth", None)
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        n_jobs = int(params.get("n_jobs", -1))

        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,  # type: ignore[arg-type]
            min_samples_leaf=min_samples_leaf,
            random_state=config.random_state,
            n_jobs=n_jobs,
        )

        metadata = {
            "downstream_model": "random_forest_regressor",
            "standardize": False,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_jobs": n_jobs,
            "random_state": config.random_state,
        }

        return estimator, metadata

    raise ValueError(f"Unsupported regression model_type: {config.model_type!r}")


def fit_predict_downstream(
    *,
    task_type: TaskType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    config: FrozenDownstreamConfig,
) -> DownstreamPrediction:
    """Fit one configured downstream learner on frozen features."""

    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")

    if X_eval.ndim != 2:
        raise ValueError(f"X_eval must be 2D, got shape {X_eval.shape}")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train length mismatch: {len(X_train)} != {len(y_train)}"
        )

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

        model, metadata = make_classification_estimator(config)
        model.fit(X_train, y_train_int)

        y_pred = np.asarray(model.predict(X_eval)).astype(int)

        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X_eval))
            if proba.shape[1] != 2:
                raise ValueError(
                    "Binary classifier predict_proba should have two columns, "
                    f"got shape {proba.shape}"
                )
            y_score = proba[:, 1].astype(float)
        elif (decision_fn := getattr(model, "decision_function", None)) is not None:
            y_score = np.asarray(decision_fn(X_eval)).astype(float)
        else:
            y_score = None

        return DownstreamPrediction(
            y_pred=y_pred,
            y_score=y_score,
            metadata=metadata,
        )

    if task_type == "regression":
        y_train_float = np.asarray(y_train).astype(float)

        model, metadata = make_regression_estimator(config)
        model.fit(X_train, y_train_float)

        final_estimator = _final_estimator(model)
        if isinstance(final_estimator, RidgeCV):
            metadata = dict(metadata)
            metadata["alpha"] = float(final_estimator.alpha_)

        y_pred = np.asarray(model.predict(X_eval)).astype(float)

        return DownstreamPrediction(
            y_pred=y_pred,
            y_score=None,
            metadata=metadata,
        )

    raise ValueError(f"Unknown task_type: {task_type!r}")
