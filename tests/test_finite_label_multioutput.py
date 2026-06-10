import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modernmolbert.eval.benchmarking_molecular_models.supervised.models import (
    ConstantProbabilityClassifier,
    FiniteLabelMultiOutputClassifier,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.train import (
    finite_label_multioutput_score,
    fit_model,
    fit_multioutput_finite_label_model,
)


class RecordingEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        p1 = np.full(n, 0.5, dtype=float)
        return np.column_stack([1.0 - p1, p1])


def _toy_multitask_labels() -> np.ndarray:
    return np.array(
        [
            [1, np.nan, 0],
            [0, 1, np.nan],
            [np.nan, 0, 1],
            [1, np.nan, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )


def _toy_X(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 6)).astype(np.float32)


def _tiny_models_for_sparse_cv() -> dict:
    return {
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=1000,
                        ),
                    ),
                ]
            ),
            "params": {"clf__C": [1.0]},
        },
        "rf": {
            "model": Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=25,
                            min_samples_split=2,
                            random_state=0,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            "params": {"clf__max_depth": [None]},
        },
    }


def test_endpoint_wise_fitting_uses_only_finite_rows() -> None:
    X = _toy_X(5)
    y = _toy_multitask_labels()

    wrapped = FiniteLabelMultiOutputClassifier(RecordingEstimator())
    wrapped.fit(X, y)

    # endpoint 0: rows 0,1,3,4
    call0 = wrapped.estimators_[0].fit_calls[0]
    assert call0[0].shape[0] == 4
    assert call0[1].tolist() == [1.0, 0.0, 1.0, 0.0]

    # endpoint 1: rows 1,2,4
    call1 = wrapped.estimators_[1].fit_calls[0]
    assert call1[0].shape[0] == 3
    assert call1[1].tolist() == [1.0, 0.0, 1.0]

    # endpoint 2: rows 0,2,3,4
    call2 = wrapped.estimators_[2].fit_calls[0]
    assert call2[0].shape[0] == 4
    assert call2[1].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_single_class_endpoint_uses_constant_classifier() -> None:
    X = _toy_X(5)
    y = np.array(
        [
            [0, np.nan],
            [0, 1],
            [0, 0],
            [0, np.nan],
            [0, 1],
        ],
        dtype=float,
    )

    wrapped = FiniteLabelMultiOutputClassifier(RecordingEstimator())
    wrapped.fit(X, y)

    assert isinstance(wrapped.estimators_[0], ConstantProbabilityClassifier)
    proba = wrapped.predict_proba(X)
    assert proba.shape == (5, 2)


def test_prediction_shape_is_stable() -> None:
    X = _toy_X(5)
    y = _toy_multitask_labels()

    wrapped = FiniteLabelMultiOutputClassifier(RecordingEstimator())
    wrapped.fit(X, y)

    y_score = wrapped.predict_proba(X)
    assert y_score.shape == (5, 3)


def test_score_masks_nans_endpoint_wise() -> None:
    y_true = np.array(
        [
            [1, np.nan],
            [0, 1],
            [1, 0],
            [0, np.nan],
        ],
        dtype=float,
    )
    y_score = np.array(
        [
            [0.9, 0.2],
            [0.1, 0.8],
            [0.8, 0.2],
            [0.2, 0.4],
        ],
        dtype=float,
    )

    score = finite_label_multioutput_score(y_true, y_score)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_sparse_tox21_muv_like_matrix_completes_for_ridge_and_rf() -> None:
    rng = np.random.default_rng(42)
    n_samples = 60
    n_outputs = 6

    X = rng.normal(size=(n_samples, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=(n_samples, n_outputs)).astype(float)

    # Make matrix sparse with many missing labels.
    missing_mask = rng.random(size=y.shape) < 0.7
    y[missing_mask] = np.nan

    # Ensure at least one endpoint has enough finite labels of both classes.
    y[:20, 0] = np.array([0, 1] * 10, dtype=float)
    y[:20, 1] = np.array([1, 0] * 10, dtype=float)

    models = _tiny_models_for_sparse_cv()

    ridge = fit_multioutput_finite_label_model(
        X=X,
        y=y,
        models=models,
        model_head="ridge",
        memory_weight=1,
    )
    rf = fit_multioutput_finite_label_model(
        X=X,
        y=y,
        models=models,
        model_head="rf",
        memory_weight=1,
    )

    ridge_pred = ridge["model_obj"].predict_proba(X)
    rf_pred = rf["model_obj"].predict_proba(X)

    assert ridge_pred.shape == (n_samples, n_outputs)
    assert rf_pred.shape == (n_samples, n_outputs)


def test_no_forbidden_nan_to_zero_in_classification_fit_path() -> None:
    source = inspect.getsource(fit_model)
    assert "np.nan_to_num(y_arr, nan=0)" not in source
