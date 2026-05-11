import numpy as np
import pytest

from modernmolbert.eval.downstream import FrozenDownstreamConfig, fit_predict_downstream


def test_logistic_regression_downstream() -> None:
    X_train = np.array([[0, 0], [0.1, 0], [1, 1], [1.1, 1]], dtype=float)
    y_train = np.array([0, 0, 1, 1])
    X_eval = np.array([[0.05, 0], [1.05, 1]], dtype=float)

    pred = fit_predict_downstream(
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        config=FrozenDownstreamConfig(
            model_type="logistic_regression",
            params={"max_iter": 1000},
        ),
    )

    assert pred.y_pred.shape == (2,)
    assert pred.y_score is not None
    assert pred.metadata["downstream_model"] == "logistic_regression"


def test_ridge_downstream() -> None:
    X_train = np.array([[0], [1], [2], [3]], dtype=float)
    y_train = np.array([0, 1, 2, 3], dtype=float)
    X_eval = np.array([[1.5], [2.5]], dtype=float)

    pred = fit_predict_downstream(
        task_type="regression",
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        config=FrozenDownstreamConfig(
            model_type="ridge",
            params={"alpha": 1.0},
        ),
    )

    assert pred.y_pred.shape == (2,)
    assert pred.y_score is None
    assert pred.metadata["downstream_model"] == "ridge"


def test_classification_rejects_single_class_train() -> None:
    with pytest.raises(ValueError, match="one class"):
        fit_predict_downstream(
            task_type="classification",
            X_train=np.ones((3, 2)),
            y_train=np.ones(3),
            X_eval=np.ones((1, 2)),
            config=FrozenDownstreamConfig(),
        )


def test_random_forest_classifier_downstream() -> None:
    X_train = np.array([[0, 0], [0.1, 0], [1, 1], [1.1, 1]], dtype=float)
    y_train = np.array([0, 0, 1, 1])
    X_eval = np.array([[0.05, 0], [1.05, 1]], dtype=float)

    pred = fit_predict_downstream(
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        config=FrozenDownstreamConfig(
            model_type="random_forest_classifier",
            params={"n_estimators": 10, "n_jobs": 1},
        ),
    )

    assert pred.y_pred.shape == (2,)
    assert pred.y_score is not None
    assert pred.metadata["downstream_model"] == "random_forest_classifier"
