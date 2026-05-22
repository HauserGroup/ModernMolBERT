import os
import numpy as np
import logging as log

from sklearn.metrics import roc_auc_score

from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    EvaluationResult,
    HeadResult,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.utils import (
    get_sklearn_scorer,
)


def evaluate_tdc(y_pred: np.ndarray, dataset_name: str) -> tuple[str, float]:
    """
    Returns:
        Tuple[str, float]: Tuple containing metric name and value
    """
    grp_path = os.path.join(os.getcwd(), "data/cache")
    os.makedirs(grp_path, exist_ok=True)
    from tdc.benchmark_group import admet_group

    group = admet_group(path=grp_path)
    benchmark = group.get(dataset_name)

    predictions = {benchmark["name"]: y_pred}

    metric_name, metric_value = list(list(group.evaluate(predictions).values())[0].items())[0]
    return metric_name, metric_value


def evaluate_ogb(y_pred: np.ndarray, y_test: np.ndarray, dataset_name: str) -> tuple[str, float]:
    """
    Returns:
        Tuple[str, float]: Tuple containing metric name and value
    """
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator(name=dataset_name)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    input_dict = {"y_true": y_test, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return list(result_dict.items())[0]


def evaluate_sklearn(y_pred: np.ndarray, y_test: np.ndarray, metric_name: str) -> tuple[str, float]:
    scorer = get_sklearn_scorer(metric_name)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    metric_value = scorer._score_func(y_test, y_pred)
    return metric_name, metric_value


def log_predictions(data: HeadResult, pred_directory: str):
    print(f"Logging predictions for {data.dataset_name} dataset")
    base_path = os.path.join(
        os.getcwd(), pred_directory, data.dataset_name, data.embedder, data.model
    )
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # Legacy: raw predict_proba output (shape varies by task).
    try:
        np.save(base_path + ".npy", data.y_test_pred)
    except ValueError:
        np.save(base_path + ".npy", _object_array(data.y_test_pred))

    # ROC-ready: y_true and y_score (positive-class probability) in one file.
    # Mirrors the extraction logic in get_skfp_roc_auc.
    y_true = np.asarray(data.y_test_true)
    n_outputs = 1 if y_true.ndim == 1 else y_true.shape[1]
    y_score = _normalize_auc_scores(
        data.y_test_pred,
        n_outputs=n_outputs,
        n_samples=y_true.shape[0],
    )

    np.savez(base_path + ".npz", y_true=y_true, y_score=y_score)
    print(f"Saving predictions to {base_path}.npy / .npz")


def evaluate_tdc_safe(
    y_pred: np.ndarray, y_test: np.ndarray, dataset_name: str, fallback_metric: str
) -> tuple[str, float]:
    """
    Safely evaluates TDC datasets, falling back to a specified metric if the dataset is not found.

    Returns:
        Tuple[str, float]: Tuple containing metric name and value
    """
    try:
        return evaluate_tdc(y_pred, dataset_name)
    except Exception as e:
        log.error(
            f"TDC failed for dataset {dataset_name}. Falling back to {fallback_metric}. Cause: {e}"
        )
        return evaluate_sklearn(y_pred, y_test, fallback_metric)


def _object_array(value):
    """Return a pickle-saveable object array without forcing nested shapes."""
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, list | tuple):
        arr = np.empty(len(value), dtype=object)
        arr[:] = list(value)
        return arr

    return np.asarray(value, dtype=object)


def _positive_column(proba) -> np.ndarray:
    """Extract one output's positive-class score from probabilities or scores."""
    try:
        arr = np.asarray(proba, dtype=float)
    except (TypeError, ValueError):
        if isinstance(proba, np.ndarray) and proba.dtype == object and proba.ndim == 1:
            arr = np.vstack([np.asarray(row, dtype=float) for row in proba])
        else:
            raise

    if arr.ndim == 1:
        return arr

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D per-output scores, got shape {arr.shape}")

    if arr.shape[1] == 0:
        raise ValueError("Cannot extract positive-class score from zero-column probabilities")

    # Some CV folds train a classifier that saw only one class for an output.
    # The single column is constant, so AUROC columns with two y_true classes
    # will score as uninformative and single-class y_true columns are skipped.
    return arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]


def _stack_per_output_scores(scores) -> np.ndarray:
    return np.column_stack([_positive_column(score) for score in scores])


def _normalize_auc_scores(
    y_score,
    *,
    n_outputs: int | None,
    n_samples: int | None,
) -> np.ndarray:
    """Normalize sklearn probability outputs to ROC-ready scores.

    MultiOutputClassifier and sklearn's scorer plumbing may provide raw scores
    as a list, tuple, object ndarray, raw 3D array, transposed 2D array, or
    already-compressed 2D matrix. This returns positive-class scores shaped as
    (n_samples, n_outputs) for multioutput targets and (n_samples,) for binary
    targets.
    """
    if isinstance(y_score, list | tuple) and n_outputs is not None and len(y_score) == n_outputs:
        return _stack_per_output_scores(y_score)

    if (
        isinstance(y_score, np.ndarray)
        and y_score.dtype == object
        and n_outputs is not None
        and y_score.ndim >= 1
        and y_score.shape[0] == n_outputs
    ):
        return _stack_per_output_scores([y_score[idx] for idx in range(n_outputs)])

    try:
        arr = np.asarray(y_score, dtype=float)
    except (TypeError, ValueError):
        if (
            n_outputs is not None
            and isinstance(y_score, np.ndarray)
            and y_score.ndim >= 1
            and y_score.shape[0] == n_outputs
        ):
            return _stack_per_output_scores([y_score[idx] for idx in range(n_outputs)])
        raise

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        if n_outputs == 1 and arr.shape[1] != 1:
            return _positive_column(arr)
        if n_outputs is not None and arr.shape[1] == n_outputs:
            return arr
        if (
            n_outputs is not None
            and arr.shape[0] == n_outputs
            and (n_samples is None or arr.shape[1] == n_samples)
        ):
            return arr.T
        return arr

    if arr.ndim == 3:
        positive = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
        if (
            n_outputs is not None
            and positive.shape[0] == n_outputs
            and (n_samples is None or positive.shape[1] == n_samples)
        ):
            return positive.T
        if (
            n_outputs is not None
            and positive.shape[1] == n_outputs
            and (n_samples is None or positive.shape[0] == n_samples)
        ):
            return positive
        return positive

    raise ValueError(f"Cannot normalize AUROC scores with shape {arr.shape}")


def get_skfp_roc_auc(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    y_test = np.asarray(y_test)
    n_outputs = 1 if y_test.ndim == 1 else y_test.shape[1]
    y_pred = _normalize_auc_scores(
        y_pred,
        n_outputs=n_outputs,
        n_samples=y_test.shape[0],
    )

    if np.isnan(np.min(y_test)):
        return multioutput_auroc_score(y_test, y_pred)
    try:
        return roc_auc_score(y_test, y_pred)
    except Exception:
        return multioutput_auroc_score(y_test, y_pred)


def multioutput_auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    n_outputs = 1 if y_true.ndim == 1 else y_true.shape[1]
    y_score = _normalize_auc_scores(
        y_score,
        n_outputs=n_outputs,
        n_samples=y_true.shape[0],
    )

    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_score))

    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    scores: list[float] = []
    for col in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, col])
        if mask.sum() == 0:
            continue
        y_col = y_true[mask, col].astype(int)
        if len(np.unique(y_col)) < 2:
            continue
        scores.append(float(roc_auc_score(y_col, y_score[mask, col])))

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def evaluate(data: HeadResult, dataset_config, pred_directory: str) -> EvaluationResult:
    y_pred = data.y_test_pred
    y_test = data.y_test_true

    log_predictions(data, pred_directory)

    metric_name = dataset_config.metric
    metric_value = get_skfp_roc_auc(y_pred, y_test)

    return EvaluationResult(
        embedder=data.embedder,
        metric_name=metric_name,
        metric_value=metric_value,
        model=data.model,
        hyperparams=data.hyperparams,
        cv_metric_value=data.cv_score,
    )
