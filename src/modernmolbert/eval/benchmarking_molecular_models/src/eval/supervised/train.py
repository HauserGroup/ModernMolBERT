import gc
import numpy as np
import logging as log

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    EmbeddedDataset,
    HeadResult,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.common.utils import (
    get_test_data,
    get_train_data,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.const import (
    CV_SPLITS,
    N_JOBS,
    VERBOSITY,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.eval_metrics import (
    multioutput_auroc_score,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.models import (
    FiniteLabelMultiOutputClassifier,
    get_clf_models,
    get_reg_models,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.utils import (
    get_sklearn_scorer,
)


def _grid_n_jobs(pipeline, outer_n_jobs: int) -> int:
    # If any pipeline step already parallelizes internally (RF, KNN with n_jobs=-1),
    # joblib serializes the inner parallelism when GridSearchCV also uses multiple
    # workers — net result is single-threaded estimators with no fold-level speedup.
    # Yield all cores to the estimator instead.
    for _, step in pipeline.steps:
        if getattr(step, "n_jobs", 1) not in (1, None):
            return 1
    return outer_n_jobs


def fit_model(
    X: np.ndarray, y: np.ndarray, task: str, metric_name: str, model_head: str, memory_weight: int
):
    y_arr = np.asarray(y)
    is_multioutput = y_arr.ndim == 2 and y_arr.shape[1] > 1
    has_missing_labels = bool(np.isnan(y_arr).any()) if y_arr.dtype.kind in {"f", "c"} else False

    if task == "classification" and is_multioutput and has_missing_labels:
        models = get_clf_models(1, X.dtype)
        return fit_multioutput_finite_label_model(
            X=X,
            y=y_arr,
            models=models,
            model_head=model_head,
            memory_weight=memory_weight,
        )

    if task == "classification":
        no_outputs = y_arr.shape[1] if is_multioutput else 1
        models = get_clf_models(no_outputs, X.dtype)
    elif task == "regression":
        models = get_reg_models(X.dtype)
    else:
        raise ValueError(f"Unknown task: {task}")

    if is_multioutput:
        log.info("Using multioutput AUROC scorer")
        scorer = make_scorer(multioutput_auroc_score, response_method="predict_proba")
        y_model = y_arr
    else:
        scorer = get_sklearn_scorer("roc_auc")
        y_model = y_arr.ravel()
    del y_arr

    log.info(f"Shapes: X={X.shape}, y={y_model.shape}")

    model = models[model_head]
    outer_n_jobs = max(1, int(N_JOBS / memory_weight))
    grid_n_jobs = _grid_n_jobs(model["model"], outer_n_jobs)
    log.info(f"GridSearchCV n_jobs={grid_n_jobs} (outer={outer_n_jobs}, head={model_head})")

    grid_search = GridSearchCV(
        model["model"],
        model["params"],
        cv=CV_SPLITS,
        scoring=scorer,
        n_jobs=grid_n_jobs,
        verbose=VERBOSITY,
        refit=True,
    )

    try:
        grid_search.fit(X, y_model)
    except ValueError as e:
        log.error(f"Error fitting model {model_head}: {e}")
        if "lbfgs" not in str(e):
            raise e
        log.error("L-BFG-S failed, replacing with SVD")
        if "clf__estimator_solver" in model["params"]:
            model["params"]["clf__estimator__solver"] = ["svd"]
        elif "clf__solver" in model["params"]:
            model["params"]["clf__solver"] = ["svd"]
        else:
            raise ValueError(
                "Model parameters do not contain 'solver' or 'estimator__solver' key, cannot replace with SVD"
            ) from e
        grid_search = GridSearchCV(
            model["model"],
            model["params"],
            cv=CV_SPLITS,
            scoring=scorer,
            n_jobs=grid_n_jobs,
            verbose=VERBOSITY,
            refit=True,
        )
        grid_search.fit(X, y_model)

    result = {
        "model": model_head,
        "model_obj": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }
    del grid_search
    gc.collect()
    return result
    # greater_is_better = scorer._sign > 0
    # # filter out nans
    # res = [x for x in res if not np.isnan(x["best_score"])]
    # if len(res) == 0:
    #     raise ValueError("All models failed to fit")

    # f = max if greater_is_better else min
    # return f(res, key=lambda x: x["best_score"])


def finite_label_multioutput_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean endpoint AUROC over finite labels only."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}")

    scores = []
    for col in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, col])
        if mask.sum() == 0:
            continue
        y_col = y_true[mask, col]
        s_col = y_score[mask, col]
        if np.unique(y_col).size < 2:
            continue
        scores.append(roc_auc_score(y_col, s_col))

    if not scores:
        return np.nan
    return float(np.mean(scores))


def fit_multioutput_finite_label_model(
    X: np.ndarray,
    y: np.ndarray,
    models: dict,
    model_head: str,
    memory_weight: int,
):
    """Manual CV for sparse multi-output classification without NaN imputation."""
    model_spec = models[model_head]
    base_pipeline = model_spec["model"]

    param_grid = list(ParameterGrid(model_spec["params"]))
    if not param_grid:
        param_grid = [{}]

    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    any_positive = np.nansum(np.where(finite, y, 0.0), axis=1) > 0

    def make_splits():
        if np.unique(any_positive).size >= 2:
            cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=0)
            return cv.split(X, any_positive.astype(int))
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=0)
        return cv.split(X)

    best_score = -np.inf
    best_params = None

    for params in param_grid:
        fold_scores = []
        for train_idx, valid_idx in make_splits():
            estimator = clone(base_pipeline)
            estimator.set_params(**params)

            wrapped = FiniteLabelMultiOutputClassifier(estimator)
            wrapped.fit(X[train_idx], y[train_idx])
            y_score = wrapped.predict_proba(X[valid_idx])

            score = finite_label_multioutput_score(y[valid_idx], y_score)
            if np.isfinite(score):
                fold_scores.append(score)

        mean_score = float(np.mean(fold_scores)) if fold_scores else np.nan
        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        best_params = {}
        best_score = np.nan

    final_estimator = clone(base_pipeline)
    final_estimator.set_params(**best_params)
    final_model = FiniteLabelMultiOutputClassifier(final_estimator)
    final_model.fit(X, y)

    return {
        "model": model_head,
        "model_obj": final_model,
        "best_params": best_params,
        "best_score": best_score,
    }


def fit_and_eval_embedding(
    dataset: EmbeddedDataset, metric_name: str, model_head: str, memory_weight: int
) -> HeadResult:
    X_train, y_train = get_train_data(dataset)
    best_model = fit_model(
        X=X_train,
        y=y_train,
        task=dataset.task,
        metric_name=metric_name,
        model_head=model_head,
        memory_weight=memory_weight,
    )
    del X_train, y_train
    X_test, y_test = get_test_data(dataset)
    print(f"Shapes: X_test={X_test.shape}, y_test={y_test.shape}", flush=True)
    if dataset.task == "regression":
        y_pred = best_model["model_obj"].predict(X_test)
    else:
        y_pred = best_model["model_obj"].predict_proba(X_test)

    return HeadResult(
        embedder=dataset.embedder,
        dataset_name=dataset.name,
        y_test_true=y_test,
        y_test_pred=y_pred,
        model=best_model["model"],
        hyperparams=best_model["best_params"],
        cv_score=best_model["best_score"],
    )
