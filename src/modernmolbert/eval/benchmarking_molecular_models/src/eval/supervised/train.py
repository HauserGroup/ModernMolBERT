import gc
import numpy as np
import logging as log

from sklearn.model_selection import GridSearchCV
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
        y_model = np.nan_to_num(y_arr, nan=0)
    else:
        scorer = get_sklearn_scorer("roc_auc")
        y_model = np.nan_to_num(y_arr, nan=0).ravel()
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
