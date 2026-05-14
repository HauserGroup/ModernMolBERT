from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modernmolbert.eval.cache import get_or_compute_features
from modernmolbert.eval.datasets import EvalDataset, load_eval_dataset_from_config
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.registry import make_featurizer_from_config
from modernmolbert.eval.runner import _select_eval_frame
from modernmolbert.eval.suite import BenchmarkSuiteConfig


LightweightHead = Literal["rf", "ridge", "knn"]

AVAILABLE_LIGHTWEIGHT_PARITY_HEADS: tuple[LightweightHead, ...] = ("rf", "ridge", "knn")

RF_CLF_GRID = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["entropy"],
}

RIDGE_CLF_GRID = {
    "clf__C": 1 / np.logspace(-2, 3, 10),
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
    "clf__max_iter": [5000],
}

RIDGE_MULTIOUTPUT_CLF_GRID = {
    "clf__estimator__C": 1 / np.logspace(-2, 3, 10),
    "clf__estimator__penalty": ["l2"],
    "clf__estimator__solver": ["lbfgs"],
    "clf__estimator__max_iter": [5000],
}

KNN_CLF_GRID = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}


@dataclass(frozen=True)
class LightweightParityResult:
    head: LightweightHead
    estimator: Any
    best_params: dict[str, Any]
    best_score: float
    roc_auc: float
    y_score: np.ndarray


def normalize_lightweight_parity_heads(heads: list[str] | None) -> list[LightweightHead]:
    if not heads or "auto" in heads:
        return list(AVAILABLE_LIGHTWEIGHT_PARITY_HEADS)

    normalized: list[LightweightHead] = []
    for head in heads:
        if head not in AVAILABLE_LIGHTWEIGHT_PARITY_HEADS:
            raise ValueError(
                f"Unsupported lightweight parity head {head!r}. Use auto, rf, ridge, or knn."
            )
        normalized.append(head)  # type: ignore[arg-type]

    return normalized


def get_knn_distance(embeddings_dtype: np.dtype[Any] | type[Any]) -> str | Any:
    if np.issubdtype(embeddings_dtype, np.integer):
        return tanimoto_count_distance

    if np.issubdtype(embeddings_dtype, np.floating):
        return "cosine"

    raise ValueError(
        f"Unsupported embeddings dtype: {embeddings_dtype}. "
        "Expected integer or floating point type."
    )


def tanimoto_count_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    denominator = np.maximum(x, y).sum()
    if denominator == 0:
        return 0.0
    return float(1.0 - np.minimum(x, y).sum() / denominator)


def lightweight_classifier_spec(
    *,
    head: LightweightHead,
    no_outputs: int,
    embeddings_dtype: np.dtype[Any],
) -> tuple[Pipeline, dict[str, Any]]:
    if head == "rf":
        return Pipeline([("clf", RandomForestClassifier(n_jobs=-1))]), dict(RF_CLF_GRID)

    if head == "ridge":
        if no_outputs == 1:
            clf = LogisticRegression(n_jobs=-1)
            params = dict(RIDGE_CLF_GRID)
        else:
            clf = MultiOutputClassifier(LogisticRegression(n_jobs=-1))
            params = dict(RIDGE_MULTIOUTPUT_CLF_GRID)

        return Pipeline([("scaler", StandardScaler()), ("clf", clf)]), params

    if head == "knn":
        clf = KNeighborsClassifier(n_jobs=-1, metric=get_knn_distance(embeddings_dtype))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)]), dict(KNN_CLF_GRID)

    raise ValueError(f"Unsupported lightweight parity head: {head!r}")


def fit_lightweight_classifier(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    head: LightweightHead,
    grid_n_jobs: int = 1,
) -> LightweightParityResult:
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if X_eval.ndim != 2:
        raise ValueError(f"X_eval must be 2D, got shape {X_eval.shape}")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} != {len(y_train)}")

    no_outputs = y_train.shape[1] if y_train.ndim > 1 else 1
    if y_train.ndim == 1:
        y_train_fit = y_train.reshape(-1, 1)
    else:
        y_train_fit = y_train

    if no_outputs > 1:
        scorer = make_scorer(multioutput_auroc_score, response_method="predict_proba")
    else:
        scorer = get_scorer("roc_auc")

    y_train_fit = np.nan_to_num(y_train_fit, nan=0).astype(int)
    model, params = lightweight_classifier_spec(
        head=head,
        no_outputs=no_outputs,
        embeddings_dtype=X_train.dtype,
    )

    grid_search = GridSearchCV(
        model,
        params,
        cv=5,
        scoring=scorer,
        n_jobs=grid_n_jobs,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train_fit)

    raw_score = grid_search.best_estimator_.predict_proba(X_eval)
    y_score = lightweight_positive_class_scores(raw_score, y_eval)
    roc_auc = lightweight_roc_auc(y_true=y_eval, y_score=y_score)

    return LightweightParityResult(
        head=head,
        estimator=grid_search.best_estimator_,
        best_params=dict(grid_search.best_params_),
        best_score=float(grid_search.best_score_),
        roc_auc=roc_auc,
        y_score=y_score,
    )


def lightweight_positive_class_scores(y_pred: Any, y_true: np.ndarray) -> np.ndarray:
    y_score = np.asarray(y_pred)

    if y_true.ndim == 1:
        return np.asarray(y_score[:, 1], dtype=float)

    if y_score.ndim == 3:
        return np.asarray(y_score[:, :, 1].T, dtype=float)

    return np.asarray(y_score, dtype=float)


def lightweight_roc_auc(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_score))

    if np.isnan(np.min(y_true)):
        return float(multioutput_auroc_score(y_true, y_score))

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float(multioutput_auroc_score(y_true, y_score))


def multioutput_auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_score))

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
        return math.nan

    return float(np.mean(scores))


def run_lightweight_parity_suite(
    *,
    suite: BenchmarkSuiteConfig,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    heads: list[str] | None = None,
) -> pd.DataFrame:
    if suite.eval_split != "test":
        raise ValueError("lightweight parity mode only supports eval_split='test'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / "cache"

    selected_heads = normalize_lightweight_parity_heads(heads)
    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for dataset_cfg in suite.datasets:
        dataset = load_eval_dataset_from_config(dataset_cfg.config)
        if dataset.task_type != "classification":
            raise ValueError(
                "lightweight parity mode only supports classification datasets. "
                f"Dataset {dataset.name!r} has task_type={dataset.task_type!r}."
            )

        for featurizer_cfg in suite.featurizers:
            featurizer = make_featurizer_from_config(featurizer_cfg.config)

            train_frame = _merged_train_valid_frame(dataset)
            eval_frame = _select_eval_frame(dataset, "test")
            train_features = _features_for_split(
                dataset=dataset,
                split_name="train_valid",
                frame=train_frame,
                featurizer=featurizer,
                cache_dir=resolved_cache_dir,
                use_cache=suite.use_cache,
                batch_size=suite.batch_size,
            )
            eval_features = _features_for_split(
                dataset=dataset,
                split_name="test",
                frame=eval_frame,
                featurizer=featurizer,
                cache_dir=resolved_cache_dir,
                use_cache=suite.use_cache,
                batch_size=suite.batch_size,
            )

            X_train, y_train = _aligned_features_and_labels(
                frame=train_frame,
                features=train_features,
                task_names=dataset.task_names,
            )
            X_eval, y_eval = _aligned_features_and_labels(
                frame=eval_frame,
                features=eval_features,
                task_names=dataset.task_names,
            )

            if len(dataset.task_names) == 1:
                y_train = y_train.reshape(-1)
                y_eval = y_eval.reshape(-1)

            for head in selected_heads:
                result = fit_lightweight_classifier(
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    head=head,
                )
                rows.append(
                    _result_row(
                        dataset=dataset,
                        featurizer_name=featurizer.name,
                        featurizer_type=str(featurizer_cfg.config.get("type")),
                        head=head,
                        result=result,
                        train_frame=train_frame,
                        eval_frame=eval_frame,
                        train_features=train_features,
                        eval_features=eval_features,
                        n_train=len(X_train),
                        n_eval=len(X_eval),
                        seed=suite.seeds[0] if suite.seeds else None,
                    )
                )
                run_records.append(
                    {
                        "dataset": dataset.name,
                        "dataset_config": _json_safe(dict(dataset_cfg.config)),
                        "featurizer": featurizer.name,
                        "featurizer_config": _json_safe(dict(featurizer_cfg.config)),
                        "downstream_name": head,
                        "downstream_config": {
                            "model_type": "lightweight_parity_classifier",
                            "params": _json_safe(result.best_params),
                        },
                        "seed": suite.seeds[0] if suite.seeds else None,
                        "eval_split": suite.eval_split,
                        "n_task_results": 1,
                        "n_skipped_tasks": 0,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    _write_manifest(
        output_dir=output_dir,
        suite=suite,
        cache_dir=resolved_cache_dir,
        run_records=run_records,
        n_result_rows=len(results),
    )
    return results


def _merged_train_valid_frame(dataset: EvalDataset) -> pd.DataFrame:
    if dataset.valid is None:
        return dataset.train
    return pd.concat([dataset.train, dataset.valid], ignore_index=True)


def _features_for_split(
    *,
    dataset: EvalDataset,
    split_name: str,
    frame: pd.DataFrame,
    featurizer: Any,
    cache_dir: Path,
    use_cache: bool,
    batch_size: int,
) -> FeatureBatch:
    return get_or_compute_features(
        dataset_name=dataset.name,
        split_name=split_name,
        frame=frame,
        smiles_column=dataset.smiles_column,
        featurizer=featurizer,
        cache_dir=cache_dir,
        use_cache=use_cache,
        batch_size=batch_size,
    )


def _aligned_features_and_labels(
    *,
    frame: pd.DataFrame,
    features: FeatureBatch,
    task_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    labels = frame.loc[features.valid_mask, task_names].to_numpy(dtype=float)
    return features.X, labels


def _result_row(
    *,
    dataset: EvalDataset,
    featurizer_name: str,
    featurizer_type: str,
    head: LightweightHead,
    result: LightweightParityResult,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    train_features: FeatureBatch,
    eval_features: FeatureBatch,
    n_train: int,
    n_eval: int,
    seed: int | None,
) -> dict[str, Any]:
    train_feature_metadata = train_features.metadata
    eval_feature_metadata = eval_features.metadata
    return {
        "suite_dataset": dataset.name,
        "dataset": dataset.name,
        "task": "__all__",
        "task_names": ",".join(dataset.task_names),
        "n_tasks": len(dataset.task_names),
        "task_type": dataset.task_type,
        "split": "test",
        "featurizer": featurizer_name,
        "featurizer_type": featurizer_type,
        "downstream_name": head,
        "downstream_model": "lightweight_parity_classifier",
        "seed": seed,
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "n_train_total": int(len(train_frame)),
        "n_eval_total": int(len(eval_frame)),
        "n_train_feature_valid": int(train_features.valid_mask.sum()),
        "n_eval_feature_valid": int(eval_features.valid_mask.sum()),
        "train_feature_invalid_rate": 1.0
        - train_features.valid_mask.sum() / max(1, len(train_frame)),
        "eval_feature_invalid_rate": 1.0 - eval_features.valid_mask.sum() / max(1, len(eval_frame)),
        "train_feature_cache_key": train_feature_metadata.get("cache_key"),
        "eval_feature_cache_key": eval_feature_metadata.get("cache_key"),
        "train_feature_dim": train_feature_metadata.get("n_features"),
        "eval_feature_dim": eval_feature_metadata.get("n_features"),
        "roc_auc": result.roc_auc,
        "cv_roc_auc": result.best_score,
        "downstream_best_params": json.dumps(_json_safe(result.best_params), sort_keys=True),
    }


def _write_manifest(
    *,
    output_dir: Path,
    suite: BenchmarkSuiteConfig,
    cache_dir: Path,
    run_records: list[dict[str, Any]],
    n_result_rows: int,
) -> None:
    manifest = {
        "suite_name": suite.name,
        "eval_split": suite.eval_split,
        "batch_size": suite.batch_size,
        "use_cache": suite.use_cache,
        "cache_dir": str(cache_dir),
        "parity": "lightweight",
        "n_result_rows": int(n_result_rows),
        "n_skipped_rows": 0,
        "runs": run_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
