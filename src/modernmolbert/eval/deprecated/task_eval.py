from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from modernmolbert.eval.downstream import (
    FrozenDownstreamConfig,
    fit_predict_downstream,
)
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.metrics import compute_metrics


@dataclass(frozen=True)
class TaskResult:
    dataset: str
    task: str
    task_type: str
    split: str
    featurizer: str
    metrics: dict[str, float]
    n_train: int
    n_eval: int
    n_train_total: int
    n_eval_total: int
    n_train_feature_valid: int
    n_eval_feature_valid: int
    downstream_metadata: dict[str, Any] = field(default_factory=dict)
    feature_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSkip:
    dataset: str
    task: str
    split: str
    reason: str
    n_train_label_valid_rows: int
    n_eval_label_valid_rows: int
    n_train_feature_valid_rows: int
    n_eval_feature_valid_rows: int


@dataclass(frozen=True)
class TaskPredictionArtifact:
    dataset: str
    task: str
    task_type: str
    split: str
    featurizer: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    eval_original_index: np.ndarray
    metrics: dict[str, float]
    downstream_metadata: dict[str, Any] = field(default_factory=dict)
    n_eval_total: int = 0
    n_eval: int = 0


@dataclass(frozen=True)
class AlignedTaskData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    train_keep_original: np.ndarray
    eval_keep_original: np.ndarray
    n_train_label_valid: int
    n_eval_label_valid: int
    n_train_feature_valid: int
    n_eval_feature_valid: int


def valid_label_mask(frame: pd.DataFrame, task: str) -> np.ndarray:
    y = pd.to_numeric(frame[task], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)

    weight_col = f"{task}__weight"
    if weight_col in frame.columns:
        w = pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float)
        mask = mask & np.isfinite(w) & (w != 0)

    return np.asarray(mask, dtype=bool)


def align_task_data(
    *,
    task: str,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    train_features: FeatureBatch,
    eval_features: FeatureBatch,
) -> AlignedTaskData:
    """Align labels with valid feature rows for one task."""

    train_label_mask = valid_label_mask(train_frame, task)
    eval_label_mask = valid_label_mask(eval_frame, task)

    if train_label_mask.shape != train_features.valid_mask.shape:
        raise ValueError(
            "Train label mask and train feature mask have different shapes: "
            f"{train_label_mask.shape} != {train_features.valid_mask.shape}"
        )

    if eval_label_mask.shape != eval_features.valid_mask.shape:
        raise ValueError(
            "Eval label mask and eval feature mask have different shapes: "
            f"{eval_label_mask.shape} != {eval_features.valid_mask.shape}"
        )

    train_keep_original = train_label_mask & train_features.valid_mask
    eval_keep_original = eval_label_mask & eval_features.valid_mask

    train_label_mask_among_valid = train_label_mask[train_features.valid_mask]
    eval_label_mask_among_valid = eval_label_mask[eval_features.valid_mask]

    X_train = train_features.X[train_label_mask_among_valid]
    X_eval = eval_features.X[eval_label_mask_among_valid]

    y_train = train_frame.loc[train_keep_original, task].to_numpy()
    y_eval = eval_frame.loc[eval_keep_original, task].to_numpy()

    return AlignedTaskData(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        train_keep_original=train_keep_original,
        eval_keep_original=eval_keep_original,
        n_train_label_valid=int(train_label_mask.sum()),
        n_eval_label_valid=int(eval_label_mask.sum()),
        n_train_feature_valid=int(train_features.valid_mask.sum()),
        n_eval_feature_valid=int(eval_features.valid_mask.sum()),
    )


def make_task_skip(
    *,
    dataset_name: str,
    task: str,
    eval_split: str,
    reason: str,
    aligned: AlignedTaskData,
) -> TaskSkip:
    return TaskSkip(
        dataset=dataset_name,
        task=task,
        split=eval_split,
        reason=reason,
        n_train_label_valid_rows=aligned.n_train_label_valid,
        n_eval_label_valid_rows=aligned.n_eval_label_valid,
        n_train_feature_valid_rows=aligned.n_train_feature_valid,
        n_eval_feature_valid_rows=aligned.n_eval_feature_valid,
    )


def evaluate_single_task(
    *,
    dataset_name: str,
    task: str,
    task_type: str,
    eval_split: str,
    featurizer_name: str,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    train_features: FeatureBatch,
    eval_features: FeatureBatch,
    downstream_config: FrozenDownstreamConfig,
) -> tuple[TaskResult | None, TaskSkip | None]:
    """Fit/evaluate one task or return a structured skip."""

    task_result, task_skip, _ = evaluate_single_task_with_predictions(
        dataset_name=dataset_name,
        task=task,
        task_type=task_type,
        eval_split=eval_split,
        featurizer_name=featurizer_name,
        train_frame=train_frame,
        eval_frame=eval_frame,
        train_features=train_features,
        eval_features=eval_features,
        downstream_config=downstream_config,
    )
    return task_result, task_skip


def evaluate_single_task_with_predictions(
    *,
    dataset_name: str,
    task: str,
    task_type: str,
    eval_split: str,
    featurizer_name: str,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    train_features: FeatureBatch,
    eval_features: FeatureBatch,
    downstream_config: FrozenDownstreamConfig,
) -> tuple[TaskResult | None, TaskSkip | None, TaskPredictionArtifact | None]:
    """Fit/evaluate one task and keep optional prediction arrays out-of-band."""

    aligned = align_task_data(
        task=task,
        train_frame=train_frame,
        eval_frame=eval_frame,
        train_features=train_features,
        eval_features=eval_features,
    )

    if len(aligned.y_train) == 0:
        return (
            None,
            make_task_skip(
                dataset_name=dataset_name,
                task=task,
                eval_split=eval_split,
                reason="no_train_rows_after_label_and_feature_filtering",
                aligned=aligned,
            ),
            None,
        )

    if len(aligned.y_eval) == 0:
        return (
            None,
            make_task_skip(
                dataset_name=dataset_name,
                task=task,
                eval_split=eval_split,
                reason="no_eval_rows_after_label_and_feature_filtering",
                aligned=aligned,
            ),
            None,
        )

    if task_type == "classification":
        y_train = aligned.y_train.astype(int)
        y_eval = aligned.y_eval.astype(int)

        if len(np.unique(y_train)) < 2:
            return (
                None,
                make_task_skip(
                    dataset_name=dataset_name,
                    task=task,
                    eval_split=eval_split,
                    reason="classification_train_has_single_class",
                    aligned=aligned,
                ),
                None,
            )

    elif task_type == "regression":
        y_train = aligned.y_train.astype(float)
        y_eval = aligned.y_eval.astype(float)

    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    pred = fit_predict_downstream(
        task_type=task_type,  # type: ignore[arg-type]
        X_train=aligned.X_train,
        y_train=y_train,
        X_eval=aligned.X_eval,
        config=downstream_config,
    )

    metrics = compute_metrics(
        task_type=task_type,  # type: ignore[arg-type]
        y_true=y_eval,
        y_pred=pred.y_pred,
        y_score=pred.y_score,
    )

    result = TaskResult(
        dataset=dataset_name,
        task=task,
        task_type=task_type,
        split=eval_split,
        featurizer=featurizer_name,
        metrics=metrics,
        n_train=int(len(y_train)),
        n_eval=int(len(y_eval)),
        n_train_total=int(len(train_frame)),
        n_eval_total=int(len(eval_frame)),
        n_train_feature_valid=aligned.n_train_feature_valid,
        n_eval_feature_valid=aligned.n_eval_feature_valid,
        downstream_metadata=pred.metadata,
        feature_metadata={
            "train": train_features.metadata,
            "eval": eval_features.metadata,
        },
    )

    prediction_artifact = TaskPredictionArtifact(
        dataset=dataset_name,
        task=task,
        task_type=task_type,
        split=eval_split,
        featurizer=featurizer_name,
        y_true=np.asarray(y_eval),
        y_pred=np.asarray(pred.y_pred),
        y_score=None if pred.y_score is None else np.asarray(pred.y_score),
        eval_original_index=np.flatnonzero(aligned.eval_keep_original).astype(np.int64),
        metrics=metrics,
        downstream_metadata=pred.metadata,
        n_eval_total=int(len(eval_frame)),
        n_eval=int(len(y_eval)),
    )

    return result, None, prediction_artifact
