from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modernmolbert.eval.cache import get_or_compute_features
from modernmolbert.eval.datasets import EvalDataset
from modernmolbert.eval.downstream import (
    FrozenDownstreamConfig,
    fit_predict_downstream,
)
from modernmolbert.eval.featurizers.base import FeatureBatch, RepresentationFeaturizer
from modernmolbert.eval.io import ensure_dir, write_json
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
class FrozenBenchmarkResult:
    dataset: str
    featurizer: str
    task_results: list[TaskResult]
    skipped_tasks: list[TaskSkip] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "featurizer": self.featurizer,
            "task_results": [asdict(x) for x in self.task_results],
            "skipped_tasks": [asdict(x) for x in self.skipped_tasks],
        }


@dataclass
class FrozenBenchmarkRunner:
    """Run frozen-representation benchmarks with a shared downstream learner."""

    downstream_config: FrozenDownstreamConfig = field(
        default_factory=FrozenDownstreamConfig
    )
    cache_dir: Path | None = None
    use_cache: bool = True
    batch_size: int = 64
    random_state: int = 13

    def run(
        self,
        *,
        dataset: EvalDataset,
        featurizer: RepresentationFeaturizer,
        output_dir: str | Path | None = None,
        eval_split: str = "test",
    ) -> FrozenBenchmarkResult:
        dataset.check()

        if eval_split == "test":
            eval_frame = dataset.test
        elif eval_split == "valid":
            if dataset.valid is None:
                raise ValueError(
                    "eval_split='valid' requested but dataset has no valid split"
                )
            eval_frame = dataset.valid
        else:
            raise ValueError("eval_split must be 'valid' or 'test'")

        train_frame = dataset.train

        train_features = self._features_for_split(
            dataset=dataset,
            split_name="train",
            frame=train_frame,
            featurizer=featurizer,
        )
        eval_features = self._features_for_split(
            dataset=dataset,
            split_name=eval_split,
            frame=eval_frame,
            featurizer=featurizer,
        )

        task_results: list[TaskResult] = []
        skipped_tasks: list[TaskSkip] = []

        for task in dataset.task_names:
            task_result, task_skip = self._run_single_task(
                dataset=dataset,
                task=task,
                eval_split=eval_split,
                featurizer=featurizer,
                train_frame=train_frame,
                eval_frame=eval_frame,
                train_features=train_features,
                eval_features=eval_features,
            )
            if task_result is not None:
                task_results.append(task_result)
            if task_skip is not None:
                skipped_tasks.append(task_skip)

        result = FrozenBenchmarkResult(
            dataset=dataset.name,
            featurizer=featurizer.name,
            task_results=task_results,
            skipped_tasks=skipped_tasks,
        )

        if output_dir is not None:
            self.write_outputs(result=result, output_dir=output_dir)

        return result

    def _features_for_split(
        self,
        *,
        dataset: EvalDataset,
        split_name: str,
        frame: pd.DataFrame,
        featurizer: RepresentationFeaturizer,
    ) -> FeatureBatch:
        return get_or_compute_features(
            dataset_name=dataset.name,
            split_name=split_name,
            frame=frame,
            smiles_column=dataset.smiles_column,
            featurizer=featurizer,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            batch_size=self.batch_size,
        )

    def _run_single_task(
        self,
        *,
        dataset: EvalDataset,
        task: str,
        eval_split: str,
        featurizer: RepresentationFeaturizer,
        train_frame: pd.DataFrame,
        eval_frame: pd.DataFrame,
        train_features: FeatureBatch,
        eval_features: FeatureBatch,
    ) -> tuple[TaskResult | None, TaskSkip | None]:
        train_label_mask = _valid_label_mask(train_frame, task)
        eval_label_mask = _valid_label_mask(eval_frame, task)

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

        n_train_label_valid = int(train_label_mask.sum())
        n_eval_label_valid = int(eval_label_mask.sum())
        n_train_feature_valid = int(train_features.valid_mask.sum())
        n_eval_feature_valid = int(eval_features.valid_mask.sum())

        if int(train_keep_original.sum()) == 0:
            return None, TaskSkip(
                dataset=dataset.name,
                task=task,
                split=eval_split,
                reason="no_train_rows_after_label_and_feature_filtering",
                n_train_label_valid_rows=n_train_label_valid,
                n_eval_label_valid_rows=n_eval_label_valid,
                n_train_feature_valid_rows=n_train_feature_valid,
                n_eval_feature_valid_rows=n_eval_feature_valid,
            )

        if int(eval_keep_original.sum()) == 0:
            return None, TaskSkip(
                dataset=dataset.name,
                task=task,
                split=eval_split,
                reason="no_eval_rows_after_label_and_feature_filtering",
                n_train_label_valid_rows=n_train_label_valid,
                n_eval_label_valid_rows=n_eval_label_valid,
                n_train_feature_valid_rows=n_train_feature_valid,
                n_eval_feature_valid_rows=n_eval_feature_valid,
            )

        train_label_mask_among_valid = train_label_mask[train_features.valid_mask]
        eval_label_mask_among_valid = eval_label_mask[eval_features.valid_mask]

        X_train = train_features.X[train_label_mask_among_valid]
        X_eval = eval_features.X[eval_label_mask_among_valid]

        y_train = train_frame.loc[train_keep_original, task].to_numpy()
        y_eval = eval_frame.loc[eval_keep_original, task].to_numpy()

        if dataset.task_type == "classification":
            y_train = y_train.astype(int)
            y_eval = y_eval.astype(int)

            if len(np.unique(y_train)) < 2:
                return None, TaskSkip(
                    dataset=dataset.name,
                    task=task,
                    split=eval_split,
                    reason="classification_train_has_single_class",
                    n_train_label_valid_rows=n_train_label_valid,
                    n_eval_label_valid_rows=n_eval_label_valid,
                    n_train_feature_valid_rows=n_train_feature_valid,
                    n_eval_feature_valid_rows=n_eval_feature_valid,
                )
        else:
            y_train = y_train.astype(float)
            y_eval = y_eval.astype(float)

        pred = fit_predict_downstream(
            task_type=dataset.task_type,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            config=self.downstream_config,
        )

        metrics = compute_metrics(
            task_type=dataset.task_type,
            y_true=y_eval,
            y_pred=pred.y_pred,
            y_score=pred.y_score,
        )

        return TaskResult(
            dataset=dataset.name,
            task=task,
            task_type=dataset.task_type,
            split=eval_split,
            featurizer=featurizer.name,
            metrics=metrics,
            n_train=int(len(y_train)),
            n_eval=int(len(y_eval)),
            n_train_total=int(len(train_frame)),
            n_eval_total=int(len(eval_frame)),
            n_train_feature_valid=n_train_feature_valid,
            n_eval_feature_valid=n_eval_feature_valid,
            downstream_metadata=pred.metadata,
            feature_metadata={
                "train": train_features.metadata,
                "eval": eval_features.metadata,
            },
        ), None

    def write_outputs(
        self,
        *,
        result: FrozenBenchmarkResult,
        output_dir: str | Path,
    ) -> None:
        out = ensure_dir(output_dir)
        write_json(out / "results.json", result.to_dict())

        rows = []
        for task_result in result.task_results:
            train_feature_metadata = task_result.feature_metadata.get("train", {})
            eval_feature_metadata = task_result.feature_metadata.get("eval", {})

            base = {
                "dataset": task_result.dataset,
                "task": task_result.task,
                "task_type": task_result.task_type,
                "split": task_result.split,
                "featurizer": task_result.featurizer,
                "n_train": task_result.n_train,
                "n_eval": task_result.n_eval,
                "n_train_total": task_result.n_train_total,
                "n_eval_total": task_result.n_eval_total,
                "n_train_feature_valid": task_result.n_train_feature_valid,
                "n_eval_feature_valid": task_result.n_eval_feature_valid,
                "train_feature_invalid_rate": 1.0
                - (
                    task_result.n_train_feature_valid
                    / max(1, task_result.n_train_total)
                ),
                "eval_feature_invalid_rate": 1.0
                - (task_result.n_eval_feature_valid / max(1, task_result.n_eval_total)),
                "train_feature_cache_key": train_feature_metadata.get("cache_key"),
                "eval_feature_cache_key": eval_feature_metadata.get("cache_key"),
                "train_feature_dim": train_feature_metadata.get("n_features"),
                "eval_feature_dim": eval_feature_metadata.get("n_features"),
            }

            for metric_name, metric_value in task_result.metrics.items():
                base[metric_name] = metric_value

            base["model_type"] = task_result.downstream_metadata.get("downstream_model")
            base["standardize"] = task_result.downstream_metadata.get("standardize")

            for key, value in task_result.downstream_metadata.items():
                if key in {"downstream_model", "standardize"}:
                    continue
                base[f"downstream_{key}"] = value

            rows.append(base)

        if rows:
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


def _valid_label_mask(frame: pd.DataFrame, task: str) -> np.ndarray:
    y = pd.to_numeric(frame[task], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)

    weight_col = f"{task}__weight"
    if weight_col in frame.columns:
        w = pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float)
        mask = mask & np.isfinite(w) & (w != 0)

    return np.asarray(mask, dtype=bool)
