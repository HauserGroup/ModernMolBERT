import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from modernmolbert.eval.cache import get_or_compute_features
from modernmolbert.eval.datasets import EvalDataset
from modernmolbert.eval.downstream import FrozenDownstreamConfig
from modernmolbert.eval.featurizers.base import FeatureBatch, RepresentationFeaturizer
from modernmolbert.eval.io import ensure_dir, write_json
from modernmolbert.eval.task_eval import (
    TaskPredictionArtifact,
    TaskResult,
    TaskSkip,
    evaluate_single_task_with_predictions,
)

EvalSplit = Literal["valid", "test"]


@dataclass(frozen=True)
class FrozenBenchmarkResult:
    dataset: str
    featurizer: str
    eval_split: str
    downstream_config: dict[str, Any]
    task_results: list[TaskResult]
    skipped_tasks: list[TaskSkip] = field(default_factory=list)
    prediction_artifacts: list[TaskPredictionArtifact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "featurizer": self.featurizer,
            "eval_split": self.eval_split,
            "downstream_config": self.downstream_config,
            "task_results": [asdict(x) for x in self.task_results],
            "skipped_tasks": [asdict(x) for x in self.skipped_tasks],
        }


@dataclass
class FrozenBenchmarkRunner:
    """Run one frozen-representation benchmark configuration.

    This runner evaluates one dataset with one featurizer and one downstream
    model configuration. Feature extraction is delegated to the cache layer and
    per-task fitting/evaluation is delegated to task_eval.py.
    """

    downstream_config: FrozenDownstreamConfig = field(default_factory=FrozenDownstreamConfig)
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
        eval_split: EvalSplit = "test",
        write_predictions: bool = False,
        prediction_dir: str | Path | None = None,
        run_context: Mapping[str, Any] | None = None,
    ) -> FrozenBenchmarkResult:
        dataset.check()

        train_frame = dataset.train
        eval_frame = _select_eval_frame(dataset, eval_split)

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
        prediction_artifacts: list[TaskPredictionArtifact] = []

        for task in dataset.task_names:
            task_result, task_skip, task_prediction = evaluate_single_task_with_predictions(
                dataset_name=dataset.name,
                task=task,
                task_type=dataset.task_type,
                eval_split=eval_split,
                featurizer_name=featurizer.name,
                train_frame=train_frame,
                eval_frame=eval_frame,
                train_features=train_features,
                eval_features=eval_features,
                downstream_config=self.downstream_config,
            )

            if task_result is not None:
                task_results.append(task_result)

            if task_skip is not None:
                skipped_tasks.append(task_skip)

            if task_prediction is not None:
                prediction_artifacts.append(task_prediction)

        result = FrozenBenchmarkResult(
            dataset=dataset.name,
            featurizer=featurizer.name,
            eval_split=eval_split,
            downstream_config=_downstream_config_to_dict(self.downstream_config),
            task_results=task_results,
            skipped_tasks=skipped_tasks,
            prediction_artifacts=prediction_artifacts,
        )

        if output_dir is not None:
            self.write_outputs(result=result, output_dir=output_dir)

        if write_predictions:
            resolved_prediction_dir = (
                Path(prediction_dir)
                if prediction_dir is not None
                else None
                if output_dir is None
                else Path(output_dir) / "predictions"
            )
            if resolved_prediction_dir is None:
                raise ValueError("write_predictions=True requires output_dir or prediction_dir.")
            write_prediction_artifacts(
                artifacts=prediction_artifacts,
                output_dir=resolved_prediction_dir,
                run_context={
                    "downstream_model": self.downstream_config.model_type,
                    "seed": self.downstream_config.random_state,
                    **({} if run_context is None else dict(run_context)),
                },
            )

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

    def write_outputs(
        self,
        *,
        result: FrozenBenchmarkResult,
        output_dir: str | Path,
    ) -> None:
        out = ensure_dir(output_dir)
        write_json(out / "results.json", result.to_dict())

        result_rows = _task_results_to_rows(result)
        if result_rows:
            pd.DataFrame(result_rows).to_csv(out / "results.csv", index=False)

        skip_rows = [asdict(skip) for skip in result.skipped_tasks]
        if skip_rows:
            pd.DataFrame(skip_rows).to_csv(out / "skipped_tasks.csv", index=False)


def _select_eval_frame(dataset: EvalDataset, eval_split: EvalSplit) -> pd.DataFrame:
    if eval_split == "test":
        return dataset.test

    if eval_split == "valid":
        if dataset.valid is None:
            raise ValueError("eval_split='valid' requested but dataset has no valid split")
        return dataset.valid

    raise ValueError("eval_split must be 'valid' or 'test'")


def _downstream_config_to_dict(config: FrozenDownstreamConfig) -> dict[str, Any]:
    return {
        "model_type": config.model_type,
        "params": {} if config.params is None else dict(config.params),
        "random_state": config.random_state,
        "standardize": config.standardize,
    }


def _task_results_to_rows(result: FrozenBenchmarkResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for task_result in result.task_results:
        train_feature_metadata = task_result.feature_metadata.get("train", {})
        eval_feature_metadata = task_result.feature_metadata.get("eval", {})

        base: dict[str, Any] = {
            "dataset": task_result.dataset,
            "task": task_result.task,
            "task_type": task_result.task_type,
            "split": task_result.split,
            "featurizer": task_result.featurizer,
            "downstream_model": result.downstream_config.get("model_type"),
            "downstream_random_state": result.downstream_config.get("random_state"),
            "n_train": task_result.n_train,
            "n_eval": task_result.n_eval,
            "n_train_total": task_result.n_train_total,
            "n_eval_total": task_result.n_eval_total,
            "n_train_feature_valid": task_result.n_train_feature_valid,
            "n_eval_feature_valid": task_result.n_eval_feature_valid,
            "train_feature_invalid_rate": 1.0
            - (task_result.n_train_feature_valid / max(1, task_result.n_train_total)),
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

    return rows


def write_prediction_artifacts(
    *,
    artifacts: list[TaskPredictionArtifact],
    output_dir: str | Path,
    run_context: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write out-of-band prediction arrays and sidecar metadata."""

    output_dir = Path(output_dir)
    run_context_dict = {} if run_context is None else dict(run_context)
    written: list[Path] = []

    for artifact in artifacts:
        downstream_model = str(
            run_context_dict.get(
                "downstream_model",
                artifact.downstream_metadata.get("downstream_model", "downstream"),
            )
        )
        downstream_name = str(run_context_dict.get("downstream_name", downstream_model))
        seed = run_context_dict.get(
            "seed",
            artifact.downstream_metadata.get("random_state", "unknown"),
        )

        artifact_dir = (
            output_dir
            / _safe_slug(artifact.dataset)
            / _safe_slug(artifact.task)
            / _safe_slug(artifact.featurizer)
            / _safe_slug(downstream_name)
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        npz_path = artifact_dir / f"seed_{_safe_slug(seed)}.npz"
        if artifact.y_score is not None:
            np.savez_compressed(
                npz_path,
                y_true=artifact.y_true,
                y_pred=artifact.y_pred,
                eval_original_index=artifact.eval_original_index,
                y_score=artifact.y_score,
            )
        else:
            np.savez_compressed(
                npz_path,
                y_true=artifact.y_true,
                y_pred=artifact.y_pred,
                eval_original_index=artifact.eval_original_index,
            )

        metadata = {
            **run_context_dict,
            "dataset": artifact.dataset,
            "task": artifact.task,
            "task_type": artifact.task_type,
            "split": artifact.split,
            "featurizer": artifact.featurizer,
            "downstream_name": downstream_name,
            "downstream_model": downstream_model,
            "seed": seed,
            "metrics": artifact.metrics,
            "downstream_metadata": artifact.downstream_metadata,
            "n_eval_total": artifact.n_eval_total,
            "n_eval": artifact.n_eval,
            "n_eval_predictions": int(len(artifact.y_pred)),
            "has_y_score": artifact.y_score is not None,
            "artifact_file": npz_path.name,
        }
        json_path = npz_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(_json_safe(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(npz_path)

    return written


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return slug or "value"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value
