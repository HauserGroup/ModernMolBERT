from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from modernmolbert.eval.cache import get_or_compute_features
from modernmolbert.eval.datasets import EvalDataset
from modernmolbert.eval.downstream import (
    FrozenDownstreamConfig,
)
from modernmolbert.eval.featurizers.base import FeatureBatch, RepresentationFeaturizer
from modernmolbert.eval.io import ensure_dir, write_json

from modernmolbert.eval.task_eval import TaskResult, TaskSkip, evaluate_single_task


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
            task_result, task_skip = evaluate_single_task(
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
