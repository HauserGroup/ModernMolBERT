from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from modernmolbert.eval.datasets import load_eval_dataset_from_config
from modernmolbert.eval.downstream import FrozenDownstreamConfig
from modernmolbert.eval.registry import make_featurizer_from_config
from modernmolbert.eval.runner import FrozenBenchmarkResult, FrozenBenchmarkRunner

EvalSplit = Literal["valid", "test"]


@dataclass(frozen=True)
class SuiteDatasetConfig:
    config: dict[str, Any]


@dataclass(frozen=True)
class SuiteFeaturizerConfig:
    config: dict[str, Any]


@dataclass(frozen=True)
class SuiteDownstreamConfig:
    name: str
    task_type: Literal["classification", "regression"]
    config: FrozenDownstreamConfig


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    name: str
    datasets: list[SuiteDatasetConfig]
    featurizers: list[SuiteFeaturizerConfig]
    downstream_models: list[SuiteDownstreamConfig]
    seeds: list[int] = field(default_factory=lambda: [13])
    eval_split: EvalSplit = "test"
    batch_size: int = 64
    use_cache: bool = True


def _normalize_downstream_models(raw: Mapping[str, Any]) -> list[SuiteDownstreamConfig]:
    out: list[SuiteDownstreamConfig] = []

    for task_type in ["classification", "regression"]:
        entries = raw.get(task_type, [])
        if entries is None:
            continue

        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"downstream_models.{task_type} must be a list of model configs")

        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"Each downstream model config for {task_type} must be a mapping")

            name = str(entry.get("name") or entry.get("model_type") or "auto")
            model_type = str(entry.get("model_type", "auto"))
            params = entry.get("params", None)
            standardize = bool(entry.get("standardize", True))
            random_state = int(entry.get("random_state", 13))

            if params is not None and not isinstance(params, Mapping):
                raise ValueError(f"params for downstream model {name!r} must be a mapping")

            out.append(
                SuiteDownstreamConfig(
                    name=name,
                    task_type=task_type,  # type: ignore[arg-type]
                    config=FrozenDownstreamConfig(
                        model_type=model_type,
                        params=None if params is None else dict(params),
                        random_state=random_state,
                        standardize=standardize,
                    ),
                )
            )

    if not out:
        raise ValueError("Suite must define at least one downstream model")

    return out


def suite_config_from_dict(config: Mapping[str, Any]) -> BenchmarkSuiteConfig:
    name = str(config.get("name", "benchmark_suite"))

    datasets_raw = config.get("datasets", [])
    if not isinstance(datasets_raw, Sequence) or isinstance(datasets_raw, (str, bytes)):
        raise ValueError("Suite field 'datasets' must be a list")
    if not datasets_raw:
        raise ValueError("Suite must define at least one dataset")

    featurizers_raw = config.get("featurizers", [])
    if not isinstance(featurizers_raw, Sequence) or isinstance(featurizers_raw, (str, bytes)):
        raise ValueError("Suite field 'featurizers' must be a list")
    if not featurizers_raw:
        raise ValueError("Suite must define at least one featurizer")

    downstream_raw = config.get("downstream_models", {})
    if not isinstance(downstream_raw, Mapping):
        raise ValueError("Suite field 'downstream_models' must be a mapping")

    seeds_raw = config.get("seeds", [13])
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
        raise ValueError("Suite field 'seeds' must be a list of integers")

    eval_split = str(config.get("eval_split", "test"))
    if eval_split not in {"valid", "test"}:
        raise ValueError("Suite eval_split must be 'valid' or 'test'")

    return BenchmarkSuiteConfig(
        name=name,
        datasets=[
            SuiteDatasetConfig(config={str(k): v for k, v in dict(item).items()})
            for item in datasets_raw
        ],
        featurizers=[
            SuiteFeaturizerConfig(config={str(k): v for k, v in dict(item).items()})
            for item in featurizers_raw
        ],
        downstream_models=_normalize_downstream_models(downstream_raw),
        seeds=[int(seed) for seed in seeds_raw],
        eval_split=eval_split,  # type: ignore[arg-type]
        batch_size=int(config.get("batch_size", 64)),
        use_cache=bool(config.get("use_cache", True)),
    )


def load_suite_config(path: str | Path) -> BenchmarkSuiteConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Suite config not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))

    elif suffix == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))

    else:
        raise ValueError(f"Unsupported suite config format: {path}")

    if not isinstance(data, Mapping):
        raise ValueError(f"Suite config must be a mapping: {path}")

    return suite_config_from_dict(data)


def _result_to_rows(
    *,
    result: FrozenBenchmarkResult,
    dataset_config: Mapping[str, Any],
    featurizer_config: Mapping[str, Any],
    downstream_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for task_result in result.task_results:
        train_feature_metadata = task_result.feature_metadata.get("train", {})
        eval_feature_metadata = task_result.feature_metadata.get("eval", {})

        row: dict[str, Any] = {
            "suite_dataset": dataset_config.get("name"),
            "dataset": task_result.dataset,
            "task": task_result.task,
            "task_type": task_result.task_type,
            "split": task_result.split,
            "featurizer": task_result.featurizer,
            "featurizer_type": featurizer_config.get("type"),
            "downstream_name": downstream_name,
            "downstream_model": result.downstream_config.get("model_type"),
            "seed": seed,
            "n_train": task_result.n_train,
            "n_eval": task_result.n_eval,
            "n_train_total": task_result.n_train_total,
            "n_eval_total": task_result.n_eval_total,
            "n_train_feature_valid": task_result.n_train_feature_valid,
            "n_eval_feature_valid": task_result.n_eval_feature_valid,
            "train_feature_invalid_rate": 1.0
            - task_result.n_train_feature_valid / max(1, task_result.n_train_total),
            "eval_feature_invalid_rate": 1.0
            - task_result.n_eval_feature_valid / max(1, task_result.n_eval_total),
            "train_feature_cache_key": train_feature_metadata.get("cache_key"),
            "eval_feature_cache_key": eval_feature_metadata.get("cache_key"),
            "train_feature_dim": train_feature_metadata.get("n_features"),
            "eval_feature_dim": eval_feature_metadata.get("n_features"),
        }

        for metric_name, metric_value in task_result.metrics.items():
            row[metric_name] = metric_value

        for key, value in task_result.downstream_metadata.items():
            row[f"downstream_{key}"] = value

        rows.append(row)

    return rows


def _skip_to_rows(
    *,
    result: FrozenBenchmarkResult,
    dataset_config: Mapping[str, Any],
    featurizer_config: Mapping[str, Any],
    downstream_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for skip in result.skipped_tasks:
        rows.append(
            {
                "suite_dataset": dataset_config.get("name"),
                "dataset": skip.dataset,
                "task": skip.task,
                "split": skip.split,
                "featurizer": result.featurizer,
                "featurizer_type": featurizer_config.get("type"),
                "downstream_name": downstream_name,
                "downstream_model": result.downstream_config.get("model_type"),
                "seed": seed,
                "reason": skip.reason,
                "n_train_label_valid_rows": skip.n_train_label_valid_rows,
                "n_eval_label_valid_rows": skip.n_eval_label_valid_rows,
                "n_train_feature_valid_rows": skip.n_train_feature_valid_rows,
                "n_eval_feature_valid_rows": skip.n_eval_feature_valid_rows,
            }
        )

    return rows


def _safe_config_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            out[str(key)] = str(value)
        elif isinstance(value, Mapping):
            out[str(key)] = _safe_config_dict(value)
        elif isinstance(value, list):
            out[str(key)] = [
                _safe_config_dict(item) if isinstance(item, Mapping) else item for item in value
            ]
        else:
            out[str(key)] = value
    return out


def run_benchmark_suite(
    *,
    suite: BenchmarkSuiteConfig,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    write_single_run_outputs: bool = False,
) -> pd.DataFrame:
    """Run a full benchmark suite and write aggregate outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / "cache"

    all_rows: list[dict[str, Any]] = []
    all_skip_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for dataset_index, dataset_cfg in enumerate(suite.datasets):
        dataset = load_eval_dataset_from_config(dataset_cfg.config)

        matching_downstreams = [
            downstream
            for downstream in suite.downstream_models
            if downstream.task_type == dataset.task_type
        ]

        if not matching_downstreams:
            raise ValueError(
                f"No downstream models configured for dataset {dataset.name!r} "
                f"with task_type={dataset.task_type!r}"
            )

        for featurizer_index, featurizer_cfg in enumerate(suite.featurizers):
            featurizer = make_featurizer_from_config(featurizer_cfg.config)

            for downstream_cfg in matching_downstreams:
                for seed in suite.seeds:
                    downstream_config = FrozenDownstreamConfig(
                        model_type=downstream_cfg.config.model_type,
                        params=downstream_cfg.config.params,
                        random_state=seed,
                        standardize=downstream_cfg.config.standardize,
                    )

                    runner = FrozenBenchmarkRunner(
                        downstream_config=downstream_config,
                        cache_dir=resolved_cache_dir,
                        use_cache=suite.use_cache,
                        batch_size=suite.batch_size,
                        random_state=seed,
                    )

                    single_output_dir = None
                    if write_single_run_outputs:
                        single_output_dir = (
                            output_dir
                            / "runs"
                            / f"dataset_{dataset_index:03d}_{dataset.name}"
                            / f"featurizer_{featurizer_index:03d}_{featurizer.name}"
                            / f"downstream_{downstream_cfg.name}"
                            / f"seed_{seed}"
                        )

                    result = runner.run(
                        dataset=dataset,
                        featurizer=featurizer,
                        output_dir=single_output_dir,
                        eval_split=suite.eval_split,
                    )

                    all_rows.extend(
                        _result_to_rows(
                            result=result,
                            dataset_config=dataset_cfg.config,
                            featurizer_config=featurizer_cfg.config,
                            downstream_name=downstream_cfg.name,
                            seed=seed,
                        )
                    )

                    all_skip_rows.extend(
                        _skip_to_rows(
                            result=result,
                            dataset_config=dataset_cfg.config,
                            featurizer_config=featurizer_cfg.config,
                            downstream_name=downstream_cfg.name,
                            seed=seed,
                        )
                    )

                    run_records.append(
                        {
                            "dataset": dataset.name,
                            "dataset_config": _safe_config_dict(dataset_cfg.config),
                            "featurizer": featurizer.name,
                            "featurizer_config": _safe_config_dict(featurizer_cfg.config),
                            "downstream_name": downstream_cfg.name,
                            "downstream_config": {
                                "model_type": downstream_config.model_type,
                                "params": downstream_config.params or {},
                                "random_state": downstream_config.random_state,
                                "standardize": downstream_config.standardize,
                            },
                            "seed": seed,
                            "eval_split": suite.eval_split,
                            "n_task_results": len(result.task_results),
                            "n_skipped_tasks": len(result.skipped_tasks),
                        }
                    )

    results = pd.DataFrame(all_rows)
    skipped = pd.DataFrame(all_skip_rows)

    results.to_csv(output_dir / "results.csv", index=False)

    if not skipped.empty:
        skipped.to_csv(output_dir / "skipped_tasks.csv", index=False)

    import json

    manifest = {
        "suite_name": suite.name,
        "eval_split": suite.eval_split,
        "batch_size": suite.batch_size,
        "use_cache": suite.use_cache,
        "cache_dir": str(resolved_cache_dir),
        "n_result_rows": int(len(results)),
        "n_skipped_rows": int(len(skipped)),
        "runs": run_records,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return results
