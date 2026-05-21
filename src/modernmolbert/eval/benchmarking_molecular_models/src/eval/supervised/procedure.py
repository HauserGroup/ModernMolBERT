import gc
import os

import joblib
import json
import torch
import logging as log
from pathlib import Path

from os.path import join
from typing import Any

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    append_result_row,
    count_result_rows,
    delete_result_rows,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    EmbeddedDataset,
    EvaluationResult,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.const import (
    DEFAULT_MEMORY_WEIGHT,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.eval_metrics import (
    evaluate,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.train import (
    fit_and_eval_embedding,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.utils import (
    NpEncoder,
    get_model_version_hash,
)


def load_embedded_dataset(
    embedded_dir: str,
    dataset_info: Any,
    model_name: str,
) -> "EmbeddedDataset | None":
    """Load and preprocess an embedding from disk.

    Handles legacy JSON format, torch tensor conversion, and 1-D reshape.
    Returns None if the file is missing; raises on empty data.
    """
    embedded_filename = join(os.getcwd(), embedded_dir, dataset_info.name, f"{model_name}.joblib")
    legacy_filename = join(os.getcwd(), embedded_dir, dataset_info.name, f"{model_name}.json")

    if os.path.exists(legacy_filename):
        log.info("Legacy embedded dataset found, converting to new format")
        embedded_data = EmbeddedDataset.deserialize_legacy(legacy_filename)
    elif not os.path.exists(embedded_filename):
        log.error(f"Embedded dataset not found: {embedded_filename}")
        return None
    else:
        embedded_data: EmbeddedDataset = joblib.load(embedded_filename, mmap_mode="r")

    if embedded_data.X is None:
        log.error("Embedded dataset is empty")
        raise RuntimeError("Embedded dataset is empty")

    if isinstance(embedded_data.X, torch.Tensor):
        log.info("Converting torch.Tensor to numpy array")
        embedded_data.X = embedded_data.X.detach().cpu().numpy()

    if len(embedded_data.X.shape) == 1:
        log.warning("Invalid X shape (1 dim), assuming invalid concatenation")
        desired_samples = embedded_data.y.shape[0]
        embedded_data.X = embedded_data.X.reshape(desired_samples, -1)

    log.info(
        f"Shape {embedded_data.X.shape} for dataset {embedded_data.name}, task {embedded_data.task}"
    )
    return embedded_data


def eval_embedding(
    data: EmbeddedDataset,
    pred_directory: str,
    dataset_config,
    metric_name: str,
    model_head: str,
) -> EvaluationResult:
    log.info("Training model")
    head_result = fit_and_eval_embedding(
        dataset=data,
        metric_name=metric_name,
        model_head=model_head,
        memory_weight=dataset_config.get("memory_weight", DEFAULT_MEMORY_WEIGHT),
    )
    log.info(f"Training complete, best CV result: {head_result.cv_score}")
    return evaluate(head_result, dataset_config, pred_directory)


def dump_hyperparams(hyperparams: dict) -> str:
    return json.dumps(hyperparams, sort_keys=True, cls=NpEncoder)


def check_if_already_evaluated(
    output_csv: str | Path,
    dataset_name: str,
    model_name: str,
    metric_name: str,
    head_name: str,
) -> bool:
    ctx = f"dataset={dataset_name!r} embedder={model_name!r} metric={metric_name!r} head={head_name!r}"
    try:
        runs = count_result_rows(
            output_csv,
            dataset=dataset_name,
            embedder=model_name,
            cv_metric_name=metric_name,
            model=head_name,
        )
    except Exception as exc:
        # Corrupt or unreadable CSV — treat as not yet evaluated so the run
        # proceeds and overwrites the bad file.
        log.warning(f"Could not read results CSV ({output_csv}): {exc}. Treating as not evaluated.")
        return False

    if runs > 1:
        # Duplicate rows indicate a previously interrupted override. Remove
        # all of them and return False so the caller reruns and writes a clean row.
        log.warning(
            f"Found {runs} duplicate result rows for {ctx}. Deleting all and re-evaluating."
        )
        delete_previous_evaluations(output_csv, dataset_name, model_name, metric_name, head_name)
        return False

    return runs == 1


def delete_previous_evaluations(
    output_csv: str | Path,
    dataset_name: str,
    model_name: str,
    metric_name: str,
    head_name: str,
):
    delete_result_rows(
        output_csv,
        dataset=dataset_name,
        embedder=model_name,
        cv_metric_name=metric_name,
        model=head_name,
    )
    log.warning(
        f"Deleted previous evaluations, dataset: {dataset_name}, model: {model_name}, metric: {metric_name}, head: {head_name}"
    )


def eval_procedure(
    dataset_info: Any,
    embedded_dir: str,
    predictions_dir: str,
    model_name: str,
    model_head: str,
    output_csv: str | Path,
    override: bool = False,
    preloaded: "EmbeddedDataset | None" = None,
):
    model_version_hash = get_model_version_hash()

    if check_if_already_evaluated(
        output_csv, dataset_info.name, model_name, dataset_info.metric, model_head
    ):
        if not override:
            log.info(
                f"Already evaluated — skipping. "
                f"dataset={dataset_info.name!r} embedder={model_name!r} head={model_head!r}"
            )
            return
        log.warning(
            f"Already evaluated — overriding. "
            f"dataset={dataset_info.name!r} embedder={model_name!r} head={model_head!r}"
        )
        delete_previous_evaluations(
            output_csv, dataset_info.name, model_name, dataset_info.metric, model_head
        )

    if model_head == "knn" and "muv" in dataset_info.name:
        log.error("Skipping KNN evaluation for MUV datasets, not supported")
        return

    owns_data = preloaded is None
    if owns_data:
        embedded_data = load_embedded_dataset(embedded_dir, dataset_info, model_name)
        if embedded_data is None:
            return
    else:
        embedded_data = preloaded

    result = eval_embedding(
        embedded_data,
        predictions_dir,
        dataset_info,
        dataset_info.metric,
        model_head,
    )
    log.info(f"Evaluation complete, test result: {result.metric_value}")

    dataset_name = embedded_data.name
    task = embedded_data.task
    if owns_data:
        del embedded_data
        gc.collect()

    append_result_row(
        output_csv,
        {
            "dataset": dataset_name,
            "task": task,
            "embedder": model_name,
            "model": result.model,
            "hyperparams": dump_hyperparams(result.hyperparams),
            "library_hash": model_version_hash,
            "cv_metric_name": dataset_info.metric,
            "cv_metric": result.cv_metric_value,
            "test_metric_name": result.metric_name,
            "test_metric": result.metric_value,
        },
    )
