from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_absolute_error, precision_recall_curve

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    PRASKI_COLUMNS,
    to_praski_schema,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    Config,
    load_dataset_registry,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.eval_metrics import (
    multioutput_auroc_score,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.utils import (
    get_model_version_hash,
)


def find_prediction_npz(predictions_dir: str | Path) -> list[Path]:
    """Find score.py .npz prediction artifacts."""
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.exists():
        return []
    return sorted(predictions_dir.rglob("*.npz"))


def _finite_arrays(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    return y_true[mask], y_score[mask]


def _average_binary_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
) -> float:
    if y_true.ndim == 1:
        yt, ys = _finite_arrays(y_true, y_score)
        if yt.size == 0 or len(np.unique(yt)) < 2:
            return float("nan")
        return float(metric_fn(yt, ys))

    scores: list[float] = []
    for col in range(y_true.shape[1]):
        yt, ys = _finite_arrays(y_true[:, col], y_score[:, col])
        if yt.size == 0 or len(np.unique(yt)) < 2:
            continue
        scores.append(float(metric_fn(yt, ys)))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    def score_column(yt: np.ndarray, ys: np.ndarray) -> float:
        precision, recall, _ = precision_recall_curve(yt, ys)
        return float(auc(recall, precision))

    return _average_binary_metric(y_true, y_score, score_column)


def score_prediction_artifact(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
) -> float:
    """Compute the benchmark test metric from stored prediction arrays."""
    if metric_name == "roc_auc":
        return multioutput_auroc_score(y_true, y_score)
    if metric_name == "pr_auc_score":
        return _pr_auc_score(y_true, y_score)
    if metric_name == "mae":
        yt, ys = _finite_arrays(y_true, y_score)
        if yt.size == 0:
            return float("nan")
        return float(mean_absolute_error(yt, ys))
    raise ValueError(f"Cannot export prediction metric {metric_name!r}")


def _dataset_configs_by_name(config_dir: str | Path) -> dict[str, Config]:
    registry = load_dataset_registry(config_dir)
    out: dict[str, Config] = {}
    for cfg in registry.values():
        out[str(cfg.name)] = cfg
    return out


def _prediction_parts(path: Path, predictions_dir: Path) -> tuple[str, str, str] | None:
    rel_parts = path.relative_to(predictions_dir).parts
    if len(rel_parts) != 3:
        return None
    dataset, embedder, head_file = rel_parts
    return dataset, embedder, Path(head_file).stem


def prediction_artifacts_to_praski_frame(
    predictions_dir: str | Path,
    *,
    config_dir: str | Path | None = None,
    library_hash: str | int | None = None,
) -> pd.DataFrame:
    """Build a Praski-schema result table from score.py .npz predictions.

    Prediction artifacts contain held-out labels and scores, but not selected
    hyperparameters or cross-validation scores. Those fields are therefore
    emitted as blank values in the returned schema.
    """
    predictions_dir = Path(predictions_dir)
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "config"

    dataset_configs = _dataset_configs_by_name(config_dir)
    resolved_library_hash = (
        str(library_hash) if library_hash is not None else get_model_version_hash()
    )

    rows: list[dict[str, Any]] = []
    for path in find_prediction_npz(predictions_dir):
        parts = _prediction_parts(path, predictions_dir)
        if parts is None:
            continue

        dataset, embedder, model = parts
        if dataset not in dataset_configs:
            raise ValueError(f"No dataset config found for prediction artifact dataset {dataset!r}")

        dataset_config = dataset_configs[dataset]
        with np.load(path) as data:
            y_true = data["y_true"]
            y_score = data["y_score"]

        metric_name = str(dataset_config.metric)
        rows.append(
            {
                "dataset": dataset,
                "task": dataset_config.task,
                "embedder": embedder,
                "model": model,
                "hyperparams": pd.NA,
                "library_hash": resolved_library_hash,
                "cv_metric_name": metric_name,
                "cv_metric": pd.NA,
                "test_metric_name": metric_name,
                "test_metric": score_prediction_artifact(y_true, y_score, metric_name),
            }
        )

    frame = to_praski_schema(pd.DataFrame(rows))
    if frame.empty:
        return pd.DataFrame(columns=PRASKI_COLUMNS)
    frame["id"] = range(1, len(frame) + 1)
    return frame


def write_prediction_praski_csv(
    predictions_dir: str | Path,
    output_csv: str | Path,
    *,
    config_dir: str | Path | None = None,
    library_hash: str | int | None = None,
) -> pd.DataFrame:
    frame = prediction_artifacts_to_praski_frame(
        predictions_dir,
        config_dir=config_dir,
        library_hash=library_hash,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export score.py .npz prediction artifacts to the Praski CSV schema."
    )
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/prediction_results.csv"))
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "config",
        help="Benchmark config directory containing datasets.yaml.",
    )
    parser.add_argument(
        "--library-hash",
        default=None,
        help="Optional library hash override. Defaults to the current scoring-grid digest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = write_prediction_praski_csv(
        predictions_dir=args.predictions_dir,
        output_csv=args.output_csv,
        config_dir=args.config_dir,
        library_hash=args.library_hash,
    )
    print(f"Wrote {len(frame)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
