from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.data import (
    DEFAULT_DATASETS,
    select_dataset_configs,
)
from modernmolbert.eval.benchmarking_molecular_models.heads import (
    downstream_configs_for_heads,
    lightweight_parity_downstream_configs_for_heads,
)
from modernmolbert.eval.benchmarking_molecular_models.lightweight_parity import (
    run_lightweight_parity_suite,
)
from modernmolbert.eval.suite import (
    BenchmarkSuiteConfig,
    run_benchmark_suite,
    suite_config_from_dict,
)

Pooling = Literal["mean", "cls"]
ParityMode = Literal["none", "lightweight"]


def build_suite_config(
    *,
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    datasets: list[str] | None = None,
    dataset_catalog: str | Path | None = None,
    prepared_root: str | Path | None = "data/eval/moleculenet_sanitized",
    pooling: Pooling = "mean",
    heads: list[str] | None = None,
    batch_size: int = 64,
    embed_batch_size: int = 32,
    max_length: int = 256,
    device: str = "auto",
    seed: int = 13,
    use_cache: bool = True,
    eval_split: Literal["valid", "test"] = "test",
    parity: ParityMode = "none",
) -> BenchmarkSuiteConfig:
    """Build a ModernMolBERT-only suite for the existing eval runner."""

    model_path = Path(model_path)
    resolved_tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else model_path
    if parity not in {"none", "lightweight"}:
        raise ValueError("parity must be 'none' or 'lightweight'")

    downstream_models = (
        lightweight_parity_downstream_configs_for_heads(heads or ["auto"])
        if parity == "lightweight"
        else downstream_configs_for_heads(heads or ["auto"], seed=seed)
    )

    config = {
        "name": "modernmolbert_molecular_benchmark",
        "datasets": select_dataset_configs(
            datasets,
            catalog_path=dataset_catalog,
            prepared_root=prepared_root,
        ),
        "featurizers": [
            {
                "type": "modernmolbert_selfies",
                "name": model_path.name or "modernmolbert",
                "model_dir": str(model_path),
                "tokenizer_path": str(resolved_tokenizer_path),
                "max_seq_length": max_length,
                "pooling": pooling,
                "device": device,
                "batch_size": embed_batch_size,
            }
        ],
        "downstream_models": downstream_models,
        "seeds": [seed],
        "eval_split": eval_split,
        "batch_size": batch_size,
        "use_cache": use_cache,
    }
    return suite_config_from_dict(config)


def run_modernmolbert_benchmark(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    datasets: list[str] | None = None,
    dataset_catalog: str | Path | None = None,
    prepared_root: str | Path | None = "data/eval/moleculenet_sanitized",
    pooling: Pooling = "mean",
    heads: list[str] | None = None,
    batch_size: int = 64,
    embed_batch_size: int = 32,
    max_length: int = 256,
    device: str = "auto",
    seed: int = 13,
    use_cache: bool = True,
    eval_split: Literal["valid", "test"] = "test",
    parity: ParityMode = "none",
) -> pd.DataFrame:
    """Run the focused ModernMolBERT frozen-representation benchmark."""

    output_dir = Path(output_dir)
    suite = build_suite_config(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        datasets=datasets,
        dataset_catalog=dataset_catalog,
        prepared_root=prepared_root,
        pooling=pooling,
        heads=heads,
        batch_size=batch_size,
        embed_batch_size=embed_batch_size,
        max_length=max_length,
        device=device,
        seed=seed,
        use_cache=use_cache,
        eval_split=eval_split,
        parity=parity,
    )

    if parity == "lightweight":
        results = run_lightweight_parity_suite(
            suite=suite,
            output_dir=output_dir,
            cache_dir=output_dir / "embeddings",
            heads=heads or ["auto"],
        )
    else:
        results = run_benchmark_suite(
            suite=suite,
            output_dir=output_dir,
            cache_dir=output_dir / "embeddings",
            write_single_run_outputs=False,
        )

    _write_jsonl(results, output_dir / "results.jsonl")
    _write_summary(results, output_dir / "summary.csv")
    _write_run_config(
        output_dir / "run_config.json",
        {
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path)
            if tokenizer_path is not None
            else str(model_path),
            "datasets": datasets or DEFAULT_DATASETS,
            "dataset_catalog": str(dataset_catalog) if dataset_catalog is not None else None,
            "prepared_root": str(prepared_root) if prepared_root is not None else None,
            "pooling": pooling,
            "heads": heads or ["auto"],
            "batch_size": batch_size,
            "embed_batch_size": embed_batch_size,
            "max_length": max_length,
            "device": device,
            "seed": seed,
            "use_cache": use_cache,
            "eval_split": eval_split,
            "parity": parity,
        },
    )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a trained ModernMolBERT checkpoint on frozen molecular tasks.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Dataset names from datasets.yaml. Defaults to: {' '.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument("--dataset-catalog", type=Path, default=None)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/eval/moleculenet_sanitized"),
        help="Root containing prepared MoleculeNet dataset directories.",
    )
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["auto"],
        help=(
            "Lightweight heads: auto, logreg, logistic_regression, rf, "
            "random_forest_classifier, ridge, ridge_cv, random_forest_regressor."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-split", choices=["valid", "test"], default="test")
    parser.add_argument(
        "--parity",
        choices=["none", "lightweight"],
        default="none",
        help=(
            "Use an opt-in parity mode. 'lightweight' matches the lightweight "
            "classification scoring/model-selection path and rejects regression datasets."
        ),
    )
    parser.add_argument("--no-cache", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_modernmolbert_benchmark(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        datasets=args.datasets,
        dataset_catalog=args.dataset_catalog,
        prepared_root=args.prepared_root,
        pooling=args.pooling,
        heads=args.heads,
        batch_size=args.batch_size,
        embed_batch_size=args.embed_batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
        use_cache=not args.no_cache,
        eval_split=args.eval_split,
        parity=args.parity,
    )

    print(f"Wrote results: {args.output_dir / 'results.csv'}", flush=True)
    print(f"Wrote JSONL: {args.output_dir / 'results.jsonl'}", flush=True)
    print(f"Wrote summary: {args.output_dir / 'summary.csv'}", flush=True)
    if not results.empty:
        columns = [
            col
            for col in ["dataset", "task", "downstream_name", "roc_auc", "rmse", "mae"]
            if col in results.columns
        ]
        print(results[columns].to_string(index=False), flush=True)


def _write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in frame.to_dict(orient="records"):
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_summary(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        frame.to_csv(path, index=False)
        return

    rows: list[pd.Series] = []
    for _, group in frame.groupby(["dataset", "task"], dropna=False):
        if "roc_auc" in group.columns and group["roc_auc"].notna().any():
            idx = group["roc_auc"].astype(float).idxmax()
        elif "rmse" in group.columns and group["rmse"].notna().any():
            idx = group["rmse"].astype(float).idxmin()
        elif "mae" in group.columns and group["mae"].notna().any():
            idx = group["mae"].astype(float).idxmin()
        else:
            idx = group.index[0]
        rows.append(frame.loc[idx])

    pd.DataFrame(rows).reset_index(drop=True).to_csv(path, index=False)


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isnan(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
