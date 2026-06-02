#!/usr/bin/env python3
"""Find and rerun missing embedding artifacts for a chosen embedder.

Default target is modernmolbert_best_hetero_span because it often has partial
coverage when earlier embedding runs were interrupted.

Usage:
  Preview only:
    uv run python analysis/validation/rerun_missing_embeddings.py

  Execute reruns:
    uv run python analysis/validation/rerun_missing_embeddings.py --run
"""

import argparse
import shlex
import subprocess
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "src/modernmolbert").exists():
            return path
    raise RuntimeError("Could not locate ModernMolBERT repo root")


def dataset_selector(dataset: str, dataset_config_text: str) -> str | None:
    clf = f"clf_{dataset}"
    if f"  {clf}:" in dataset_config_text:
        return clf

    reg = f"reg_{dataset}"
    if f"  {reg}:" in dataset_config_text:
        return reg

    if f"  {dataset}:" in dataset_config_text:
        return dataset

    return None


def embedding_path(root: Path, dataset: str, embedder: str) -> Path:
    return root / "data/embedded" / dataset / f"{embedder}.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun missing embedding jobs for one embedder.")
    parser.add_argument("--embedder", default="modernmolbert_best_hetero_span")
    parser.add_argument(
        "--model-dir",
        default="runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_hetero_span",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument(
        "--best-csv",
        default="outputs/eval/best_metric_by_dataset_embedder.csv",
    )
    parser.add_argument(
        "--dataset-config",
        default="src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=["ogbg-moltoxcast"],
        help="Datasets to skip entirely.",
    )
    parser.add_argument("--run", action="store_true", help="Execute commands instead of preview.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failures when --run is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_repo_root()

    best_csv = root / args.best_csv
    if not best_csv.exists():
        raise FileNotFoundError(best_csv)

    dataset_config = root / args.dataset_config
    if not dataset_config.exists():
        raise FileNotFoundError(dataset_config)

    dataset_config_text = dataset_config.read_text()

    model_dir = root / args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    tokenizer_path = root / args.tokenizer_path if args.tokenizer_path else model_dir

    best = pd.read_csv(best_csv)
    datasets = sorted(str(x) for x in best["dataset"].dropna().unique())
    excluded = set(args.exclude_datasets)

    missing: list[tuple[str, str]] = []
    selector_missing: list[str] = []

    for dataset in datasets:
        if dataset in excluded:
            continue

        selector = dataset_selector(dataset, dataset_config_text)
        if selector is None:
            selector_missing.append(dataset)
            continue

        if not embedding_path(root, dataset, args.embedder).exists():
            missing.append((dataset, selector))

    print(f"repo: {root}")
    print(f"embedder: {args.embedder}")
    print(f"datasets in benchmark CSV: {len(datasets)}")
    print(f"excluded datasets: {sorted(excluded)}")
    print(f"missing embeddings to rerun: {len(missing)}")

    if selector_missing:
        print("\nNo dataset selector found, skipping:")
        for dataset in selector_missing:
            print(f"  - {dataset}")

    if not missing:
        print("\nNo missing embeddings found.")
        return 0

    failures: list[tuple[str, int]] = []

    print("\nCommands:")
    for dataset, selector in missing:
        cmd = [
            "uv",
            "run",
            "python",
            str(
                root / "src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py"
            ),
            "--datasets",
            selector,
            "--model-dir",
            str(model_dir),
            "--tokenizer-path",
            str(tokenizer_path),
            "--embedder",
            args.embedder,
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--max-seq-length",
            str(args.max_seq_length),
            "--pooling",
            args.pooling,
        ]

        print(f"\n# {dataset}")
        print(shlex.join(cmd))

        if args.run:
            try:
                subprocess.run(cmd, cwd=root, check=True)
            except subprocess.CalledProcessError as exc:
                failures.append((dataset, exc.returncode))
                if not args.keep_going:
                    break

    if failures:
        print("\nFailed runs:")
        for dataset, code in failures:
            print(f"  - {dataset}: returncode={code}")
        return 1

    if args.run:
        print("\nAll rerun commands completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
