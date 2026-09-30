"""Audit common supervised rows for the frozen 25-task factorial cohort."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.hf_upload import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml"
PREPARED = ROOT / "data/prepared"
EMBEDDED = ROOT / "data/embedded"
RUN_IDS = (
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
)


def row_hash(indices: set[int]) -> str:
    ordered = np.asarray(sorted(indices), dtype="<i8")
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def task_names() -> list[str]:
    configs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["datasets"]
    names = [entry["name"] for entry in configs.values() if entry["name"] != "ogbg-moltoxcast"]
    if len(names) != 25 or len(set(names)) != 25:
        raise ValueError("Expected exactly 25 distinct frozen paper tasks")
    return names


def label_coverage(labels: np.ndarray, task: str) -> list[dict[str, int]]:
    if labels.ndim == 1:
        labels = labels[:, None]
    coverage: list[dict[str, int]] = []
    for column in labels.T:
        finite = np.isfinite(column)
        record = {"finite": int(finite.sum())}
        if task == "classification":
            # The primary G7 policy treats missing labels as negative.
            values = np.nan_to_num(column, nan=0.0)
            record["positive"] = int(np.count_nonzero(values == 1))
            record["negative"] = int(np.count_nonzero(values == 0))
        coverage.append(record)
    return coverage


def inspect_task(name: str, prefix: str) -> dict[str, Any]:
    prepared_json = PREPARED / f"{name}.json"
    prepared = Dataset.deserialize_legacy(prepared_json)
    n_rows = len(prepared.data)
    prepared_sha = file_sha256(prepared_json)
    source_splits = {split: set(map(int, ids)) for split, ids in prepared.splits.items()}
    if set(source_splits) != {"train", "valid", "test"}:
        raise ValueError(f"Unexpected split names for {name}: {sorted(source_splits)}")
    if (
        set.union(*source_splits.values()) != set(range(n_rows))
        or sum(map(len, source_splits.values())) != n_rows
    ):
        raise ValueError(f"Source splits are not a complete partition for {name}")

    retained_sets: list[set[int]] = []
    model_coverage: dict[str, Any] = {}
    for run_id in RUN_IDS:
        path = EMBEDDED / name / f"{prefix}{run_id}.joblib"
        embedded = joblib.load(path)
        metadata = embedded.metadata
        retained = list(map(int, metadata["source_row_indices"]))
        failed = list(map(int, metadata["failed_source_row_indices"]))
        retained_set = set(retained)
        if (
            embedded.name != name
            or embedded.embedder != f"{prefix}{run_id}"
            or metadata.get("prepared_data_sha256") != prepared_sha
            or metadata.get("pooling") != "mean"
            or metadata.get("max_seq_length") != 384
            or int(metadata.get("source_n_rows", -1)) != n_rows
            or len(retained_set) != len(retained)
            or len(retained) != len(embedded.X)
        ):
            raise ValueError(f"Embedding identity or row provenance mismatch: {path}")
        if retained_set | set(failed) != set(range(n_rows)) or retained_set & set(failed):
            raise ValueError(f"Incomplete retained/failed source partition: {path}")
        if not np.isfinite(embedded.X).all():
            raise ValueError(f"Nonfinite retained embedding: {path}")
        retained_sets.append(retained_set)
        model_coverage[run_id] = {
            "embedding_sha256": file_sha256(path),
            "retained": len(retained),
            "failed": len(failed),
            "failed_source_row_indices": failed,
            "n_tokenization_failures": metadata.get("n_tokenization_failures"),
            "n_truncated": metadata.get("n_truncated"),
            "retained_by_split": {
                split: len(retained_set & ids) for split, ids in source_splits.items()
            },
        }

    common = set.intersection(*retained_sets)
    labels = prepared.labels.to_numpy(dtype=float)
    splits: dict[str, Any] = {}
    for split, source_ids in source_splits.items():
        common_ids = common & source_ids
        splits[split] = {
            "source_rows": len(source_ids),
            "common_rows": len(common_ids),
            "common_source_row_indices_sha256": row_hash(common_ids),
            "labels": label_coverage(labels[sorted(common_ids)], prepared.task),
        }
    return {
        "prepared_sha256": prepared_sha,
        "source_rows": n_rows,
        "common_rows": len(common),
        "common_source_row_indices_sha256": row_hash(common),
        "models": model_coverage,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder-prefix", default="PREFLIGHT_")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/audit/revision_factorial_v1/pilot_benchmark_coverage.json",
    )
    args = parser.parse_args()
    result = {
        "schema": 1,
        "embedder_prefix": args.embedder_prefix,
        "run_ids": RUN_IDS,
        "tasks": {name: inspect_task(name, args.embedder_prefix) for name in task_names()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['tasks'])} tasks to {args.output}")
    print(f"SHA-256 {file_sha256(args.output)}")


if __name__ == "__main__":
    main()
