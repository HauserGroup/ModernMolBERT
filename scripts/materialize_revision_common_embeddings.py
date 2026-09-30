"""Write five aligned embedding cohorts from the frozen benchmark source rows."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from modernmolbert.eval.benchmarking_molecular_models.common.types import (
    Dataset,
    EmbeddedDataset,
)
from modernmolbert.hf_upload import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml"
PREPARED = ROOT / "data/prepared"
EMBEDDED = ROOT / "data/embedded"
RUN_IDS = (
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
)


def names() -> list[str]:
    configs = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    selected = [entry["name"] for entry in configs.values() if entry["name"] != "ogbg-moltoxcast"]
    if len(selected) != 25 or len(set(selected)) != 25:
        raise ValueError("Expected exactly 25 frozen paper tasks")
    return selected


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def materialize_task(
    name: str,
    source_prefix: str,
    common_prefix: str,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    prepared_path = PREPARED / f"{name}.json"
    prepared = Dataset.deserialize_legacy(prepared_path)
    prepared_sha = file_sha256(prepared_path)
    source_splits = {split: set(map(int, rows)) for split, rows in prepared.splits.items()}
    if set(source_splits) != {"train", "valid", "test"}:
        raise ValueError(f"Unexpected split names for {name}")
    supervised = set.union(*source_splits.values())
    if sum(map(len, source_splits.values())) != len(supervised):
        raise ValueError(f"Overlapping supervised splits for {name}")

    sources: dict[str, EmbeddedDataset] = {}
    source_maps: dict[str, dict[int, int]] = {}
    common = supervised.copy()
    for run_id in RUN_IDS:
        path = EMBEDDED / name / f"{source_prefix}{run_id}.joblib"
        source: EmbeddedDataset = joblib.load(path)
        rows = list(map(int, source.metadata["source_row_indices"]))
        if (
            source.name != name
            or source.metadata.get("prepared_data_sha256") != prepared_sha
            or source.metadata.get("pooling") != "mean"
            or source.metadata.get("max_seq_length") != 384
            or len(rows) != len(source.X)
            or len(set(rows)) != len(rows)
            or not np.isfinite(source.X).all()
        ):
            raise ValueError(f"Invalid source embedding identity or row map: {path}")
        sources[run_id] = source
        source_maps[run_id] = {row: position for position, row in enumerate(rows)}
        common &= set(rows)

    ordered_common = sorted(common)
    if not ordered_common:
        raise ValueError(f"No common supervised rows for {name}")
    splits = {
        split: [position for position, row in enumerate(ordered_common) if row in source_rows]
        for split, source_rows in source_splits.items()
    }
    if not splits["train"] or not splits["test"]:
        raise ValueError(f"Common cohort lost train or test rows for {name}")
    labels = prepared.labels.iloc[ordered_common].reset_index(drop=True)
    manifest: dict[str, Any] = {
        "prepared_sha256": prepared_sha,
        "source_rows": len(prepared.data),
        "common_supervised_rows": len(ordered_common),
        "common_source_row_indices_sha256": index_hash(ordered_common),
        "splits": {split: len(rows) for split, rows in splits.items()},
        "models": {},
    }

    for run_id in RUN_IDS:
        source = sources[run_id]
        positions = [source_maps[run_id][row] for row in ordered_common]
        metadata = dict(source.metadata)
        metadata.update(
            {
                "common_row_policy": "five_model_supervised_intersection_v1",
                "source_embedder": source.embedder,
                "source_embedding_failed_source_row_indices": source.metadata.get(
                    "failed_source_row_indices", []
                ),
                "source_n_rows": len(prepared.data),
                "source_row_indices": ordered_common,
                "failed_source_row_indices": sorted(set(range(len(prepared.data))) - common),
                "common_source_row_indices_sha256": manifest["common_source_row_indices_sha256"],
                "source_split_counts": {split: len(rows) for split, rows in source_splits.items()},
                "retained_split_counts": manifest["splits"],
                "prepared_data_sha256": prepared_sha,
                "n_inputs": len(prepared.data),
                "n_valid": len(ordered_common),
                "invalid_fraction": 1.0 - len(ordered_common) / len(prepared.data),
            }
        )
        common_embedding = EmbeddedDataset(
            name=name,
            task=prepared.task,
            embedder=f"{common_prefix}{run_id}",
            splits=splits,
            X=source.X[positions].copy(),
            y=labels.copy(),
            metadata=metadata,
        )
        output = EMBEDDED / name / f"{common_prefix}{run_id}.joblib"
        if output.exists() and not overwrite:
            raise FileExistsError(f"Common embedding already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(common_embedding, output)
        manifest["models"][run_id] = {
            "source_embedding_sha256": file_sha256(
                EMBEDDED / name / f"{source_prefix}{run_id}.joblib"
            ),
            "common_embedding_sha256": file_sha256(output),
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", default="PREFLIGHT_")
    parser.add_argument("--common-prefix", default="PREFLIGHT_COMMON_")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/audit/revision_factorial_v1/pilot_common_embeddings.json",
    )
    args = parser.parse_args()
    if args.source_prefix == args.common_prefix:
        raise ValueError("Common embedder prefix must differ from source prefix")
    result = {
        "schema": 1,
        "source_prefix": args.source_prefix,
        "common_prefix": args.common_prefix,
        "run_ids": RUN_IDS,
        "tasks": {
            name: materialize_task(
                name, args.source_prefix, args.common_prefix, overwrite=args.overwrite
            )
            for name in names()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['tasks'])} common task cohorts to {args.output}")
    print(f"SHA-256 {file_sha256(args.output)}")


if __name__ == "__main__":
    main()
