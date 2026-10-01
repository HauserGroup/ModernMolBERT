"""Write five aligned embedding cohorts from the frozen benchmark source rows."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
SPEC = ROOT / "configs/revision_factorial_v1.json"
RUN_IDS = tuple(json.loads(SPEC.read_text(encoding="utf-8"))["runs"])


def names() -> list[str]:
    configs = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    selected = [entry["name"] for entry in configs.values() if entry["name"] != "ogbg-moltoxcast"]
    if len(selected) != 25 or len(set(selected)) != 25:
        raise ValueError("Expected exactly 25 frozen paper tasks")
    return selected


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def label_hash(labels: pd.DataFrame) -> str:
    digest = hashlib.sha256(json.dumps(list(labels.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(labels, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def final_model_identity(source: EmbeddedDataset, run_id: str) -> dict[str, str]:
    model_dir = source.metadata.get("model_dir")
    if not model_dir:
        raise ValueError(f"Source embedding lacks model_dir for {run_id}")
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    expected = ROOT / "runs/revision_factorial_v1" / run_id / "seed42/final_model"
    if model_path.resolve() != expected.resolve():
        raise ValueError(f"Embedding uses an unexpected final model: {model_path}")
    identity_path = expected.parent / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    result = identity.get("result", {})
    weights = expected / result.get("final_model_file", "")
    if result.get("terminal_step") != 30_000 or not weights.is_file():
        raise ValueError(f"Incomplete final model for {run_id}")
    if file_sha256(weights) != result.get("final_model_sha256"):
        raise ValueError(f"Final weights changed for {run_id}")
    return {
        "run_identity_sha256": file_sha256(identity_path),
        "final_model_sha256": result["final_model_sha256"],
    }


def materialize_task(
    name: str,
    source_prefix: str,
    common_prefix: str,
    *,
    overwrite: bool,
    require_final_models: bool = False,
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
            or len(rows) != len(source.y)
            or len(set(rows)) != len(rows)
            or any(row < 0 or row >= len(prepared.data) for row in rows)
            or not np.isfinite(source.X).all()
        ):
            raise ValueError(f"Invalid source embedding identity or row map: {path}")
        expected_labels = prepared.labels.iloc[rows].reset_index(drop=True)
        if not source.y.reset_index(drop=True).equals(expected_labels):
            raise ValueError(f"Source embedding labels differ from the frozen task: {path}")
        if set(source.splits) != set(source_splits):
            raise ValueError(f"Source embedding split names differ from the frozen task: {path}")
        for split, prepared_rows in source_splits.items():
            positions = list(map(int, source.splits[split]))
            expected = set(rows) & prepared_rows
            if (
                len(set(positions)) != len(positions)
                or any(position < 0 or position >= len(rows) for position in positions)
                or {rows[position] for position in positions} != expected
            ):
                raise ValueError(f"Source embedding split differs from the frozen task: {path}")
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
    test_labels = labels.iloc[splits["test"]]
    endpoint_viability = {}
    for endpoint in test_labels.columns:
        observed = pd.to_numeric(test_labels[endpoint], errors="coerce").dropna()
        positives = int((observed == 1).sum())
        negatives = int((observed == 0).sum())
        endpoint_viability[endpoint] = {
            "observed_test_rows": len(observed),
            "positive_test_rows": positives,
            "negative_test_rows": negatives,
            "roc_auc_defined": positives > 0 and negatives > 0,
        }
    manifest: dict[str, Any] = {
        "prepared_sha256": prepared_sha,
        "source_rows": len(prepared.data),
        "common_supervised_rows": len(ordered_common),
        "common_source_row_indices_sha256": index_hash(ordered_common),
        "labels_sha256": label_hash(labels),
        "split_source_row_indices_sha256": {
            split: index_hash([ordered_common[position] for position in positions])
            for split, positions in splits.items()
        },
        "splits": {split: len(rows) for split, rows in splits.items()},
        "endpoint_viability": endpoint_viability,
        "models": {},
    }

    for run_id in RUN_IDS:
        source = sources[run_id]
        model_record = final_model_identity(source, run_id) if require_final_models else {}
        positions = [source_maps[run_id][row] for row in ordered_common]
        metadata = {
            key: source.metadata[key]
            for key in (
                "pooling",
                "pooling_special_tokens_excluded",
                "model_dir",
                "tokenizer_path",
                "max_seq_length",
            )
            if key in source.metadata
        }
        metadata.update(
            {
                "common_row_policy": "five_model_supervised_intersection_v1",
                "source_row_indices": ordered_common,
                "common_source_row_indices_sha256": manifest["common_source_row_indices_sha256"],
                "prepared_data_sha256": prepared_sha,
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
            "source_retained_rows": len(source_maps[run_id]),
            "common_retained_rows": len(ordered_common),
            "source_tokenization_failures": int(source.metadata.get("n_tokenization_failures", 0)),
            "source_over_context_rows": int(source.metadata.get("n_truncated", 0)),
            **model_record,
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", default="REVISION_")
    parser.add_argument("--common-prefix", default="REVISION_COMMON_")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        default=ROOT / "outputs/revision_factorial_v1/campaign_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/revision_factorial_v1/evaluation_manifest.json",
    )
    args = parser.parse_args()
    if args.source_prefix == args.common_prefix:
        raise ValueError("Common embedder prefix must differ from source prefix")
    campaign = args.campaign_manifest
    if not campaign.is_file():
        raise FileNotFoundError(f"Stage the campaign manifest first: {campaign}")
    campaign_data = json.loads(campaign.read_text(encoding="utf-8"))
    if set(campaign_data.get("run_ids", [])) != set(RUN_IDS):
        raise ValueError("Campaign manifest does not contain the five expected models")
    result = {
        "schema": 2,
        "campaign_manifest_sha256": file_sha256(campaign),
        "code_commit": campaign_data["code_commit"],
        "source_prefix": args.source_prefix,
        "common_prefix": args.common_prefix,
        "run_ids": RUN_IDS,
        "cv": {"folds": 5, "shuffle": True, "seed": 0},
        "missing_labels": "as-negative",
        "tasks": {
            name: materialize_task(
                name,
                args.source_prefix,
                args.common_prefix,
                overwrite=args.overwrite,
                require_final_models=True,
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
