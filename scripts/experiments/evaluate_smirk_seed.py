"""Run the isolated SMIRK embedding, matched-cohort, and scoring stages."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pandas.core.util.hashing import hash_pandas_object

from modernmolbert.eval.benchmarking_molecular_models.common.types import (
    Dataset,
    EmbeddedDataset,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.const import (
    PRODUCTION_CV_POLICY,
)
from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = Path(os.environ.get("SMIRK_TRAINING_ROOT", "/home/jakob/projects/ModernMolBERT"))
OUTPUT = ROOT / "outputs/experimental_smirk_v1"
RUN_IDS = (
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
)
SEEDS = (42, 43, 44, 45, 46)
CONFIG_DIR = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config"


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def label_hash(labels: pd.DataFrame) -> str:
    digest = hashlib.sha256(json.dumps(list(labels.columns)).encode("utf-8"))
    digest.update(hash_pandas_object(labels, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def reference_path(seed: int) -> Path:
    if seed == 42:
        return ROOT / "outputs/revision_factorial_v1/evaluation_manifest.json"
    return ROOT / f"outputs/revision_factorial_multiseed_v1/evaluation_seed{seed}.json"


def reference_prefix(seed: int) -> str:
    return "REVISION_COMMON_" if seed == 42 else f"REVISION_COMMON_s{seed}_"


def source_embedder(seed: int) -> str:
    return f"SMIRK_s{seed}_small_smirk_smiles"


def common_embedder(seed: int) -> str:
    return f"SMIRK_COMMON_s{seed}_small_smirk_smiles"


def matched_embedder(seed: int, run_id: str) -> str:
    return f"SMIRK_MATCHED_s{seed}_{run_id}"


def evaluation_path(seed: int) -> Path:
    return OUTPUT / f"evaluation_seed{seed}.json"


def verify_run(seed: int) -> tuple[Path, str, str]:
    campaign_path = TRAINING_ROOT / "outputs/experimental_smirk_v1/campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    run = TRAINING_ROOT / f"runs/experimental_smirk_v1/small_smirk_smiles/seed{seed}"
    identity_path = run / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    result = identity["result"]
    weights = run / "final_model" / result["final_model_file"]
    if (
        campaign["seeds"] != list(SEEDS)
        or identity["args"]["seed"] != seed
        or identity["args"]["tokenizer_algorithm"] != "SMIRK"
        or identity["args"]["representation"] != "SMILES"
        or identity["inputs"]["campaign_manifest_sha256"] != file_sha256(campaign_path)
        or identity["git"]["commit"] != campaign["code_commit"]
        or result["terminal_step"] != result["selected_step"]
        or result["terminal_step"] != 30_000
        or file_sha256(weights) != result["final_model_sha256"]
    ):
        raise ValueError(f"SMIRK seed {seed} does not match its completed campaign")
    return run / "final_model", result["final_model_sha256"], file_sha256(identity_path)


def verify_reference(seed: int) -> dict:
    path = reference_path(seed)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != 2
        or manifest.get("seed", 42) != seed
        or manifest.get("common_prefix") != reference_prefix(seed)
        or set(manifest.get("run_ids", [])) != set(RUN_IDS)
        or len(manifest.get("tasks", {})) != 25
        or manifest.get("cv") != PRODUCTION_CV_POLICY
        or manifest.get("missing_labels") != "as-negative"
    ):
        raise ValueError(f"Unexpected accepted reference evaluation: {path}")
    return manifest


def check_gpu_idle() -> None:
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    )
    if occupied.strip():
        raise RuntimeError(f"GPU in use before SMIRK embedding: {occupied.strip()}")


def embed(seed: int) -> None:
    model_dir, _, _ = verify_run(seed)
    verify_reference(seed)
    check_gpu_idle()
    command = [
        "/opt/lab/bin/uv",
        "run",
        "--locked",
        "--no-sync",
        "python",
        str(ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py"),
        "--datasets",
        "all",
        "--model-dir",
        str(model_dir),
        "--tokenizer-path",
        str(model_dir),
        "--embedder",
        source_embedder(seed),
        "--batch-size",
        "32",
        "--device",
        "cuda",
        "--pooling",
        "mean",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def save_embedding(path: Path, value: EmbeddedDataset) -> str:
    if path.exists():
        existing: EmbeddedDataset = joblib.load(path)
        if (
            existing.embedder != value.embedder
            or existing.metadata.get("source_row_indices") != value.metadata["source_row_indices"]
            or existing.metadata.get("model_weights_sha256")
            != value.metadata.get("model_weights_sha256")
        ):
            raise ValueError(f"Existing pilot cohort conflicts with current run: {path}")
        return file_sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)
    return file_sha256(path)


def verify_source_rows(source: EmbeddedDataset, prepared: Dataset) -> list[int]:
    rows = list(map(int, source.metadata["source_row_indices"]))
    if (
        len(rows) != len(source.X)
        or len(rows) != len(source.y)
        or len(set(rows)) != len(rows)
        or any(row < 0 or row >= len(prepared.data) for row in rows)
        or not np.isfinite(source.X).all()
        or not source.y.reset_index(drop=True).equals(
            prepared.labels.iloc[rows].reset_index(drop=True)
        )
        or set(source.splits) != set(prepared.splits)
    ):
        raise ValueError(f"Invalid embedded source rows or labels: {source.embedder}")
    for split, prepared_positions in prepared.splits.items():
        positions = list(map(int, source.splits[split]))
        if (
            len(set(positions)) != len(positions)
            or any(position < 0 or position >= len(rows) for position in positions)
            or {rows[position] for position in positions}
            != set(rows) & set(map(int, prepared_positions))
        ):
            raise ValueError(f"Invalid embedded split {split}: {source.embedder}")
    return rows


def subset(
    source: EmbeddedDataset, rows: list[int], name: str, prepared: Dataset
) -> EmbeddedDataset:
    source_rows = verify_source_rows(source, prepared)
    row_positions = {row: position for position, row in enumerate(source_rows)}
    if len(row_positions) != len(source_rows) or not set(rows) <= set(row_positions):
        raise ValueError(f"Cannot align {source.embedder} to frozen source rows")
    original_splits = {split: set(map(int, ids)) for split, ids in prepared.splits.items()}
    splits = {
        split: [position for position, row in enumerate(rows) if row in original]
        for split, original in original_splits.items()
    }
    if not splits["train"] or not splits["test"]:
        raise ValueError(f"Matched cohort lacks train or test rows: {prepared.name}")
    metadata = dict(source.metadata)
    metadata.update(
        {
            "source_row_indices": rows,
            "common_row_policy": "smirk_pilot_intersection_v1",
            "common_source_row_indices_sha256": index_hash(rows),
            "matched_source_embedder": source.embedder,
        }
    )
    return EmbeddedDataset(
        name=prepared.name,
        task=prepared.task,
        embedder=name,
        splits=splits,
        X=source.X[[row_positions[row] for row in rows]].copy(),
        y=prepared.labels.iloc[rows].reset_index(drop=True),
        metadata=metadata,
    )


def materialize(seed: int) -> None:
    _, weights_sha, identity_sha = verify_run(seed)
    reference = verify_reference(seed)
    registry = yaml.safe_load((CONFIG_DIR / "datasets.yaml").read_text(encoding="utf-8"))
    registry_names = {item["name"] for item in registry["datasets"].values()}
    registry_names.discard("ogbg-moltoxcast")
    if registry_names != set(reference["tasks"]):
        raise ValueError("Prepared benchmark registry differs from accepted reference")
    tasks = {}
    for task in sorted(registry_names):
        record = reference["tasks"][task]
        prepared_path = ROOT / "data/prepared" / f"{task}.json"
        if file_sha256(prepared_path) != record["prepared_sha256"]:
            raise ValueError(f"Prepared task changed: {task}")
        prepared = Dataset.deserialize_legacy(prepared_path)
        source_path = ROOT / "data/embedded" / task / f"{source_embedder(seed)}.joblib"
        source: EmbeddedDataset = joblib.load(source_path)
        source_rows = verify_source_rows(source, prepared)
        if (
            source.name != task
            or source.embedder != source_embedder(seed)
            or source.metadata.get("prepared_data_sha256") != record["prepared_sha256"]
            or source.metadata.get("model_weights_sha256") != weights_sha
            or source.metadata.get("pooling") != "mean"
            or source.metadata.get("max_seq_length") != 384
        ):
            raise ValueError(f"SMIRK source embedding identity or labels differ: {source_path}")
        reference_path_first = (
            ROOT / "data/embedded" / task / f"{reference_prefix(seed)}{RUN_IDS[0]}.joblib"
        )
        if (
            file_sha256(reference_path_first)
            != record["models"][RUN_IDS[0]]["common_embedding_sha256"]
        ):
            raise ValueError(f"Accepted reference embedding changed: {reference_path_first}")
        reference_first: EmbeddedDataset = joblib.load(reference_path_first)
        reference_rows = verify_source_rows(reference_first, prepared)
        if (
            len(reference_rows) != record["common_supervised_rows"]
            or index_hash(reference_rows) != record["common_source_row_indices_sha256"]
            or label_hash(reference_first.y.reset_index(drop=True)) != record["labels_sha256"]
        ):
            raise ValueError(f"Accepted reference cohort changed: {task}")
        retained = set(source_rows)
        matched_rows = [row for row in reference_rows if row in retained]
        missing_rows = [row for row in reference_rows if row not in retained]
        common_name = common_embedder(seed)
        common_path = ROOT / "data/embedded" / task / f"{common_name}.joblib"
        common_sha = save_embedding(
            common_path, subset(source, matched_rows, common_name, prepared)
        )
        model_records = {}
        if missing_rows:
            for run_id in RUN_IDS:
                original_path = (
                    ROOT / "data/embedded" / task / f"{reference_prefix(seed)}{run_id}.joblib"
                )
                expected = record["models"][run_id]["common_embedding_sha256"]
                if file_sha256(original_path) != expected:
                    raise ValueError(f"Accepted comparator embedding changed: {original_path}")
                original: EmbeddedDataset = joblib.load(original_path)
                if verify_source_rows(original, prepared) != reference_rows:
                    raise ValueError(f"Comparator source rows differ: {original_path}")
                matched_name = matched_embedder(seed, run_id)
                matched_path = ROOT / "data/embedded" / task / f"{matched_name}.joblib"
                model_records[run_id] = {
                    "accepted_embedding_sha256": expected,
                    "matched_embedding_sha256": save_embedding(
                        matched_path, subset(original, matched_rows, matched_name, prepared)
                    ),
                }
        tasks[task] = {
            "prepared_sha256": record["prepared_sha256"],
            "source_embedding_sha256": file_sha256(source_path),
            "common_embedding_sha256": common_sha,
            "accepted_rows": len(reference_rows),
            "matched_rows": len(matched_rows),
            "excluded_accepted_source_rows": missing_rows,
            "matched_source_row_indices_sha256": index_hash(matched_rows),
            "matched_labels_sha256": label_hash(
                prepared.labels.iloc[matched_rows].reset_index(drop=True)
            ),
            "comparators_rescored": model_records,
        }
        print(f"{task}: {len(matched_rows)}/{len(reference_rows)} paired rows", flush=True)
    result = {
        "schema": 1,
        "seed": seed,
        "training_run_identity_sha256": identity_sha,
        "final_model_sha256": weights_sha,
        "accepted_reference_evaluation_sha256": file_sha256(reference_path(seed)),
        "cv": PRODUCTION_CV_POLICY,
        "missing_labels": "as-negative",
        "heads": ["rf", "ridge", "knn"],
        "tasks": tasks,
    }
    output = evaluation_path(seed)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"Existing SMIRK evaluation manifest differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(f"Staged matched SMIRK evaluation: {output}", flush=True)


def score(seed: int, task: str, model: str, jobs: int) -> None:
    evaluation = json.loads(evaluation_path(seed).read_text(encoding="utf-8"))
    reference = verify_reference(seed)
    if (
        evaluation.get("seed") != seed
        or evaluation.get("accepted_reference_evaluation_sha256")
        != file_sha256(reference_path(seed))
        or task not in evaluation["tasks"]
        or task not in reference["tasks"]
    ):
        raise ValueError("SMIRK pilot evaluation manifest differs from reference")
    task_record = evaluation["tasks"][task]
    if model == "smirk":
        embedder = common_embedder(seed)
        expected = task_record["common_embedding_sha256"]
    elif model in task_record["comparators_rescored"]:
        embedder = matched_embedder(seed, model)
        expected = task_record["comparators_rescored"][model]["matched_embedding_sha256"]
    else:
        raise ValueError(f"No matched scoring required for {task}/{model}")
    embedding = ROOT / "data/embedded" / task / f"{embedder}.joblib"
    if file_sha256(embedding) != expected:
        raise ValueError(f"Matched embedding changed: {embedding}")
    registry = yaml.safe_load((CONFIG_DIR / "datasets.yaml").read_text(encoding="utf-8"))
    selectors = [key for key, item in registry["datasets"].items() if item["name"] == task]
    if len(selectors) != 1:
        raise ValueError(f"Expected exactly one dataset selector for {task}")
    output = OUTPUT / "scoring" / f"seed{seed}" / model / f"{task}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [
        "/opt/lab/bin/uv",
        "run",
        "--locked",
        "--no-sync",
        "python",
        str(ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/score.py"),
        "--datasets",
        selectors[0],
        "--embedder",
        embedder,
        "--heads",
        "rf",
        "ridge",
        "knn",
        "--missing-labels",
        "as-negative",
        "--resume",
        "--no-safe",
        "--n-jobs",
        str(jobs),
        "--output-csv",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["embed", "materialize", "score"])
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("task", nargs="?")
    parser.add_argument("model", nargs="?")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "embed":
        embed(args.seed)
    elif args.stage == "materialize":
        materialize(args.seed)
    elif args.task and args.model and args.n_jobs > 0:
        score(args.seed, args.task, args.model, args.n_jobs)
    else:
        parser.error("score requires TASK MODEL and positive --n-jobs")


if __name__ == "__main__":
    sys.exit(main())
