#!/usr/bin/env python3
"""Inventory the selected five-seed prediction archives for publication."""

import argparse
import csv
import hashlib
from pathlib import Path

MODELS = {
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
}
SEEDS = range(42, 47)
FIELDS = (
    "seed",
    "run_id",
    "dataset",
    "selected_head",
    "cv_roc_auc",
    "test_roc_auc",
    "n_predicted_test",
    "prepared_data_sha256",
    "final_model_sha256",
    "tokenizer_sha256",
    "prediction_path",
    "archive_sha256",
    "archive_size_bytes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--training-models", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, help="Verify selected archive bytes here.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identities = read_csv(args.training_models)
    models = {(int(row["seed"]), row["run_id"]): row for row in identities}
    if len(identities) != 25 or len(models) != 25:
        raise ValueError("Expected 25 distinct model/seed training identities")
    if set(models) != {(seed, model) for seed in SEEDS for model in MODELS}:
        raise ValueError("Training identities do not cover the frozen campaign")

    records = []
    seen = set()
    for seed in SEEDS:
        source = args.selection_root / f"seed{seed}_selected_heads" / "selected_heads.csv"
        if not source.exists():
            source = args.selection_root / f"seed{seed}" / "selected_heads" / "selected_heads.csv"
        selected = read_csv(source)
        native = [row for row in selected if row["archive_status"] == "ok"]
        external = [row for row in selected if row["archive_status"] == "table_only"]
        if len(selected) != 225 or len(native) != 125 or len(external) != 100:
            raise ValueError(f"Unexpected selection coverage in {source}")
        for row in native:
            model_dir = Path(row["embedding_model_dir"])
            run_id, seed_dir = model_dir.parts[-3:-1]
            if run_id not in MODELS or seed_dir != f"seed{seed}":
                raise ValueError(f"Unexpected model identity in {source}: {model_dir}")
            key = (seed, run_id, row["dataset"])
            if key in seen:
                raise ValueError(f"Duplicate selected archive: {key}")
            seen.add(key)
            if row["archive_prepared_sha256"] != row["prepared_sha256"]:
                raise ValueError(f"Prepared file hash mismatch: {key}")
            prediction_path = Path(row["prediction_path"])
            if prediction_path.is_absolute() or ".." in prediction_path.parts:
                raise ValueError(f"Unsafe archive path: {prediction_path}")
            size = ""
            if args.repo_root is not None:
                archive = args.repo_root / prediction_path
                if digest(archive) != row["archive_sha256"]:
                    raise ValueError(f"Prediction archive hash mismatch: {key}")
                size = str(archive.stat().st_size)
            training = models[(seed, run_id)]
            records.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "dataset": row["dataset"],
                    "selected_head": row["model"],
                    "cv_roc_auc": row["cv_metric"],
                    "test_roc_auc": row["test_metric"],
                    "n_predicted_test": int(float(row["n_predicted_test"])),
                    "prepared_data_sha256": row["prepared_sha256"],
                    "final_model_sha256": training["final_model_sha256"],
                    "tokenizer_sha256": training["tokenizer_sha256"],
                    "prediction_path": prediction_path.as_posix(),
                    "archive_sha256": row["archive_sha256"],
                    "archive_size_bytes": size,
                }
            )
    if len(seen) != 625 or len({row["prediction_path"] for row in records}) != 625:
        raise ValueError("Expected 625 unique model/seed/task prediction archives")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            sorted(records, key=lambda row: (row["seed"], row["run_id"], row["dataset"]))
        )
    print(f"Verified {len(records)} selected archives; byte checks: {args.repo_root is not None}")


if __name__ == "__main__":
    main()
