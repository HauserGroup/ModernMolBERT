#!/usr/bin/env python3
"""Migrate frozen benchmark JSON datasets into current-module joblib files.

The legacy JSON contains the full row order and split indices and remains the
source of truth. This migration does not rerun downloads or split generation.
"""

import argparse
import json
import shutil
from pathlib import Path

import joblib

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.utils import file_sha256


def migrate(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    files = sorted(source.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No legacy dataset JSON files in {source}")

    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for legacy_json in files:
        dataset = Dataset.deserialize_legacy(legacy_json)
        if dataset.name != legacy_json.stem:
            raise ValueError(f"Dataset name mismatch: {legacy_json}")
        copied_json = destination / legacy_json.name
        shutil.copy2(legacy_json, copied_json)
        if file_sha256(copied_json) != file_sha256(legacy_json):
            raise ValueError(f"JSON copy changed: {legacy_json}")
        output_joblib = destination / f"{dataset.name}.joblib"
        joblib.dump(dataset, output_joblib)
        reloaded = joblib.load(output_joblib)
        if not isinstance(reloaded, Dataset):
            raise TypeError(f"Incorrect migrated type: {output_joblib}")
        if reloaded.splits != dataset.splits or not reloaded.data.equals(dataset.data):
            raise ValueError(f"Migrated dataset differs: {output_joblib}")
        records.append(
            {
                "dataset": dataset.name,
                "rows": len(dataset.data),
                "splits": {key: len(indices) for key, indices in dataset.splits.items()},
                "legacy_json_sha256": file_sha256(copied_json),
                "joblib_sha256": file_sha256(output_joblib),
            }
        )

    manifest = {
        "source": str(source),
        "destination": str(destination),
        "method": "Dataset.deserialize_legacy then joblib.dump; row order and splits preserved",
        "datasets": records,
    }
    (destination / "migration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = migrate(args.source, args.destination)
    print(f"Migrated {len(manifest['datasets'])} frozen benchmark datasets")


if __name__ == "__main__":
    main()
