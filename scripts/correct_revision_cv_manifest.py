"""Correct the seed-42 evaluation manifest's CV metadata without touching score files.

The completed production scorer used GridSearchCV(cv=5), which defaults to
unshuffled folds. An earlier manifest generator incorrectly recorded shuffled
folds. This script changes only that metadata field and keeps the original
manifest as an adjacent backup for provenance. Rerun seed-42 head selection
afterward because its selection manifest binds the evaluation-manifest hash.
"""

import argparse
import hashlib
import json
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.supervised.const import (
    PRODUCTION_CV_POLICY,
)

OLD_POLICY = {"folds": 5, "shuffle": True, "seed": 0}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def correct(path: Path, expected_sha256: str) -> tuple[str, str]:
    original = path.read_bytes()
    original_hash = sha256(original)
    if original_hash != expected_sha256:
        raise ValueError(f"Manifest hash differs from expected: {original_hash}")
    record = json.loads(original)
    if (
        record.get("schema") != 2
        or record.get("seed") != 42
        or len(record.get("run_ids", [])) != 5
        or len(record.get("tasks", {})) != 25
        or record.get("missing_labels") != "as-negative"
        or record.get("cv") != OLD_POLICY
    ):
        raise ValueError("Only the known seed-42 production manifest can be corrected")
    record["cv"] = PRODUCTION_CV_POLICY
    corrected = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    backup = path.with_name(path.stem + ".cv_metadata_original.json")
    with backup.open("xb") as handle:
        handle.write(original)
    temporary = path.with_name(path.name + ".cv_metadata_tmp")
    temporary.write_bytes(corrected)
    temporary.replace(path)
    return original_hash, sha256(corrected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    old_hash, new_hash = correct(args.manifest, args.expected_sha256)
    print(f"Corrected CV metadata only: {old_hash} -> {new_hash}")
    print("Rerun scripts/run_revision_multiseed_analysis.sh select-seed 42")


if __name__ == "__main__":
    main()
