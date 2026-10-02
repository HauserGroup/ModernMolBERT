#!/usr/bin/env python3
"""Export five-seed dataset coverage and model-specific exclusion source data."""

import argparse
import csv
import json
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.supervised.const import (
    PRODUCTION_CV_POLICY,
)
from modernmolbert.utils import file_sha256, get_git_revision

SEEDS = (42, 43, 44, 45, 46)
COHORT_KEYS = (
    "prepared_sha256",
    "source_rows",
    "common_supervised_rows",
    "common_source_row_indices_sha256",
    "labels_sha256",
    "split_source_row_indices_sha256",
    "splits",
    "endpoint_viability",
)


def seed_paths(values: list[str]) -> dict[int, Path]:
    paths = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text.isdecimal() or not path_text:
            raise ValueError(f"Expected SEED=PATH, got {value!r}")
        seed = int(seed_text)
        if seed in paths:
            raise ValueError(f"Duplicate seed {seed}")
        paths[seed] = Path(path_text)
    if set(paths) != set(SEEDS):
        raise ValueError(f"Expected exactly seeds {SEEDS}")
    return paths


def export(
    summary_path: Path, training_path: Path, evaluation_paths: dict[int, Path]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary = {row["dataset"]: row for row in summary_rows}
    if len(summary) != 25 or len(summary_rows) != 25:
        raise ValueError("Expected exactly 25 distinct dataset summary rows")
    with training_path.open(newline="", encoding="utf-8") as handle:
        training_rows = list(csv.DictReader(handle))
    training_weights = {
        (row["run_id"], int(row["seed"])): row["final_model_sha256"] for row in training_rows
    }
    if len(training_weights) != 25 or len(training_rows) != 25:
        raise ValueError("Expected exactly 25 accepted training identities")

    manifests = {}
    for seed in SEEDS:
        manifest = json.loads(evaluation_paths[seed].read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != 2
            or manifest.get("seed", 42) != seed
            or manifest.get("cv") != PRODUCTION_CV_POLICY
            or manifest.get("missing_labels") != "as-negative"
            or len(manifest.get("run_ids", [])) != 5
            or set(manifest.get("tasks", {})) != set(summary)
        ):
            raise ValueError(f"Incomplete or incompatible evaluation manifest for seed {seed}")
        manifests[seed] = manifest

    reference = manifests[42]
    for seed in SEEDS[1:]:
        if set(manifests[seed]["run_ids"]) != set(reference["run_ids"]):
            raise ValueError(f"Model cohort differs at seed {seed}")
        for task in summary:
            current = manifests[seed]["tasks"][task]
            original = reference["tasks"][task]
            if any(current[key] != original[key] for key in COHORT_KEYS):
                raise ValueError(f"Supervised cohort differs for {task} seed {seed}")

    datasets = []
    for task in sorted(summary):
        raw = summary[task]
        accepted = reference["tasks"][task]
        raw_splits = {split: int(raw[f"n_{split}"]) for split in ("train", "valid", "test")}
        retained = accepted["splits"]
        if (
            int(raw["n_prepared"]) != accepted["source_rows"]
            or sum(raw_splits.values()) + int(raw["n_unassigned"]) != accepted["source_rows"]
            or any(retained[split] > raw_splits[split] for split in raw_splits)
            or sum(retained.values()) != accepted["common_supervised_rows"]
        ):
            raise ValueError(f"Raw and retained row counts disagree for {task}")
        viable = accepted["endpoint_viability"]
        scored = [record for record in viable.values() if record["roc_auc_defined"]]
        if len(viable) != int(raw["n_endpoints"]) or not scored:
            raise ValueError(f"Endpoint count or viability disagrees for {task}")
        datasets.append(
            {
                "dataset": task,
                "source": raw["source"],
                "split_rule": raw["split_rule"],
                "n_config_samples": int(raw["config_n_samples"]),
                "n_prepared": accepted["source_rows"],
                "n_unassigned": int(raw["n_unassigned"]),
                **{f"raw_{split}": raw_splits[split] for split in raw_splits},
                **{f"common_{split}": retained[split] for split in raw_splits},
                **{
                    f"excluded_{split}": raw_splits[split] - retained[split] for split in raw_splits
                },
                "common_supervised_rows": accepted["common_supervised_rows"],
                "n_test_endpoints": len(viable),
                "n_scored_test_endpoints": len(scored),
                "n_test_labelled_cells": int(raw["test_labelled_cells"]),
                "n_test_missing_cells": int(raw["test_missing_cells"]),
                "common_test_observed_cells": sum(
                    record["observed_test_rows"] for record in viable.values()
                ),
                "common_test_positive_cells": sum(
                    record["positive_test_rows"] for record in viable.values()
                ),
                "min_positive_per_scored_endpoint": min(
                    record["positive_test_rows"] for record in scored
                ),
                "prepared_sha256": accepted["prepared_sha256"],
                "common_source_row_indices_sha256": accepted["common_source_row_indices_sha256"],
            }
        )

    failures = []
    for seed in SEEDS:
        for task in sorted(summary):
            accepted = manifests[seed]["tasks"][task]
            for run_id in sorted(accepted["models"]):
                model = accepted["models"][run_id]
                pre_tokenization = (
                    accepted["source_rows"]
                    - model["source_retained_rows"]
                    - model["source_tokenization_failures"]
                    - model["source_over_context_rows"]
                )
                if (
                    pre_tokenization < 0
                    or model["common_retained_rows"] != accepted["common_supervised_rows"]
                    or model["source_retained_rows"] < model["common_retained_rows"]
                    or model["final_model_sha256"] != training_weights.get((run_id, seed))
                ):
                    raise ValueError(f"Inconsistent embedding counts for {task}/{run_id}/{seed}")
                failures.append(
                    {
                        "dataset": task,
                        "run_id": run_id,
                        "seed": seed,
                        "source_rows": accepted["source_rows"],
                        "source_retained_rows": model["source_retained_rows"],
                        "pre_tokenization_failures": pre_tokenization,
                        "tokenization_failures": model["source_tokenization_failures"],
                        "over_context_rows": model["source_over_context_rows"],
                        "common_retained_rows": model["common_retained_rows"],
                        "source_retained_not_in_common": model["source_retained_rows"]
                        - model["common_retained_rows"],
                        "final_model_sha256": model["final_model_sha256"],
                    }
                )
    if len(failures) != 625:
        raise ValueError("Expected 625 model/seed/task failure records")
    return datasets, failures


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-manifest", action="append", required=True, metavar="SEED=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation_paths = seed_paths(args.evaluation_manifest)
    datasets, failures = export(args.summary, args.training_manifest, evaluation_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "revision_dataset_coverage.csv": datasets,
        "revision_embedding_failures.csv": failures,
    }
    for name, rows in outputs.items():
        write_csv(args.output_dir / name, rows)
    (args.output_dir / "revision_dataset_coverage_manifest.json").write_text(
        json.dumps(
            {
                "code": get_git_revision(),
                "summary_sha256": file_sha256(args.summary),
                "training_manifest_sha256": file_sha256(args.training_manifest),
                "evaluation_manifest_sha256": {
                    str(seed): file_sha256(path) for seed, path in evaluation_paths.items()
                },
                "output_sha256": {name: file_sha256(args.output_dir / name) for name in outputs},
                "missing_labels": "as-negative",
                "cv": PRODUCTION_CV_POLICY,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
