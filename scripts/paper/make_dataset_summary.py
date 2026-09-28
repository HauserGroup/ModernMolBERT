#!/usr/bin/env python3
"""Summarise the prepared benchmark datasets for the supplementary dataset table.

For each dataset in the benchmark configuration, reports the split rule, the
number of endpoints, prepared row counts by split, and labelled, positive, and
missing label cells by split. Counts come from the prepared files that the
embedding and scoring steps read, not from the original source publications.

Per-model retained and failed rows are not included: take them from the
``n_predicted_test`` and ``test_coverage`` columns of
``build_common_row_benchmark.py`` output, which are measured per run.

Usage:
    uv run python scripts/paper/make_dataset_summary.py \\
        --output outputs/audit/benchmark_dataset_summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset

CONFIG = Path("src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml")
SPLIT_RULES = {
    ("TDC", "admet"): "TDC ADMET group train_val/test split",
    ("TDC", None): "Murcko scaffold split (create_tdc_scaffold_split); valid merged into train",
    ("OGB", None): "OGB scaffold split",
}
SPLITS = ("train", "valid", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise prepared benchmark datasets by split.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def summarise(dataset: Dataset) -> dict[str, object]:
    labels = dataset.labels.to_numpy(dtype=float)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    record: dict[str, object] = {
        "n_endpoints": labels.shape[1],
        "n_prepared": labels.shape[0],
    }
    assigned = 0
    for split in SPLITS:
        rows = np.asarray(dataset.splits.get(split, []), dtype=np.int64)
        assigned += len(rows)
        block = labels[rows]
        finite = np.isfinite(block)
        record[f"n_{split}"] = len(rows)
        record[f"{split}_labelled_cells"] = int(finite.sum())
        record[f"{split}_positive_cells"] = int((block[finite] == 1).sum())
        record[f"{split}_missing_cells"] = int((~finite).sum())
    record["n_unassigned"] = labels.shape[0] - assigned
    test_labelled = int(str(record["test_labelled_cells"]))
    test_positive = int(str(record["test_positive_cells"]))
    record["test_prevalence"] = test_positive / test_labelled if test_labelled else np.nan
    return record


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text())["datasets"]
    rows = []
    for key, entry in config.items():
        source = entry.get("source", {})
        rule = SPLIT_RULES.get((source.get("name"), source.get("benchmark")))
        path = args.prepared_dir / f"{entry['name']}.json"
        record: dict[str, object] = {
            "dataset": entry["name"],
            "config_key": key,
            "source": source.get("name"),
            "split_rule": rule or "unknown",
            "config_n_samples": entry.get("n_samples"),
        }
        if path.exists():
            record |= summarise(Dataset.deserialize_legacy(path))
        else:
            record["n_prepared"] = np.nan
        rows.append(record)
    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Summarised {len(summary)} datasets; wrote {args.output}")
    missing = summary.loc[summary["n_prepared"].isna(), "dataset"].tolist()
    if missing:
        print(f"No prepared file for: {missing}")


if __name__ == "__main__":
    main()
