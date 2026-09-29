#!/usr/bin/env python3
"""Count test molecules shared between benchmark datasets and check the task families.

Related datasets are not independent evidence in a benchmark average. The task
families in ``task_families.yaml`` rest on shared assay sources; this
model-free audit checks them against molecule overlap. For every pair of
configured datasets it counts test-split molecules with the same first
InChIKey block (the molecular skeleton, without stereochemistry) and flags
whether the pair belongs to one family.

It fails if the families file names a dataset that is not configured.

Usage:
    uv run python scripts/paper/audit_task_overlap.py \\
        --output outputs/audit/benchmark_task_overlap.csv
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, RDLogger

from compute_bootstrap_cis import FAMILIES_PATH, read_task_families
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset

CONFIG = Path("src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--families", type=Path, default=FAMILIES_PATH)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def connectivity_keys(smiles: list[str]) -> set[str]:
    keys = set()
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is not None and (key := Chem.MolToInchiKey(mol)):
            keys.add(key.split("-")[0])
    return keys


def pair_overlap(test_keys: dict[str, set[str]], families: dict[str, str]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(test_keys, 2):
        shared = len(test_keys[a] & test_keys[b])
        smaller = min(len(test_keys[a]), len(test_keys[b]))
        rows.append(
            {
                "dataset_a": a,
                "dataset_b": b,
                "n_test_molecules_a": len(test_keys[a]),
                "n_test_molecules_b": len(test_keys[b]),
                "n_shared": shared,
                "shared_fraction_of_smaller": shared / smaller if smaller else float("nan"),
                "family_a": families.get(a, a),
                "family_b": families.get(b, b),
                "same_family": families.get(a, a) == families.get(b, b),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    args = parse_args(argv)
    names = [
        entry["name"] for entry in yaml.safe_load(args.config.read_text())["datasets"].values()
    ]
    families = read_task_families(args.families)
    if unknown := sorted(set(families) - set(names)):
        raise ValueError(f"Task families name datasets that are not configured: {unknown}")

    test_keys = {}
    for name in names:
        dataset = Dataset.deserialize_legacy(args.prepared_dir / f"{name}.json")
        smiles = dataset.data["smiles"].astype(str).iloc[list(dataset.splits["test"])]
        test_keys[name] = connectivity_keys(smiles.tolist())

    overlap = pair_overlap(test_keys, families)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.output, index=False)
    within = overlap.loc[overlap["same_family"], "shared_fraction_of_smaller"]
    across = overlap.loc[~overlap["same_family"], "shared_fraction_of_smaller"]
    print(f"{len(overlap)} dataset pairs; wrote {args.output}")
    print(f"Within families: shared fraction {within.min():.2f}-{within.max():.2f}")
    print(f"Across families: median {across.median():.3f}, max {across.max():.2f}")


if __name__ == "__main__":
    main()
