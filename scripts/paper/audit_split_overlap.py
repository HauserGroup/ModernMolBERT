#!/usr/bin/env python3
"""Audit molecule identity shared by supervised train/valid and test splits.

The supervised head fits on ``train`` plus ``valid``. This script retains the
prepared test-row index so a later common-row sensitivity analysis can apply
one exclusion policy to every model. It reports exact canonical-isomeric,
standard InChIKey, and stereo-insensitive canonical-SMILES matches separately.

Usage:
    uv run python scripts/paper/audit_split_overlap.py \
        --summary-output outputs/audit/split_overlap_summary.csv \
        --row-output outputs/audit/split_overlap_test_rows.csv
"""

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, RDLogger

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.utils import file_sha256

CONFIG = Path("src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--row-output", type=Path, required=True)
    return parser.parse_args(argv)


@lru_cache(maxsize=300_000)
def molecule_ids(smiles: str) -> tuple[str | None, str | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        Chem.MolToInchiKey(mol) or None,
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
    )


def audit_dataset(dataset: Dataset, prepared_sha256: str) -> tuple[dict[str, object], list[dict]]:
    smiles = dataset.data["smiles"].astype(str)
    fit_rows = list(dataset.splits.get("train", [])) + list(dataset.splits.get("valid", []))
    test_rows = list(dataset.splits.get("test", []))
    if not fit_rows or not test_rows:
        raise ValueError(f"Missing training-side or test rows for {dataset.name}")

    fit_raw = {smiles.iloc[i] for i in fit_rows}
    fit_keys: tuple[set[str], set[str], set[str]] = (set(), set(), set())
    for raw in fit_raw:
        for keys, value in zip(fit_keys, molecule_ids(raw), strict=True):
            if value is not None:
                keys.add(value)

    records = []
    for row_index in test_rows:
        raw = smiles.iloc[row_index]
        isomeric, inchi_key, nonisomeric = molecule_ids(raw)
        same_isomeric = isomeric is not None and isomeric in fit_keys[0]
        same_inchikey = inchi_key is not None and inchi_key in fit_keys[1]
        same_nonisomeric = nonisomeric is not None and nonisomeric in fit_keys[2]
        records.append(
            {
                "dataset": dataset.name,
                "prepared_sha256": prepared_sha256,
                "test_source_row_index": int(row_index),
                "smiles": raw,
                "valid_smiles": isomeric is not None,
                "same_raw_smiles": raw in fit_raw,
                "same_isomeric_smiles": same_isomeric,
                "same_inchikey": same_inchikey,
                "same_nonisomeric_smiles": same_nonisomeric,
                "stereo_variant_only": same_nonisomeric and not same_isomeric,
            }
        )

    frame = pd.DataFrame(records)
    summary: dict[str, object] = {
        "dataset": dataset.name,
        "prepared_sha256": prepared_sha256,
        "n_fit_rows": len(fit_rows),
        "n_test_rows": len(test_rows),
        "n_invalid_test_smiles": int((~frame["valid_smiles"]).sum()),
    }
    for column in (
        "same_raw_smiles",
        "same_isomeric_smiles",
        "same_inchikey",
        "same_nonisomeric_smiles",
        "stereo_variant_only",
    ):
        summary[f"n_test_{column}"] = int(frame[column].sum())
    return summary, records


def main(argv: list[str] | None = None) -> None:
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text())["datasets"]
    summaries = []
    rows = []
    for entry in config.values():
        name = entry["name"]
        path = args.prepared_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summary, test_rows = audit_dataset(Dataset.deserialize_legacy(path), file_sha256(path))
        summaries.append(summary)
        rows.extend(test_rows)
    for output, frame in (
        (args.summary_output, pd.DataFrame(summaries)),
        (args.row_output, pd.DataFrame(rows)),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    print(f"Audited {len(summaries)} datasets and {len(rows)} test rows")


if __name__ == "__main__":
    main()
