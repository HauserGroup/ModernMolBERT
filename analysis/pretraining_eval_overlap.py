#!/usr/bin/env python3
"""Compute overlap between the ChEMBL pretraining set and evaluation datasets.

This script answers the question:
    "How much of our pretraining is in our evaluation sets? What percentage of overlap?"

An evaluation molecule counts as present in pretraining when either
    - its RDKit standard InChIKey matches a ChEMBL ``standard_inchi_key`` in the
      pretraining split, or
    - its canonical SMILES matches a pretraining ``smiles_canonical_clean`` string.

The InChIKey match is the primary criterion: the evaluation SMILES are canonicalized
with full RDKit sanitization while the pretraining SMILES were canonicalized with
partial sanitization, so identical molecules do not always produce identical
strings (tautomers, charge/nitro normalization, stereo notation). The SMILES match
catches the few cases where ChEMBL's InChIKey differs from the one RDKit computes.

Overlap is reported for the full evaluation dataset and for its test split, counted
over unique canonical SMILES.

Usage:
    uv run python analysis/pretraining_eval_overlap.py

The script expects prepared files in the repo's standard data locations:
    - data/pretrain/chembl36_selfies/train.parquet (or sharded train/*.parquet)
    - data/prepared/<dataset>.json or <dataset>.joblib (as written by
      modernmolbert.eval.benchmarking_molecular_models.download)

If prepared evaluation data are not available yet, run the benchmark download step first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = ROOT / "data" / "pretrain" / "chembl36_selfies"
PREPARED_DIR = ROOT / "data" / "prepared"
OUTPUT_PATH = ROOT / "analysis" / "pretraining_eval_overlap.csv"

PRETRAIN_COLUMNS = ["standard_inchi_key", "smiles_canonical_clean"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute overlap between ChEMBL pretraining and evaluation datasets."
    )
    parser.add_argument("--pretrain-dir", type=Path, default=PRETRAIN_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _clean_strings(series: pd.Series) -> set[str]:
    values = series.dropna().astype(str).str.strip()
    return set(values[values != ""])


def _pretraining_paths(pretrain_dir: Path) -> list[Path]:
    single = pretrain_dir / "train.parquet"
    if single.exists():
        return [single]
    shards = sorted((pretrain_dir / "train").glob("*.parquet"))
    if shards:
        return shards
    raise FileNotFoundError(
        f"Pretraining dataset not found at {single} or in {pretrain_dir / 'train'}. "
        "Prepare ChEMBL36 SELFIES data first."
    )


def _load_pretraining_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    inchikeys: set[str] = set()
    smiles: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=PRETRAIN_COLUMNS)
        inchikeys |= _clean_strings(frame.loc[:, "standard_inchi_key"])
        smiles |= _clean_strings(frame.loc[:, "smiles_canonical_clean"])
    if not inchikeys and not smiles:
        raise ValueError(f"No molecule identifiers found in {paths}.")
    return inchikeys, smiles


def _load_prepared_dataset(path: Path) -> Dataset:
    # Mirrors embed_modernmolbert.load_prepared_dataset: the legacy JSON is what the
    # embedding step reads, and it does not depend on pickled module paths.
    legacy_path = path.with_suffix(".json")
    if legacy_path.exists():
        return Dataset.deserialize_legacy(legacy_path)
    return joblib.load(path.with_suffix(".joblib"))


def _prepared_dataset_paths(prepared_dir: Path) -> list[Path]:
    stems = {p.with_suffix("") for p in prepared_dir.glob("*") if p.suffix in {".json", ".joblib"}}
    return sorted(stems)


_INCHIKEY_CACHE: dict[str, str | None] = {}


def _inchikey(smiles: str) -> str | None:
    if smiles not in _INCHIKEY_CACHE:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        key = Chem.MolToInchiKey(mol) if mol is not None else None
        _INCHIKEY_CACHE[smiles] = key or None
    return _INCHIKEY_CACHE[smiles]


def _overlap_counts(
    smiles: pd.Series, pretrain_inchikeys: set[str], pretrain_smiles: set[str]
) -> dict[str, int]:
    unique = pd.Series(sorted(_clean_strings(smiles)), dtype=object)
    inchikeys = unique.map(_inchikey)
    by_inchikey = inchikeys.isin(pretrain_inchikeys)
    by_smiles = unique.isin(pretrain_smiles)
    return {
        "n": len(unique),
        "n_overlap": int((by_inchikey | by_smiles).sum()),
        "n_overlap_inchikey": int(by_inchikey.sum()),
        "n_overlap_smiles": int(by_smiles.sum()),
    }


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _dataset_row(
    dataset: Dataset, pretrain_inchikeys: set[str], pretrain_smiles: set[str]
) -> dict[str, object]:
    df = dataset.data
    if "smiles" not in df.columns:
        raise ValueError(f"Dataset {dataset.name!r} has no 'smiles' column.")

    full = _overlap_counts(df["smiles"], pretrain_inchikeys, pretrain_smiles)
    test_idx = list(dataset.splits.get("test", []))
    test = _overlap_counts(df["smiles"].iloc[test_idx], pretrain_inchikeys, pretrain_smiles)

    return {
        "n_eval": full["n"],
        "n_overlap": full["n_overlap"],
        "overlap_pct": _pct(full["n_overlap"], full["n"]),
        "n_overlap_inchikey": full["n_overlap_inchikey"],
        "n_overlap_smiles": full["n_overlap_smiles"],
        "n_test": test["n"],
        "n_test_overlap": test["n_overlap"],
        "test_overlap_pct": _pct(test["n_overlap"], test["n"]),
    }


def main() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # type: ignore
    args = _parse_args()

    pretrain_paths = _pretraining_paths(args.pretrain_dir)
    pretrain_inchikeys, pretrain_smiles = _load_pretraining_keys(pretrain_paths)

    dataset_paths = _prepared_dataset_paths(args.prepared_dir)
    if not dataset_paths:
        raise FileNotFoundError(
            f"No prepared benchmark datasets found under {args.prepared_dir}. "
            "Run the benchmark download/preparation pipeline first."
        )

    rows: list[dict[str, object]] = []
    for path in dataset_paths:
        row: dict[str, object] = {"dataset": path.name}
        try:
            dataset = _load_prepared_dataset(path)
            row.update(_dataset_row(dataset, pretrain_inchikeys, pretrain_smiles))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["overlap_pct", "dataset"], ascending=[False, True])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    n_ok = sum("error" not in row for row in rows)
    print(
        json.dumps(
            {
                "pretraining_paths": [str(p) for p in pretrain_paths],
                "n_pretraining_inchikeys": len(pretrain_inchikeys),
                "n_pretraining_smiles": len(pretrain_smiles),
                "output_csv": str(args.output),
                "dataset_count": len(rows),
                "successful_overlap_checks": n_ok,
            },
            indent=2,
        )
    )
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
