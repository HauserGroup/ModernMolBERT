#!/usr/bin/env python3
"""Compute overlap between the ChEMBL pretraining set and evaluation datasets.

This script answers the question:
    "How much of our pretraining is in our evaluation sets? What percentage of overlap?"

It compares molecules by standard InChIKey when available and falls back to
canonical SMILES, which is the most stable representation for exact-molecule overlap.

Usage:
    uv run python analysis/pretraining_eval_overlap.py

The script expects prepared parquet files in the repo's standard data locations:
    - data/pretrain/chembl36_selfies/train.parquet
    - data/prepared/<dataset>.joblib or other prepared benchmark files

If prepared evaluation data are not available yet, run the benchmark download step first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = ROOT / "data" / "pretrain" / "chembl36_selfies"
PRETRAIN_PATH = PRETRAIN_DIR / "train.parquet"
PRETRAIN_SHARD_DIR = PRETRAIN_DIR / "train"
PREPARED_DIR = ROOT / "data" / "prepared"
OUTPUT_PATH = ROOT / "analysis" / "pretraining_eval_overlap.csv"


def _normalize_key_series(series: pd.Series) -> set[str]:
    values = series.dropna().astype(str)
    normalized = set()
    for value in values:
        cleaned = value.strip()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def _load_pretraining_frames() -> list[pd.DataFrame]:
    if PRETRAIN_PATH.exists():
        return [pd.read_parquet(PRETRAIN_PATH)]

    if PRETRAIN_SHARD_DIR.exists():
        shard_paths = sorted(PRETRAIN_SHARD_DIR.glob("*.parquet"))
        if shard_paths:
            return [pd.read_parquet(path) for path in shard_paths]

    raise FileNotFoundError(
        f"Pretraining dataset not found at {PRETRAIN_PATH} or in {PRETRAIN_SHARD_DIR}. "
        "Prepare ChEMBL36 SELFIES data first."
    )


def _load_pretraining_keys() -> set[str]:
    frames = _load_pretraining_frames()
    combined = pd.concat(frames, ignore_index=True)
    keys = set()
    for col in ["standard_inchi_key", "smiles_canonical_clean", "canonical_smiles"]:
        if col in combined.columns:
            keys |= _normalize_key_series(combined[col])
    if not keys:
        raise ValueError(
            "No molecule identifiers found in pretraining parquet. "
            "Expected standard_inchi_key or canonical SMILES columns."
        )
    return keys


def _load_benchmark_df(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix == ".joblib":
        import joblib

        obj = joblib.load(dataset_path)
        if isinstance(obj, dict):
            for key in ["data", "df", "frame"]:
                if key in obj and isinstance(obj[key], pd.DataFrame):
                    return obj[key]
            raise ValueError(f"Unsupported joblib object structure in {dataset_path}")
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, "data") and isinstance(obj.data, pd.DataFrame):
            return obj.data
        raise TypeError(f"Unsupported joblib object type for {dataset_path}: {type(obj)!r}")
    raise ValueError(f"Unsupported dataset file type: {dataset_path}")


def _extract_eval_keys(df: pd.DataFrame) -> set[str]:
    keys = set()
    for col in ["standard_inchi_key", "smiles_canonical_clean", "canonical_smiles", "smiles"]:
        if col in df.columns:
            keys |= _normalize_key_series(df[col])
    if not keys:
        raise ValueError(
            "No molecule identifiers found in evaluation data. "
            "Expected standard_inchi_key or a SMILES-like column."
        )
    return keys


def _iter_prepared_dataset_paths() -> Iterable[Path]:
    if not PREPARED_DIR.exists():
        return []
    return sorted(PREPARED_DIR.rglob("*"))


def _is_supported_dataset_file(path: Path) -> bool:
    return path.suffix.lower() in {".parquet", ".csv", ".joblib"}


def main() -> None:
    pretraining_keys = _load_pretraining_keys()

    rows: list[dict[str, object]] = []
    dataset_paths = [p for p in _iter_prepared_dataset_paths() if _is_supported_dataset_file(p)]

    if not dataset_paths:
        raise FileNotFoundError(
            f"No prepared benchmark datasets found under {PREPARED_DIR}. "
            "Run the benchmark download/preparation pipeline first."
        )

    for path in dataset_paths:
        try:
            df = _load_benchmark_df(path)
            eval_keys = _extract_eval_keys(df)
            overlap = pretraining_keys & eval_keys
            overlap_pct = (len(overlap) / len(eval_keys)) * 100.0 if eval_keys else 0.0
            rows.append(
                {
                    "dataset": path.parent.name + "/" + path.name,
                    "n_eval": len(eval_keys),
                    "n_overlap": len(overlap),
                    "overlap_pct": round(overlap_pct, 4),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": path.parent.name + "/" + path.name,
                    "n_eval": None,
                    "n_overlap": None,
                    "overlap_pct": None,
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows).sort_values(["overlap_pct", "dataset"], ascending=[False, True])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(json.dumps(
        {
            "pretraining_path": str(PRETRAIN_PATH),
            "output_csv": str(OUTPUT_PATH),
            "dataset_count": len(rows),
            "successful_overlap_checks": int(result["n_eval"].notna().sum()),
        },
        indent=2,
    ))
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
