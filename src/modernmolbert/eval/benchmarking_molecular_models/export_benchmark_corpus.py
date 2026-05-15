#!/usr/bin/env python3
"""Export all unique SMILES (and their SELFIES) from prepared benchmark datasets.

Reads every .joblib file in the prepared directory and writes one tab-separated
line per unique SMILES:

    smiles<TAB>selfies

Lines where SELFIES conversion fails are written with an empty selfies column.

Example:
    uv run python -m modernmolbert.eval.benchmarking_molecular_models.export_benchmark_corpus \
        --prepared_dir data/prepared \
        --output data/prepared/benchmark_selfies_symbols_source.txt
"""

import argparse
from pathlib import Path

import joblib
import selfies as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared_dir",
        type=Path,
        default=Path("data/prepared"),
        help="Directory containing prepared .joblib dataset files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared/benchmark_selfies_symbols_source.txt"),
        help="Output file path.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split(s) to include. 'all' includes every row regardless of split.",
    )
    return parser.parse_args()


def iter_smiles_for_split(dataset, split: str):
    """Yield SMILES strings for the requested split."""
    if split == "all" or "split" not in dataset.data.columns:
        yield from dataset.data["smiles"].dropna().astype(str)
    else:
        mask = dataset.data["split"].str.lower() == split
        yield from dataset.data.loc[mask, "smiles"].dropna().astype(str)


def main() -> None:
    args = parse_args()

    joblib_files = sorted(args.prepared_dir.glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"No .joblib files found in {args.prepared_dir}")

    unique_smiles: dict[str, str] = {}  # smiles -> selfies (or "")

    for path in joblib_files:
        dataset = joblib.load(path)
        if "smiles" not in dataset.data.columns:
            print(f"Skipping {path.name}: no 'smiles' column")
            continue

        for smi in iter_smiles_for_split(dataset, args.split):
            smi = smi.strip()
            if not smi or smi in unique_smiles:
                continue
            try:
                unique_smiles[smi] = sf.encoder(smi) or ""
            except Exception:
                unique_smiles[smi] = ""

        print(f"  {path.stem}: {len(unique_smiles)} unique SMILES so far")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("smiles\tselfies\n")
        for smi, sel in sorted(unique_smiles.items()):
            f.write(f"{smi}\t{sel}\n")

    print(f"\nWrote {len(unique_smiles)} unique SMILES to {args.output}")


if __name__ == "__main__":
    main()
