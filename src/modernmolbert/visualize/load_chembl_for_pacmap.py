# scripts/visualize_embeddings/load_chembl_for_umap.py

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROPERTY_COLUMNS = [
    "alogp",
    "psa",
    "hba",
    "hbd",
    "qed_weighted",
    "mw_freebase",
    "rtb",
    "aromatic_rings",
    "heavy_atoms",
    "num_ro5_violations",
]


def parquet_columns(path: str | Path) -> list[str]:
    """Return available column names without loading the full parquet file."""
    return pq.ParquetFile(path).schema.names


def load_chembl_selfies(
    parquet_path: str | Path,
    *,
    property_column: str,
    smiles_column: str = "smiles_canonical_clean",
    selfies_column: str = "selfies",
    id_column: str = "chembl_id",
    only_valid: bool = True,
    sample_size: int | None = None,
    seed: int = 13,
) -> pd.DataFrame:
    """Load a filtered ChEMBL SELFIES subset for embedding/UMAP visualization."""

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet file does not exist: {parquet_path}")

    required_columns = [
        id_column,
        smiles_column,
        selfies_column,
        property_column,
    ]

    optional_columns = [
        "split",
        "canonical_smiles",
        "standard_inchi_key",
        "is_valid",
    ]

    available_columns = parquet_columns(parquet_path)
    selected_columns = [
        column
        for column in dict.fromkeys(required_columns + optional_columns)
        if column in available_columns
    ]

    missing = sorted(set(required_columns) - set(selected_columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {parquet_path}: {missing}. "
            f"Available columns: {available_columns}"
        )

    df = pd.read_parquet(parquet_path, columns=selected_columns)

    if only_valid and "is_valid" in df.columns:
        df = df[df["is_valid"].astype(bool)]

    df = df.dropna(subset=[selfies_column, property_column])
    df = df[df[selfies_column].astype(str).str.len() > 0]

    df[property_column] = pd.to_numeric(df[property_column], errors="coerce")
    df = df.dropna(subset=[property_column])

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("--sample-size must be positive, or use 0 to disable sampling.")
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=seed)

    return df.reset_index(drop=True)


def summarize_loaded_data(
    df: pd.DataFrame,
    *,
    property_column: str,
) -> None:
    """Print a compact sanity-check summary."""

    print(f"Loaded rows: {len(df):,}")
    print(f"Property: {property_column}")
    print()

    print(df[property_column].describe())
    print()

    if "split" in df.columns:
        print("Split counts:")
        print(df["split"].value_counts(dropna=False))
        print()

    if "is_valid" in df.columns:
        print("Valid counts:")
        print(df["is_valid"].value_counts(dropna=False))
        print()

    preview_columns = [
        column
        for column in [
            "split",
            "chembl_id",
            "smiles_canonical_clean",
            "selfies",
            property_column,
            "standard_inchi_key",
        ]
        if column in df.columns
    ]

    print("Preview:")
    print(df[preview_columns].head())


def write_subset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Write subset as parquet or CSV based on file suffix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output suffix {suffix!r}. Use .parquet, .pq, or .csv.")

    print(f"\nWrote subset to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ChEMBL SELFIES parquet data for embedding/UMAP visualization."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/pretrain/chembl36_selfies/valid.parquet"),
        help=(
            "Input parquet file. Prefer valid.parquet/tvalid.parquet for visualization; "
            "use train.parquet only with --sample-size."
        ),
    )
    parser.add_argument(
        "--property",
        default="alogp",
        choices=PROPERTY_COLUMNS,
        help="Property column used later for coloring the UMAP.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Random subsample size. Use 0 to disable subsampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Do not filter is_valid == True.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path for the loaded subset: .parquet, .pq, or .csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size

    df = load_chembl_selfies(
        args.parquet,
        property_column=args.property,
        only_valid=not args.include_invalid,
        sample_size=sample_size,
        seed=args.seed,
    )

    summarize_loaded_data(df, property_column=args.property)

    if args.out is not None:
        write_subset(df, args.out)


if __name__ == "__main__":
    main()
