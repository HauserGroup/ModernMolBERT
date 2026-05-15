import argparse
from pathlib import Path

from modernmolbert.data.chembl36 import (
    ChemBL36SelfiesPrepConfig,
    prepare_chembl36_selfies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ChEMBL36 canonical SMILES as SELFIES pretraining data.",
    )

    parser.add_argument("--dataset_name", default="lukaskim/ChEMBL-36")
    parser.add_argument("--dataset_config", default="molecules")
    parser.add_argument("--split", default="train")
    parser.add_argument("--smiles_column", default="canonical_smiles")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/pretrain/chembl36_selfies"),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=0.01,
        help="Validation fraction used for MLM monitoring and checkpoint selection.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.0,
        help=(
            "Optional held-out pretraining test fraction. Defaults to 0.0 because "
            "MLM pretraining usually only needs train/validation; downstream "
            "benchmarks provide the final test evaluation."
        ),
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--dedupe_column", default="standard_inchi_key")
    parser.add_argument("--min_heavy_atoms", type=int, default=3)
    parser.add_argument("--max_heavy_atoms", type=int, default=100)
    parser.add_argument("--max_mw", type=float, default=1000.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ChemBL36SelfiesPrepConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        smiles_column=args.smiles_column,
        output_dir=args.output_dir,
        seed=args.seed,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        max_rows=args.max_rows,
        dedupe_column=args.dedupe_column,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        max_mw=args.max_mw,
    )

    prepare_chembl36_selfies(config)


if __name__ == "__main__":
    main()
