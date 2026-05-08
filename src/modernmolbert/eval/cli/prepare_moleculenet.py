import argparse
from pathlib import Path

from modernmolbert.eval.moleculenet import ALL_SPECS, CORE_SPECS, EXTENDED_SPECS
from modernmolbert.eval.moleculenet import prepare_many


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DeepChem/MoleculeNet datasets as local sanitized "
            "SMILES/SELFIES Parquet files."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(CORE_SPECS),
        choices=sorted(ALL_SPECS),
        help=(
            "Dataset names to prepare. Defaults to the core suite: "
            + ", ".join(CORE_SPECS)
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/eval/moleculenet_sanitized"),
        help="Directory where sanitized datasets will be written.",
    )
    parser.add_argument(
        "--deepchem_data_dir",
        type=Path,
        default=Path("data/deepchem/raw"),
        help="DeepChem raw data cache directory.",
    )
    parser.add_argument(
        "--deepchem_save_dir",
        type=Path,
        default=Path("data/deepchem/processed"),
        help="DeepChem processed/reload cache directory.",
    )
    parser.add_argument(
        "--split",
        default="scaffold",
        choices=["scaffold", "random", "index"],
        help="Local split to apply after sanitization.",
    )
    parser.add_argument(
        "--keep_invalid",
        action="store_true",
        help="Keep rows that fail RDKit parsing or SELFIES conversion.",
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List supported datasets and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for local random/scaffold splitting.",
    )
    parser.add_argument(
        "--frac_train",
        type=float,
        default=0.8,
        help="Fraction of rows assigned to the training split.",
    )
    parser.add_argument(
        "--frac_valid",
        type=float,
        default=0.1,
        help="Fraction of rows assigned to the validation split.",
    )
    parser.add_argument(
        "--frac_test",
        type=float,
        default=0.1,
        help="Fraction of rows assigned to the test split.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.keep_invalid and args.split == "scaffold":
        raise SystemExit(
            "--keep_invalid cannot be used with --split scaffold because invalid "
            "molecules do not have meaningful scaffolds. Use --split random/index "
            "or omit --keep_invalid."
        )

    if args.list_datasets:
        print("Core datasets:")
        for name, spec in CORE_SPECS.items():
            print(f"  {name:14s} {spec.task_type:14s} metric={spec.preferred_metric}")

        print("\nExtended datasets:")
        for name, spec in EXTENDED_SPECS.items():
            print(f"  {name:14s} {spec.task_type:14s} metric={spec.preferred_metric}")

        return

    prepare_many(
        dataset_names=args.datasets,
        output_root=args.output_root,
        deepchem_data_dir=args.deepchem_data_dir,
        deepchem_save_dir=args.deepchem_save_dir,
        split=args.split,
        seed=args.seed,
        frac_train=args.frac_train,
        frac_valid=args.frac_valid,
        frac_test=args.frac_test,
        keep_invalid=args.keep_invalid,
    )


if __name__ == "__main__":
    main()
