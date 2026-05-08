import argparse
from pathlib import Path

from modernmolbert.eval.datasets import load_prepared_moleculenet_dataset
from modernmolbert.eval.featurizers.modernmolbert_selfies import (
    ModernMolBERTSelfiesFeaturizer,
)
from modernmolbert.eval.runner import FrozenBenchmarkRunner

from modernmolbert.eval.downstream import FrozenDownstreamConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a ModernMolBERT checkpoint as a frozen SELFIES featuriser "
            "through the shared benchmark runner."
        )
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help=(
            "Prepared dataset directory containing metadata.json and parquet splits, "
            "for example data/eval/moleculenet_sanitized/esol"
        ),
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer directory or vocabulary path. Defaults to --model_dir.",
    )
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")

    parser.add_argument(
        "--eval_split",
        choices=["valid", "test"],
        default="test",
    )
    parser.add_argument(
        "--smiles_column",
        default="smiles_canonical",
    )
    parser.add_argument(
        "--selfies_column",
        default="selfies",
    )
    parser.add_argument(
        "--merge_train_valid",
        action="store_true",
    )

    parser.add_argument("--cache_dir", default="eval_cache")
    parser.add_argument("--no_cache", action="store_true")

    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--ridge_cv", action="store_true")
    parser.add_argument("--classification_max_iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_prepared_moleculenet_dataset(
        dataset_dir=Path(args.dataset_dir),
        eval_split=args.eval_split,
        smiles_column=args.smiles_column,
        selfies_column=args.selfies_column,
        merge_train_valid=args.merge_train_valid,
    )

    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=Path(args.model_dir),
        tokenizer_path=Path(args.tokenizer_path)
        if args.tokenizer_path is not None
        else None,
        max_seq_length=args.max_seq_length,
        pooling=args.pooling,
        device=args.device,
    )

    downstream_config = FrozenDownstreamConfig(
        classification_max_iter=args.classification_max_iter,
        regression_alpha=args.ridge_alpha,
        use_ridge_cv=args.ridge_cv,
        random_state=args.seed,
        standardize=args.standardize,
    )

    runner = FrozenBenchmarkRunner(
        downstream_config=downstream_config,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        use_cache=not args.no_cache,
        batch_size=args.batch_size,
        random_state=args.seed,
    )

    result = runner.run(
        dataset=dataset,
        featurizer=featurizer,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
    )

    print(f"Wrote outputs to {args.output_dir}")

    print(
        f"Dataset: {result.dataset} | "
        f"Featurizer: {result.featurizer} | "
        f"Completed tasks: {len(result.task_results)} | "
        f"Skipped tasks: {len(result.skipped_tasks)}"
    )

    for task_result in result.task_results:
        print(f"  ✓ {task_result.task}: {task_result.metrics}")

    for skipped in result.skipped_tasks:
        print(f"  - skipped {skipped.task}: {skipped.reason}")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
