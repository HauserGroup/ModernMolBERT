import argparse
from pathlib import Path

from modernmolbert.eval.datasets import load_csv_eval_dataset
from modernmolbert.eval.downstream import FrozenDownstreamConfig
from modernmolbert.eval.registry import make_featurizer_from_config
from modernmolbert.eval.runner import FrozenBenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen-representation benchmark."
    )

    parser.add_argument("--name", required=True, help="Dataset/run name.")
    parser.add_argument(
        "--task_type",
        required=True,
        choices=["classification", "regression"],
    )
    parser.add_argument(
        "--task_names",
        required=True,
        help="Comma-separated label/task columns.",
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", default=None)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--smiles_column", default="smiles")

    parser.add_argument(
        "--featurizer_config",
        required=True,
        help="JSON config with at least {'type': ...}.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="eval_cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_split", choices=["valid", "test"], default="test")

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

    task_names = [x.strip() for x in args.task_names.split(",") if x.strip()]
    if not task_names:
        raise ValueError("--task_names must contain at least one task")

    dataset = load_csv_eval_dataset(
        name=args.name,
        task_type=args.task_type,
        task_names=task_names,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        smiles_column=args.smiles_column,
    )

    featurizer = make_featurizer_from_config(args.featurizer_config)

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

    print(f"wrote: {args.output_dir}")
    for task_result in result.task_results:
        print(task_result.task, task_result.metrics)


if __name__ == "__main__":
    main()
