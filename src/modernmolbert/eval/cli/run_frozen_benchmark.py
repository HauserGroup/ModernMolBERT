import argparse
from pathlib import Path

from modernmolbert.eval.datasets import load_table_eval_dataset
from modernmolbert.eval.downstream import FrozenDownstreamConfig
from modernmolbert.eval.registry import make_featurizer_from_config
from modernmolbert.eval.runner import FrozenBenchmarkRunner


DEFAULT_RIDGE_CV_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single frozen-representation benchmark."
    )

    # Dataset
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
    parser.add_argument(
        "--train_path",
        "--train_csv",
        dest="train_path",
        required=True,
        help="Train split table path. Supports CSV, TSV, TXT, Parquet, and PQ.",
    )
    parser.add_argument(
        "--valid_path",
        "--valid_csv",
        dest="valid_path",
        default=None,
        help="Optional validation split table path.",
    )
    parser.add_argument(
        "--test_path",
        "--test_csv",
        dest="test_path",
        required=True,
        help="Test split table path. Supports CSV, TSV, TXT, Parquet, and PQ.",
    )
    parser.add_argument("--smiles_column", default="smiles")
    parser.add_argument("--selfies_column", default="selfies")
    parser.add_argument("--eval_split", choices=["valid", "test"], default="test")

    # Featurizer
    parser.add_argument(
        "--featurizer_config",
        required=True,
        help="JSON featurizer config with a required 'type' field.",
    )

    # Outputs/cache
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="eval_cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)

    # Downstream model
    parser.add_argument(
        "--downstream_model",
        default="auto",
        help=(
            "Downstream model type. For classification: auto, "
            "logistic_regression, random_forest_classifier. For regression: "
            "auto, ridge, ridge_cv, random_forest_regressor."
        ),
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=13)

    # Logistic regression options
    parser.add_argument("--classification_max_iter", type=int, default=5000)
    parser.add_argument("--classification_class_weight", default="balanced")
    parser.add_argument("--classification_C", type=float, default=1.0)
    parser.add_argument("--classification_solver", default="lbfgs")

    # Ridge/RidgeCV options
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument(
        "--ridge_cv_alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_RIDGE_CV_ALPHAS),
    )

    # Random forest options
    parser.add_argument("--rf_n_estimators", type=int, default=500)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    parser.add_argument("--rf_class_weight", default="balanced")

    return parser.parse_args()


def parse_task_names(text: str) -> list[str]:
    task_names = [item.strip() for item in text.split(",") if item.strip()]
    if not task_names:
        raise ValueError("--task_names must contain at least one task")
    return task_names


def resolve_downstream_model_type(args: argparse.Namespace) -> str:
    """Resolve 'auto' to the default downstream model for the task type."""

    if args.downstream_model != "auto":
        return str(args.downstream_model)

    if args.task_type == "classification":
        return "logistic_regression"

    if args.task_type == "regression":
        return "ridge"

    raise ValueError(f"Unsupported task_type: {args.task_type!r}")


def make_downstream_config_from_args(
    args: argparse.Namespace,
) -> FrozenDownstreamConfig:
    model_type = resolve_downstream_model_type(args)

    if args.task_type == "classification":
        if model_type == "logistic_regression":
            return FrozenDownstreamConfig(
                model_type="logistic_regression",
                params={
                    "max_iter": args.classification_max_iter,
                    "class_weight": args.classification_class_weight,
                    "C": args.classification_C,
                    "solver": args.classification_solver,
                },
                random_state=args.seed,
                standardize=args.standardize,
            )

        if model_type == "random_forest_classifier":
            return FrozenDownstreamConfig(
                model_type="random_forest_classifier",
                params={
                    "n_estimators": args.rf_n_estimators,
                    "max_depth": args.rf_max_depth,
                    "min_samples_leaf": args.rf_min_samples_leaf,
                    "class_weight": args.rf_class_weight,
                    "n_jobs": args.rf_n_jobs,
                },
                random_state=args.seed,
                standardize=False,
            )

        raise ValueError(f"Unsupported classification downstream model: {model_type!r}")

    if args.task_type == "regression":
        if model_type == "ridge":
            return FrozenDownstreamConfig(
                model_type="ridge",
                params={
                    "alpha": args.ridge_alpha,
                },
                random_state=args.seed,
                standardize=args.standardize,
            )

        if model_type == "ridge_cv":
            return FrozenDownstreamConfig(
                model_type="ridge_cv",
                params={
                    "alphas": tuple(args.ridge_cv_alphas),
                },
                random_state=args.seed,
                standardize=args.standardize,
            )

        if model_type == "random_forest_regressor":
            return FrozenDownstreamConfig(
                model_type="random_forest_regressor",
                params={
                    "n_estimators": args.rf_n_estimators,
                    "max_depth": args.rf_max_depth,
                    "min_samples_leaf": args.rf_min_samples_leaf,
                    "n_jobs": args.rf_n_jobs,
                },
                random_state=args.seed,
                standardize=False,
            )

        raise ValueError(f"Unsupported regression downstream model: {model_type!r}")

    raise ValueError(f"Unsupported task_type: {args.task_type!r}")


def main() -> None:
    args = parse_args()

    dataset = load_table_eval_dataset(
        name=args.name,
        task_type=args.task_type,
        task_names=parse_task_names(args.task_names),
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        smiles_column=args.smiles_column,
        selfies_column=args.selfies_column,
    )

    featurizer = make_featurizer_from_config(args.featurizer_config)
    downstream_config = make_downstream_config_from_args(args)

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

    if result.skipped_tasks:
        print("skipped tasks:")
        for skipped in result.skipped_tasks:
            print(f"  {skipped.task}: {skipped.reason}")

    for task_result in result.task_results:
        print(task_result.task, task_result.metrics)


if __name__ == "__main__":
    main()
