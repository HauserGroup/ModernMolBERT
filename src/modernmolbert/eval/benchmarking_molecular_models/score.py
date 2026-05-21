import argparse
import logging as log
from pathlib import Path
from typing import Any

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    write_dataset_checkpoint,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
    load_embedding_config,
    load_yaml_config,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    EmbeddingConfig,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.models import (
    AVAILABLE_HEADS,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.procedure import (
    eval_procedure,
)


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a config value from either dict-like or attribute-like configs."""
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass

    return getattr(cfg, key, default)


def normalize_dataset_name(name: str | Path) -> str:
    """Normalize dataset references to config stems.

    Examples
    --------
    bace -> bace
    bace.yaml -> bace
    config/datasets/bace.yaml -> bace
    """
    return Path(str(name)).stem


def as_list(value: Any) -> list[Any]:
    """Return value as a list while treating None as an empty list."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return list(value)


def run_eval(
    *,
    safe: bool,
    embed_config: EmbeddingConfig,
    full_model_name: str,
    short_model_name: str,
    dataset_info,
    model_head: str,
    output_csv: Path,
    override: bool,
) -> None:
    """Run one dataset/head evaluation.

    `full_model_name` is kept for logging.
    `short_model_name` is passed to `eval_procedure`, because this is the
    name expected for the precomputed embedding files.
    """
    log.info(
        "Evaluating model %s on dataset %s with metric %s using head %s",
        full_model_name,
        dataset_info.name,
        dataset_info.metric,
        model_head,
    )

    if safe:
        log.info("Running in safe mode")

        try:
            eval_procedure(
                dataset_info=dataset_info,
                embedded_dir=embed_config.embedded_directory,
                predictions_dir=embed_config.predictions_directory,
                model_name=short_model_name,
                model_head=model_head,
                output_csv=output_csv,
                override=override,
            )
        except Exception as e:
            import traceback

            log.error("Error during evaluation: %s", e)
            log.error(traceback.format_exc())
            return

    else:
        eval_procedure(
            dataset_info=dataset_info,
            embedded_dir=embed_config.embedded_directory,
            predictions_dir=embed_config.predictions_directory,
            model_name=short_model_name,
            model_head=model_head,
            output_csv=output_csv,
            override=override,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing molecular embedding files.")

    parser.add_argument(
        "--embedder",
        "--model-name",
        dest="model_name",
        required=False,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset config stems, globs, or 'all'. Defaults to config/score.yaml.",
    )

    parser.add_argument(
        "--heads",
        nargs="+",
        choices=AVAILABLE_HEADS,
        default=AVAILABLE_HEADS,
        help="Scoring heads to run.",
    )

    parser.add_argument(
        "--skip_datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Skip one or more datasets by name. Accepts stems such as 'bace' "
            "or config filenames such as 'bace.yaml'. Defaults to "
            "skip_datasets in config/score.yaml."
        ),
    )

    parser.add_argument("--config-dir", default="config")

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/benchmark_results.csv"),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for per-dataset result checkpoint CSVs.",
    )

    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--safe",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Compatibility support for key=value overrides such as model_name=my_embedder.",
    )

    return parser.parse_args()


def resolve_model_name(cfg: Any, args: argparse.Namespace) -> str | None:
    """Resolve embedder/model name from CLI, compatibility overrides, or config."""
    model_name = args.model_name

    for override in args.overrides:
        if override.startswith("model_name="):
            model_name = override.split("=", 1)[1]

    model_cfg = cfg_get(cfg, "model", {})

    model_name = (
        model_name
        or cfg_get(model_cfg, "embedding_name", None)
        or cfg_get(cfg, "model_name", None)
        or cfg_get(model_cfg, "model_name", None)
    )

    return model_name


def resolve_dataset_names(config_dir: Path, cfg: Any, args: argparse.Namespace) -> list[str]:
    """Resolve selected datasets, then remove skipped datasets.

    Dataset selections and skip selections may come either from CLI or score.yaml.
    Skip selections are normalized to stems, so both `bace` and `bace.yaml`
    match expanded dataset name `bace`.
    """
    dataset_selections = args.datasets or cfg_get(cfg, "datasets", ["all"])
    dataset_names = expand_dataset_selection(config_dir, dataset_selections)

    skip_selections = (
        args.skip_datasets
        or cfg_get(cfg, "skip_datasets", None)
        or cfg_get(cfg, "skip_dataset", [])
    )

    skip_set = frozenset(normalize_dataset_name(name) for name in as_list(skip_selections))

    if skip_set:
        print(f"Skipping datasets: {sorted(skip_set)}", flush=True)

        dataset_names = [
            name for name in dataset_names if normalize_dataset_name(name) not in skip_set
        ]

    return dataset_names


def make_short_model_name(model_name: str) -> str:
    """Convert model/embedder path or identifier to the stored embedding name."""
    if "gpt" in model_name.lower():
        return model_name.split("/")[-1]

    return model_name.split("/")[-1].split(".")[0]


def main() -> None:
    log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()

    root = Path(__file__).resolve().parent
    config_dir = root / args.config_dir

    cfg = load_yaml_config(config_dir / "score.yaml")
    embedding_cfg = load_embedding_config(config_dir)

    model_name = resolve_model_name(cfg, args)

    if not model_name:
        raise ValueError(
            "Scoring requires --embedder <name>. "
            "Expected embeddings at data/embedded/<dataset>/<embedder>.joblib."
        )

    dataset_names = resolve_dataset_names(config_dir, cfg, args)

    cache = cfg_get(cfg, "cache", True) if args.cache is None else args.cache
    safe = cfg_get(cfg, "safe", False) if args.safe is None else args.safe

    override = not cache

    embed_config = EmbeddingConfig(**embedding_cfg)
    short_model_name = make_short_model_name(model_name)

    n_total = len(dataset_names)

    print(
        f"[score] embedder={short_model_name}  datasets={n_total}  override={override}",
        flush=True,
    )

    for idx, dataset_name in enumerate(dataset_names, start=1):
        dataset_info = load_dataset_config(config_dir, dataset_name)

        for model_head in args.heads:
            print(
                f"[{idx:>2}/{n_total}] {dataset_info.name}  head={model_head}",
                flush=True,
            )

            run_eval(
                safe=safe,
                embed_config=embed_config,
                full_model_name=model_name,
                short_model_name=short_model_name,
                dataset_info=dataset_info,
                model_head=model_head,
                output_csv=args.output_csv,
                override=override,
            )

        if args.checkpoint_dir is not None:
            write_dataset_checkpoint(
                results_csv=args.output_csv,
                checkpoint_dir=args.checkpoint_dir,
                dataset=dataset_info.name,
                embedder=short_model_name,
            )


if __name__ == "__main__":
    main()
    print("All done", flush=True)
