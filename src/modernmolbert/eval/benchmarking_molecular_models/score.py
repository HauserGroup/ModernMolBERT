import argparse
import logging as log
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
    load_embedding_config,
    load_yaml_config,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.db import DbContex
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import EmbeddingConfig
from modernmolbert.eval.benchmarking_molecular_models.src.eval import (
    AVAILABLE_HEADS,
    eval_procedure,
)


def eval(cfg, embed_config, model_name, dataset_info, short_model_name, model_head, override):
    log.info(
        f"Evaluating model {model_name} on dataset {dataset_info.name} with metric {dataset_info.metric} with head {model_head}"
    )
    if "safe" in cfg and cfg.safe:
        log.info("Running in safe mode")
        try:
            eval_procedure(
                dataset_info=dataset_info,
                embedded_dir=embed_config.embedded_directory,
                predictions_dir=embed_config.predictions_directory,
                model_name=short_model_name,
                model_head=model_head,
                override=override,
            )
        except Exception as e:
            import traceback

            log.error(f"Error during evaluation: {e}")
            log.error(traceback.format_exc())
            return
    else:
        eval_procedure(
            dataset_info=dataset_info,
            embedded_dir=embed_config.embedded_directory,
            predictions_dir=embed_config.predictions_directory,
            model_name=short_model_name,
            model_head=model_head,
            override=override,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing molecular embedding files.")
    parser.add_argument("--embedder", "--model-name", dest="model_name", required=False)
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
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--safe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Compatibility support for key=value overrides such as model_name=my_embedder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_dir = root / args.config_dir
    cfg = load_yaml_config(config_dir / "score.yaml")
    cfg.embedding = load_embedding_config(config_dir)

    model_name = args.model_name
    for override in args.overrides:
        if override.startswith("model_name="):
            model_name = override.split("=", 1)[1]

    if "model" in cfg and "embedding_name" in cfg.model:
        model_name = model_name or cfg.model.embedding_name
    elif "model_name" in cfg:
        model_name = model_name or cfg.model_name
    elif "model" in cfg and "model_name" in cfg.model:
        model_name = model_name or cfg.model.model_name

    if not model_name:
        raise ValueError(
            "Scoring requires --embedder <name>. "
            "Expected embeddings at data/embedded/<dataset>/<embedder>.joblib."
        )

    dataset_selections = args.datasets or cfg.get("datasets", ["all"])
    dataset_names = expand_dataset_selection(config_dir, dataset_selections)
    cache = cfg.get("cache", True) if args.cache is None else args.cache
    safe = cfg.get("safe", False) if args.safe is None else args.safe
    cfg.cache = cache
    cfg.safe = safe

    embed_config = EmbeddingConfig(**cfg.embedding)
    if "gpt" in model_name.lower():
        short_model_name = model_name.split("/")[-1]
    else:
        short_model_name = model_name.split("/")[-1].split(".")[0]
    override = not cache
    print(f"Override status {override}")

    with DbContex(embed_config):
        for dataset_name in dataset_names:
            dataset_info = load_dataset_config(config_dir, dataset_name)
            for model_head in args.heads:
                eval(
                    cfg,
                    embed_config,
                    model_name,
                    dataset_info,
                    short_model_name,
                    model_head,
                    override,
                )


if __name__ == "__main__":
    main()
    print("All done")
