import argparse
import os
import joblib
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
    load_embedding_config,
    load_yaml_config,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.data_v2 import load
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import EmbeddingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset config stems, globs, or 'all'.",
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_dir = root / args.config_dir
    cfg = load_yaml_config(config_dir / "downloader.yaml")
    embed_config = EmbeddingConfig(**load_embedding_config(config_dir))
    destination = os.path.join(os.getcwd(), embed_config.prepared_directory)
    dataset_names = expand_dataset_selection(
        config_dir, args.datasets or cfg.get("datasets", ["all"])
    )
    cache = cfg.get("cache", True) if args.cache is None else args.cache

    os.makedirs(destination, exist_ok=True)
    for dataset_config_name in dataset_names:
        dataset_config = load_dataset_config(config_dir, dataset_config_name)
        dataset_name = dataset_config.name
        filename = os.path.join(destination, f"{dataset_name}.joblib")
        legacy_filename = os.path.join(destination, f"{dataset_name}.json")

        if (os.path.exists(filename) or os.path.exists(legacy_filename)) and cache:
            print(f"Dataset {dataset_name} already exists at {filename}")
            continue

        dataset = load(dataset_config, embed_config.raw_directory)

        joblib.dump(dataset, filename)
        dataset.serialize_legacy(legacy_filename)
        print(f"Dataset {dataset_name} saved to {filename}")


if __name__ == "__main__":
    main()
