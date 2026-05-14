from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
    load_embedding_config,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    Dataset,
    EmbeddedDataset,
    EmbeddingConfig,
)

DEFAULT_MODEL_DIR = Path("runs/pubchem10m_mps_base_pilot_256/final_model")
DEFAULT_EMBEDDER = "modernmolbert_pubchem10m_mps_base_pilot_256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed prepared Praski benchmark datasets with a ModernMolBERT checkpoint.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def full_feature_matrix(feature_batch: Any, *, n_inputs: int) -> np.ndarray:
    feature_batch.check(n_inputs=n_inputs)
    if feature_batch.X.shape[0] > 0:
        n_features = feature_batch.X.shape[1]
    else:
        n_features = int(
            feature_batch.metadata.get("hidden_size")
            or feature_batch.metadata.get("n_features")
            or 0
        )
        if n_features <= 0:
            raise ValueError("Cannot infer feature dimension from an all-invalid feature batch")

    out = np.full((n_inputs, n_features), np.nan, dtype=np.float32)
    out[feature_batch.valid_mask] = feature_batch.X.astype(np.float32, copy=False)
    return out


def embed_dataset(dataset: Dataset, *, featurizer: Any, embedder_name: str, batch_size: int):
    smiles = dataset.data["smiles"].astype(str).tolist()
    feature_batch = featurizer.featurize_smiles(smiles, batch_size=batch_size)
    X = full_feature_matrix(feature_batch, n_inputs=len(smiles))

    embedded = EmbeddedDataset(
        name=dataset.name,
        task=dataset.task,
        embedder=embedder_name,
        splits=dataset.splits,
        X=X,
        y=dataset.labels.copy(),
    )
    embedded.remove_failed_embeddings()
    return embedded


def load_prepared_dataset(path: Path) -> Dataset:
    legacy_path = path.with_suffix(".json")
    if legacy_path.exists():
        return Dataset.deserialize_legacy(legacy_path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    return joblib.load(path)


def make_featurizer(args: argparse.Namespace):
    from modernmolbert.eval.featurizers.modernmolbert_selfies import (
        ModernMolBERTSelfiesFeaturizer,
    )

    return ModernMolBERTSelfiesFeaturizer(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        name=args.embedder,
        max_seq_length=args.max_seq_length,
        pooling=args.pooling,
        device=args.device,
        batch_size=args.batch_size,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_dir = root / args.config_dir
    embed_config = EmbeddingConfig(**load_embedding_config(config_dir))
    dataset_names = expand_dataset_selection(config_dir, args.datasets)
    featurizer = make_featurizer(args)

    for dataset_config_name in dataset_names:
        dataset_config = load_dataset_config(config_dir, dataset_config_name)
        dataset_name = dataset_config.name
        prepared_path = (
            Path(os.getcwd()) / embed_config.prepared_directory / f"{dataset_name}.joblib"
        )
        output_dir = Path(os.getcwd()) / embed_config.embedded_directory / dataset_name
        output_path = output_dir / f"{args.embedder}.joblib"

        if output_path.exists() and not args.overwrite:
            print(f"Embedding already exists, skipping: {output_path}", flush=True)
            continue

        dataset = load_prepared_dataset(prepared_path)
        embedded = embed_dataset(
            dataset,
            featurizer=featurizer,
            embedder_name=args.embedder,
            batch_size=args.batch_size,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(embedded, output_path)
        print(
            f"Embedded {dataset_name}: X={embedded.X.shape}, y={embedded.y.shape}, output={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
