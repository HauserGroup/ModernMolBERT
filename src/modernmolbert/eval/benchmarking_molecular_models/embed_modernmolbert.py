"""Embed molecules using a ModernMolBERT checkpoint.

Note: loading a checkpoint saved from ModernBertForMaskedLM into ModernBertModel
(encoder-only, no prediction head) will log UNEXPECTED keys for
``decoder.bias``, ``head.norm.weight``, and ``head.dense.weight``. These are
the MLM head weights and are intentionally discarded — this is expected and safe.
"""

import argparse
import gc
import os
import time
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


def expand_to_nan_matrix(X_valid: np.ndarray, valid_mask: np.ndarray, n_inputs: int) -> np.ndarray:
    """Expand compact valid-only X into a full (n_inputs, n_features) NaN matrix."""
    n_features = X_valid.shape[1] if X_valid.shape[0] > 0 else 0
    out = np.full((n_inputs, n_features), np.nan, dtype=np.float32)
    out[valid_mask] = X_valid
    return out


def embed_dataset(dataset: Dataset, *, featurizer: Any, embedder_name: str, batch_size: int):
    smiles = dataset.data["smiles"].astype(str).tolist()
    feature_batch = featurizer.featurize_smiles(smiles, batch_size=batch_size)
    metadata = dict(feature_batch.metadata)

    # Expand to full matrix then immediately free the compact batch
    X = expand_to_nan_matrix(feature_batch.X, feature_batch.valid_mask, n_inputs=len(smiles))
    del feature_batch
    gc.collect()

    embedded = EmbeddedDataset(
        name=dataset.name,
        task=dataset.task,
        embedder=embedder_name,
        splits=dataset.splits,
        X=X,
        y=dataset.labels.copy(),
        metadata=metadata,
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


def _warn_if_not_best_model(model_dir: Path) -> None:
    resolved = str(model_dir.resolve())
    if "best" not in resolved.lower():
        import warnings

        warnings.warn(
            f"model-dir does not contain 'best' in its path: {model_dir}\n"
            "Pass runs/best_<name> to use the designated best checkpoint.",
            UserWarning,
            stacklevel=3,
        )


def main() -> None:
    args = parse_args()
    _warn_if_not_best_model(args.model_dir)
    root = Path(__file__).resolve().parent
    config_dir = root / args.config_dir
    embed_config = EmbeddingConfig(**load_embedding_config(config_dir))
    dataset_names = expand_dataset_selection(config_dir, args.datasets)
    n_total = len(dataset_names)

    print(f"[embed] model:    {args.model_dir}", flush=True)
    print(f"[embed] embedder: {args.embedder}", flush=True)
    print(f"[embed] datasets: {n_total}", flush=True)

    featurizer = make_featurizer(args)

    wall_start = time.perf_counter()

    for idx, dataset_config_name in enumerate(dataset_names, start=1):
        dataset_config = load_dataset_config(config_dir, dataset_config_name)
        dataset_name = dataset_config.name
        prepared_path = (
            Path(os.getcwd()) / embed_config.prepared_directory / f"{dataset_name}.joblib"
        )
        output_dir = Path(os.getcwd()) / embed_config.embedded_directory / dataset_name
        output_path = output_dir / f"{args.embedder}.joblib"

        if output_path.exists() and not args.overwrite:
            print(
                f"[{idx:>2}/{n_total}] SKIP  {dataset_name} — embedding exists",
                flush=True,
            )
            continue

        print(
            f"[{idx:>2}/{n_total}] START {dataset_name}",
            flush=True,
        )
        t0 = time.perf_counter()

        dataset = load_prepared_dataset(prepared_path)
        n_samples = len(dataset.data)
        print(f"         loaded {n_samples:,} samples", flush=True)

        embedded = embed_dataset(
            dataset,
            featurizer=featurizer,
            embedder_name=args.embedder,
            batch_size=args.batch_size,
        )

        # Free the prepared dataset before writing the embedded one
        del dataset
        gc.collect()

        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(embedded, output_path)

        elapsed = time.perf_counter() - t0
        print(
            f"         DONE  X={embedded.X.shape}  y={embedded.y.shape}"
            f"  [{elapsed:.1f}s]  → {output_path}",
            flush=True,
        )

        del embedded
        gc.collect()

    total_elapsed = time.perf_counter() - wall_start
    print(f"\n[embed] finished {n_total} datasets in {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
