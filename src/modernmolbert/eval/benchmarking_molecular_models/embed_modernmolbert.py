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

from modernmolbert.eval.benchmarking_molecular_models.common.config import (
    expand_dataset_selection,
    load_dataset_config,
    load_embedding_config,
)
from modernmolbert.eval.benchmarking_molecular_models.common.types import (
    Dataset,
    EmbeddedDataset,
    EmbeddingConfig,
)
from modernmolbert.utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed prepared Praski benchmark datasets with a ModernMolBERT checkpoint.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer bundle path; defaults to --model-dir.",
    )
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=None)
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
    if X.shape[1] == 0:
        raise ValueError(f"No valid embeddings for {dataset.name}")
    retained = np.flatnonzero(~np.isnan(X).any(axis=1))
    dropped = np.flatnonzero(np.isnan(X).any(axis=1))
    metadata["source_n_rows"] = len(smiles)
    metadata["source_row_indices"] = retained.tolist()
    metadata["failed_source_row_indices"] = dropped.tolist()
    metadata["source_split_counts"] = {
        split: len(indices) for split, indices in dataset.splits.items()
    }
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
    n_failed = embedded.remove_failed_embeddings()
    if n_failed != len(dropped) or embedded.X.shape[0] != len(retained):
        raise ValueError(f"Embedding row provenance mismatch for {dataset.name}")
    metadata["retained_split_counts"] = {
        split: len(indices) for split, indices in embedded.splits.items()
    }
    return embedded


def load_prepared_dataset(path: Path) -> Dataset:
    legacy_path = path.with_suffix(".json")
    if legacy_path.exists():
        return Dataset.deserialize_legacy(legacy_path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    return joblib.load(path)


def assert_reusable_embedding(output_path: Path, prepared_path: Path) -> None:
    """Never skip a stale or unreadable pickle as if it matched the prepared cohort."""
    source_path = prepared_path.with_suffix(".json")
    if not source_path.exists():
        source_path = prepared_path
    try:
        embedded = joblib.load(output_path, mmap_mode="r")
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(
            f"Existing embedding {output_path} uses an obsolete pickle class; "
            "regenerate it with --overwrite"
        ) from exc
    except (EOFError, OSError) as exc:
        raise ValueError(
            f"Existing embedding {output_path} is truncated or unreadable; "
            "regenerate it with --overwrite"
        ) from exc
    recorded = getattr(embedded, "metadata", {}).get("prepared_data_sha256")
    if recorded != file_sha256(source_path):
        raise ValueError(
            f"Existing embedding {output_path} has a different prepared-data hash; "
            "regenerate it with --overwrite"
        )


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
            assert_reusable_embedding(output_path, prepared_path)
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
        source_path = prepared_path.with_suffix(".json")
        if not source_path.exists():
            source_path = prepared_path
        embedded.metadata["prepared_data_path"] = str(source_path)
        embedded.metadata["prepared_data_sha256"] = file_sha256(source_path)

        # Free the prepared dataset before writing the embedded one
        del dataset
        gc.collect()

        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
        joblib.dump(embedded, tmp_output_path)
        os.replace(tmp_output_path, output_path)

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
