from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from modernmolbert.eval.featurizers.base import FeatureBatch, RepresentationFeaturizer


def _json_safe(value: Any) -> Any:
    """Convert common Python objects to JSON-safe values for cache metadata."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def hash_molecule_values(values: Sequence[Any]) -> str:
    """Hash ordered molecule values exactly as passed to the featurizer.

    The order is intentionally part of the hash because FeatureBatch.X rows and
    valid_mask are tied to the input order.
    """

    h = hashlib.sha256()

    for value in values:
        if value is None or pd.isna(value):
            text = "<NA>"
        else:
            text = str(value)

        h.update(text.encode("utf-8"))
        h.update(b"\0")

    return h.hexdigest()


def _public_featurizer_params(featurizer: RepresentationFeaturizer) -> dict[str, Any]:
    """Return JSON-safe public featurizer parameters.

    Avoid storing loaded model/tokenizer objects. This identity should reflect
    the configuration that affects the generated features.
    """

    if is_dataclass(featurizer):
        params = asdict(featurizer)
    else:
        params = {
            key: value
            for key, value in vars(featurizer).items()
            if not key.startswith("_")
        }

    # Exclude heavy/runtime objects if present.
    for key in [
        "model",
        "tokenizer",
        "_device",
        "_eligible_replacement_ids",
    ]:
        params.pop(key, None)

    return _json_safe(params)


def featurizer_cache_identity(
    featurizer: RepresentationFeaturizer,
) -> dict[str, Any]:
    """Return a stable-ish identity for a featurizer instance."""

    return {
        "name": featurizer.name,
        "class": (
            f"{featurizer.__class__.__module__}.{featurizer.__class__.__qualname__}"
        ),
        "params": _public_featurizer_params(featurizer),
    }


def compute_feature_cache_key(
    *,
    dataset_name: str,
    split_name: str,
    smiles_column: str,
    molecule_hash: str,
    featurizer_identity: Mapping[str, Any],
) -> str:
    """Compute a stable cache key for one dataset split and featurizer."""

    payload = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "smiles_column": smiles_column,
        "molecule_hash": molecule_hash,
        "featurizer": _json_safe(dict(featurizer_identity)),
    }

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def feature_cache_entry_dir(cache_dir: Path, cache_key: str) -> Path:
    """Return the directory for one cached feature entry."""

    return cache_dir / "features" / cache_key


def save_feature_batch(
    *,
    batch: FeatureBatch,
    cache_entry_dir: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Save a FeatureBatch and metadata to disk.

    Writes through a temporary directory and then atomically renames it into
    place. This avoids partially-written cache entries.
    """

    cache_entry_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{cache_entry_dir.name}.tmp-",
        dir=str(cache_entry_dir.parent),
    ) as tmp_name:
        tmp_dir = Path(tmp_name)

        np.save(tmp_dir / "features.npy", batch.X)
        np.save(tmp_dir / "valid_mask.npy", batch.valid_mask)

        metadata_out = _json_safe(dict(metadata))
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata_out, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if cache_entry_dir.exists():
            # Replace old cache entry atomically enough for local single-process use.
            for child in cache_entry_dir.iterdir():
                child.unlink()
            cache_entry_dir.rmdir()

        tmp_dir.rename(cache_entry_dir)


def load_feature_batch(
    *,
    cache_entry_dir: Path,
    n_inputs: int,
) -> FeatureBatch:
    """Load a FeatureBatch from a cache entry directory."""

    features_path = cache_entry_dir / "features.npy"
    mask_path = cache_entry_dir / "valid_mask.npy"
    metadata_path = cache_entry_dir / "metadata.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing cached features: {features_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing cached valid mask: {mask_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cached metadata: {metadata_path}")

    X = np.load(features_path)
    valid_mask = np.load(mask_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    batch = FeatureBatch(
        X=X,
        valid_mask=valid_mask,
        metadata=metadata,
    )
    batch.check(n_inputs=n_inputs)
    return batch


def _validate_cache_metadata(
    *,
    metadata: Mapping[str, Any],
    dataset_name: str,
    split_name: str,
    smiles_column: str,
    n_rows: int,
    molecule_hash: str,
    featurizer_name: str,
) -> None:
    """Validate that a cache entry matches the requested computation."""

    checks = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "smiles_column": smiles_column,
        "n_rows": n_rows,
        "molecule_hash": molecule_hash,
        "featurizer_name": featurizer_name,
    }

    for key, expected in checks.items():
        observed = metadata.get(key)
        if observed != expected:
            raise ValueError(
                f"Cache metadata mismatch for {key!r}: "
                f"expected {expected!r}, observed {observed!r}"
            )


def get_or_compute_features(
    *,
    dataset_name: str,
    split_name: str,
    frame: pd.DataFrame,
    smiles_column: str,
    featurizer: RepresentationFeaturizer,
    cache_dir: Path | None,
    use_cache: bool,
    batch_size: int,
) -> FeatureBatch:
    """Load features from cache or compute them with a featurizer.

    Cache identity depends only on dataset split, ordered molecule values,
    molecule column name, and featurizer identity. It intentionally does not
    depend on downstream model settings or random seeds.
    """

    if smiles_column not in frame.columns:
        raise ValueError(
            f"Split {split_name!r} is missing SMILES column {smiles_column!r}"
        )

    smiles_values = frame[smiles_column].tolist()
    molecule_hash = hash_molecule_values(smiles_values)
    featurizer_identity = featurizer_cache_identity(featurizer)

    cache_key = compute_feature_cache_key(
        dataset_name=dataset_name,
        split_name=split_name,
        smiles_column=smiles_column,
        molecule_hash=molecule_hash,
        featurizer_identity=featurizer_identity,
    )

    cache_entry_dir: Path | None = None

    if cache_dir is not None:
        cache_entry_dir = feature_cache_entry_dir(Path(cache_dir), cache_key)

    if use_cache and cache_entry_dir is not None and cache_entry_dir.exists():
        batch = load_feature_batch(
            cache_entry_dir=cache_entry_dir,
            n_inputs=len(frame),
        )
        _validate_cache_metadata(
            metadata=batch.metadata,
            dataset_name=dataset_name,
            split_name=split_name,
            smiles_column=smiles_column,
            n_rows=len(frame),
            molecule_hash=molecule_hash,
            featurizer_name=featurizer.name,
        )
        return batch

    batch = featurizer.featurize_smiles(
        [str(value) if not pd.isna(value) else "" for value in smiles_values],
        batch_size=batch_size,
    )
    batch.check(n_inputs=len(frame))

    metadata = {
        "cache_key": cache_key,
        "created_at_unix": time.time(),
        "dataset_name": dataset_name,
        "split_name": split_name,
        "smiles_column": smiles_column,
        "n_rows": int(len(frame)),
        "n_valid": int(batch.valid_mask.sum()),
        "n_invalid": int((~batch.valid_mask).sum()),
        "invalid_fraction": float((~batch.valid_mask).mean()) if len(frame) else 0.0,
        "n_features": int(batch.X.shape[1]) if batch.X.ndim == 2 else 0,
        "molecule_hash": molecule_hash,
        "featurizer_name": featurizer.name,
        "featurizer_identity": featurizer_identity,
        "featurizer_metadata": batch.metadata,
    }

    batch = FeatureBatch(
        X=batch.X,
        valid_mask=batch.valid_mask,
        metadata=metadata,
    )
    batch.check(n_inputs=len(frame))

    if use_cache and cache_entry_dir is not None:
        save_feature_batch(
            batch=batch,
            cache_entry_dir=cache_entry_dir,
            metadata=metadata,
        )

    return batch
