import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.io import ensure_dir, read_json, write_json


@dataclass(frozen=True)
class FeatureCacheKey:
    dataset_name: str
    split_name: str
    smiles_hash: str
    featurizer_name: str
    featurizer_metadata: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class FeatureCache:
    root: Path

    def split_dir(self, key: FeatureCacheKey) -> Path:
        safe_dataset = _safe_name(key.dataset_name)
        safe_featurizer = _safe_name(key.featurizer_name)
        return (
            self.root / safe_dataset / key.split_name / safe_featurizer / key.digest()
        )

    def exists(self, key: FeatureCacheKey) -> bool:
        path = self.split_dir(key)
        return (
            (path / "features.npy").exists()
            and (path / "valid_mask.npy").exists()
            and (path / "metadata.json").exists()
        )

    def load(self, key: FeatureCacheKey, n_inputs: int) -> FeatureBatch:
        path = self.split_dir(key)
        X = np.load(path / "features.npy", allow_pickle=False)
        valid_mask = np.load(path / "valid_mask.npy", allow_pickle=False)
        metadata = read_json(path / "metadata.json")
        out = FeatureBatch(X=X, valid_mask=valid_mask, metadata=metadata)
        out.check(n_inputs)
        return out

    def save(self, key: FeatureCacheKey, features: FeatureBatch) -> Path:
        path = ensure_dir(self.split_dir(key))
        np.save(path / "features.npy", features.X)
        np.save(path / "valid_mask.npy", features.valid_mask)
        write_json(
            path / "metadata.json",
            {
                "cache_key": asdict(key),
                "feature_metadata": features.metadata,
                "n_valid": int(features.valid_mask.sum()),
                "n_inputs": int(features.valid_mask.shape[0]),
                "feature_dim": int(features.X.shape[1])
                if features.X.ndim == 2
                else None,
            },
        )
        return path


def _safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def get_or_compute_features(
    *,
    cache: FeatureCache | None,
    cache_key: FeatureCacheKey,
    smiles: Sequence[str],
    featurizer,
    batch_size: int,
    use_cache: bool = True,
) -> FeatureBatch:
    n_inputs = len(smiles)

    if use_cache and cache is not None and cache.exists(cache_key):
        return cache.load(cache_key, n_inputs=n_inputs)

    features = featurizer.featurize_smiles(smiles, batch_size=batch_size)
    features.check(n_inputs)

    if use_cache and cache is not None:
        cache.save(cache_key, features)

    return features
