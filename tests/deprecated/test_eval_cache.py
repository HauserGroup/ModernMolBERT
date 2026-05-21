from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.cache import (
    compute_feature_cache_key,
    featurizer_cache_identity,
    get_or_compute_features,
    hash_molecule_values,
    load_feature_batch,
    save_feature_batch,
)
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer


def test_hash_molecule_values_depends_on_order() -> None:
    assert hash_molecule_values(["CCO", "CCN"]) != hash_molecule_values(["CCN", "CCO"])


def test_cache_key_changes_with_featurizer_identity() -> None:
    molecule_hash = hash_molecule_values(["CCO"])

    key_a = compute_feature_cache_key(
        dataset_name="toy",
        split_name="train",
        smiles_column="smiles",
        molecule_hash=molecule_hash,
        featurizer_identity=featurizer_cache_identity(
            DummyFeaturizer(name="dummy_4", n_features=4)
        ),
    )
    key_b = compute_feature_cache_key(
        dataset_name="toy",
        split_name="train",
        smiles_column="smiles",
        molecule_hash=molecule_hash,
        featurizer_identity=featurizer_cache_identity(
            DummyFeaturizer(name="dummy_8", n_features=8)
        ),
    )

    assert key_a != key_b


def test_save_load_feature_batch_round_trip(tmp_path: Path) -> None:
    batch = FeatureBatch(
        X=np.array([[1.0, 2.0]], dtype=np.float32),
        valid_mask=np.array([True, False]),
    )

    entry = tmp_path / "entry"
    save_feature_batch(
        batch=batch,
        cache_entry_dir=entry,
        metadata={
            "dataset_name": "toy",
            "split_name": "train",
            "smiles_column": "smiles",
            "n_rows": 2,
            "molecule_hash": "abc",
            "featurizer_name": "dummy",
        },
    )

    loaded = load_feature_batch(cache_entry_dir=entry, n_inputs=2)

    np.testing.assert_array_equal(loaded.X, batch.X)
    np.testing.assert_array_equal(loaded.valid_mask, batch.valid_mask)


def test_get_or_compute_features_writes_and_reuses_cache(tmp_path: Path) -> None:
    frame = pd.DataFrame({"smiles": ["CCO", "", "CCN"]})
    featurizer = DummyFeaturizer(name="dummy_4", n_features=4)

    first = get_or_compute_features(
        dataset_name="toy",
        split_name="train",
        frame=frame,
        smiles_column="smiles",
        featurizer=featurizer,
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    second = get_or_compute_features(
        dataset_name="toy",
        split_name="train",
        frame=frame,
        smiles_column="smiles",
        featurizer=featurizer,
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    np.testing.assert_array_equal(first.X, second.X)
    assert first.metadata["cache_key"] == second.metadata["cache_key"]
    assert list((tmp_path / "cache").rglob("features.npy"))


def test_get_or_compute_features_rejects_missing_smiles_column(tmp_path: Path) -> None:
    frame = pd.DataFrame({"other": ["CCO"]})

    with pytest.raises(ValueError, match="SMILES column"):
        get_or_compute_features(
            dataset_name="toy",
            split_name="train",
            frame=frame,
            smiles_column="smiles",
            featurizer=DummyFeaturizer(),
            cache_dir=tmp_path / "cache",
            use_cache=True,
            batch_size=2,
        )
