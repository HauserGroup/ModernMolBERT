import numpy as np
import pytest

from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.featurizers.rdkit_ecfp import ECFP4Featurizer


def test_feature_batch_check_accepts_valid_batch() -> None:
    batch = FeatureBatch(
        X=np.zeros((2, 8), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )

    batch.check(n_inputs=3)


def test_feature_batch_check_rejects_bad_row_count() -> None:
    batch = FeatureBatch(
        X=np.zeros((1, 8), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )

    with pytest.raises(ValueError):
        batch.check(n_inputs=3)


def test_dummy_featurizer_contract() -> None:
    featurizer = DummyFeaturizer(name="dummy_4", n_features=4)

    batch = featurizer.featurize_smiles(["CCO", "", "CCN"])

    batch.check(n_inputs=3)
    assert batch.valid_mask.tolist() == [True, False, True]
    assert batch.X.shape == (2, 4)
    assert batch.metadata["featurizer"] == "dummy_4"


def test_ecfp4_featurizer_contract() -> None:
    featurizer = ECFP4Featurizer(name="ecfp4_128", n_bits=128)

    batch = featurizer.featurize_smiles(["CCO", "not_a_smiles", "CCN"])

    batch.check(n_inputs=3)
    assert batch.valid_mask.tolist() == [True, False, True]
    assert batch.X.shape == (2, 128)
    assert batch.metadata["featurizer"] == "ecfp4_128"


def test_feature_batch_check_rejects_non_boolean_valid_mask() -> None:
    batch = FeatureBatch(
        X=np.zeros((2, 8), dtype=np.float32),
        valid_mask=np.array([1, 0, 1]),
    )

    with pytest.raises(TypeError, match="valid_mask must be boolean"):
        batch.check(n_inputs=3)


def test_feature_batch_check_rejects_wrong_mask_shape() -> None:
    batch = FeatureBatch(
        X=np.zeros((2, 8), dtype=np.float32),
        valid_mask=np.array([[True, False, True]]),
    )

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        batch.check(n_inputs=3)


def test_feature_batch_check_rejects_non_numeric_or_non_finite_features() -> None:
    non_numeric = FeatureBatch(
        X=np.array([["bad"]], dtype=object),
        valid_mask=np.array([True]),
    )
    non_finite = FeatureBatch(
        X=np.array([[np.inf]], dtype=np.float32),
        valid_mask=np.array([True]),
    )

    with pytest.raises(TypeError, match="X must be numeric"):
        non_numeric.check(n_inputs=1)
    with pytest.raises(ValueError, match="X contains non-finite"):
        non_finite.check(n_inputs=1)
