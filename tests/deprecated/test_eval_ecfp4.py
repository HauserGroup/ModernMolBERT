import numpy as np

from modernmolbert.eval.featurizers.rdkit_ecfp import ECFP4Featurizer
from modernmolbert.eval.registry import make_featurizer


def test_ecfp4_featurizer_basic_shape() -> None:
    featurizer = ECFP4Featurizer(n_bits=2048)

    out = featurizer.featurize_smiles(["CCO", "c1ccccc1", "not_a_smiles", ""])

    out.check(4)

    assert out.valid_mask.tolist() == [True, True, False, False]
    assert out.X.shape == (2, 2048)
    assert out.X.dtype == np.float32

    # Binary bit vector.
    assert set(np.unique(out.X)).issubset({0.0, 1.0})

    # Non-empty molecules should set at least one bit.
    assert np.all(out.X.sum(axis=1) > 0)

    assert out.metadata["featurizer"] == "ecfp4"


def test_ecfp4_registry() -> None:
    featurizer = make_featurizer("ecfp4", name="ecfp4_test", n_bits=512)

    out = featurizer.featurize_smiles(["CCO"])

    assert out.X.shape == (1, 512)
    assert out.valid_mask.tolist() == [True]

    assert featurizer.name == "ecfp4_test"
    assert out.metadata["featurizer"] == "ecfp4_test"


def test_ecfp4_is_deterministic() -> None:
    featurizer = ECFP4Featurizer(n_bits=1024)

    a = featurizer.featurize_smiles(["CCO"]).X
    b = featurizer.featurize_smiles(["CCO"]).X

    np.testing.assert_array_equal(a, b)
