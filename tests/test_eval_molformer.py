"""Optional MoLFormer evaluation smoke tests.

These are skipped by default because MoLFormer requires a separate
Transformers 4.x environment and downloads/executes Hugging Face remote code.

Run from the repo root inside the molformer-only conda env:

    conda activate molformer-only
    PYTHONPATH="$PWD/src" MODERNMOLBERT_RUN_MOLFORMER_TESTS=1 \
      python -m pytest tests/test_eval_molformer.py -q -s
"""

import os

import numpy as np
import pytest

from modernmolbert.eval.featurizers.hf_smiles import HuggingFaceSmilesFeaturizer
from modernmolbert.eval.registry import make_featurizer


MOLFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"


def _molformer_enabled() -> bool:
    return os.environ.get("MODERNMOLBERT_RUN_MOLFORMER_TESTS") == "1"


@pytest.mark.model
@pytest.mark.molformer
def test_molformer_registry_constructs_featurizer() -> None:
    featurizer = make_featurizer(
        "hf_smiles",
        name="molformer_xl_both_10pct",
        model_name_or_path=MOLFORMER_MODEL,
        revision=MOLFORMER_REVISION,
        max_seq_length=64,
        pooling="mean",
        device="cpu",
        trust_remote_code=True,
    )

    assert featurizer.name == "molformer_xl_both_10pct"


@pytest.mark.model
@pytest.mark.molformer
def test_molformer_embedding_smoke() -> None:
    if not _molformer_enabled():
        pytest.skip(
            "Set MODERNMOLBERT_RUN_MOLFORMER_TESTS=1 to run MoLFormer smoke tests."
        )

    featurizer = HuggingFaceSmilesFeaturizer(
        name="molformer_xl_both_10pct",
        model_name_or_path=MOLFORMER_MODEL,
        revision=MOLFORMER_REVISION,
        max_seq_length=128,
        pooling="mean",
        device="cpu",
        trust_remote_code=True,
    )

    out = featurizer.featurize_smiles(
        ["CCO", "c1ccccc1", "CC(=O)O", "", None],  # type: ignore[list-item]
        batch_size=2,
    )

    out.check(5)

    assert out.valid_mask.tolist() == [True, True, True, False, False]
    assert out.X.shape == (3, 768)
    assert out.X.dtype == np.float32
    assert np.isfinite(out.X).all()

    assert out.metadata["backend"] == "huggingface_transformers"
    assert out.metadata["featurizer"] == "molformer_xl_both_10pct"
    assert out.metadata["model_name_or_path"] == MOLFORMER_MODEL
    assert out.metadata["revision"] == MOLFORMER_REVISION
    assert out.metadata["pooling"] == "mean"
    assert out.metadata["hidden_size"] == 768
    assert out.metadata["num_hidden_layers"] == 12
    assert out.metadata["vocab_size"] == 2362
    assert out.metadata["trust_remote_code"] is True
    assert int(out.metadata["num_parameters"]) == 44_375_040
