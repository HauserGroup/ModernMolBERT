import os

import numpy as np
import pytest
import torch

from modernmolbert.eval.featurizers.hf_smiles import HuggingFaceSmilesFeaturizer
from modernmolbert.eval.registry import make_featurizer
from modernmolbert.eval.pooling import mean_pool_excluding_token_ids


def _hf_enabled() -> bool:
    return os.environ.get("MODERNMOLBERT_RUN_HF_TESTS") == "1"


@pytest.mark.model
def test_hf_smiles_registry_constructs_featurizer() -> None:
    featurizer = make_featurizer(
        "hf_smiles",
        name="chemberta_test",
        model_name_or_path="DeepChem/ChemBERTa-77M-MLM",
        max_seq_length=64,
        pooling="mean",
        device="cpu",
    )

    assert featurizer.name == "chemberta_test"


@pytest.mark.model
def test_chemberta2_embedding_smoke() -> None:
    """Optional network/model-loading smoke test for ChemBERTa embeddings.

    Enable with:
        MODERNMOLBERT_RUN_HF_TESTS=1 uv run pytest tests/test_eval_hf_smiles.py -q -s
    """
    if not _hf_enabled():
        pytest.skip("Set MODERNMOLBERT_RUN_HF_TESTS=1 to run Hugging Face model tests.")

    featurizer = HuggingFaceSmilesFeaturizer(
        name="chemberta_77m_mlm",
        model_name_or_path="DeepChem/ChemBERTa-77M-MLM",
        max_seq_length=64,
        pooling="mean",
        device="cpu",
    )

    out = featurizer.featurize_smiles(
        ["CCO", "c1ccccc1", "CC(=O)O", "", None],  # type: ignore[list-item]
        batch_size=2,
    )

    out.check(5)

    assert out.valid_mask.tolist() == [True, True, True, False, False]
    assert out.X.shape[0] == 3
    assert out.X.shape[1] == 384
    assert out.X.dtype == np.float32
    assert np.isfinite(out.X).all()

    assert out.metadata["model_name_or_path"] == "DeepChem/ChemBERTa-77M-MLM"
    assert out.metadata["pooling"] == "mean"
    assert out.metadata["hidden_size"] == 384


@pytest.mark.model
def test_chemberta2_cls_pooling_smoke() -> None:
    if not _hf_enabled():
        pytest.skip("Set MODERNMOLBERT_RUN_HF_TESTS=1 to run Hugging Face model tests.")

    featurizer = HuggingFaceSmilesFeaturizer(
        name="chemberta_77m_mlm_cls",
        model_name_or_path="DeepChem/ChemBERTa-77M-MLM",
        max_seq_length=64,
        pooling="cls",
        device="cpu",
    )

    out = featurizer.featurize_smiles(["CCO", "c1ccccc1"], batch_size=2)

    out.check(2)

    assert out.valid_mask.tolist() == [True, True]
    assert out.X.shape == (2, 384)
    assert np.isfinite(out.X).all()


def test_sanitize_modernbert_rope_config_handles_rope_scaling() -> None:
    from modernmolbert.eval.featurizers.hf_smiles import (
        _sanitize_modernbert_rope_config,
    )

    cfg = {
        "model_type": "modernbert",
        "rope_scaling": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "rope_type": "default",
            "rope_theta": None,
        },
    }

    out = _sanitize_modernbert_rope_config(cfg)

    assert set(out["rope_scaling"]) == {"sliding_attention", "full_attention"}
    assert "rope_type" not in out["rope_scaling"]
    assert "rope_theta" not in out["rope_scaling"]


def test_mean_pool_excludes_special_tokens() -> None:

    # shape: [1, 3 tokens, 2 dims]
    hidden = torch.tensor([[[10.0, 10.0], [2.0, 2.0], [20.0, 20.0]]])
    attention = torch.tensor([[1, 1, 1]])
    input_ids = torch.tensor([[0, 7, 2]])

    pooled = mean_pool_excluding_token_ids(
        last_hidden_state=hidden,
        attention_mask=attention,
        input_ids=input_ids,
        excluded_token_ids={0, 2},
    )

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))
