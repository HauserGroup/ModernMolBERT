# tests/test_eval_modernmolbert_selfies.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modernmolbert.eval.featurizers.modernmolbert_selfies import (
    ModernMolBERTSelfiesFeaturizer,
)
from modernmolbert.eval.pooling import mean_pool_excluding_token_ids


class TinyTokenizer:
    """Minimal tokenizer stub for ModernMolBERTSelfiesFeaturizer tests."""

    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 4

    def __call__(
        self,
        texts,
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 32,
        return_tensors: str = "pt",
    ):
        assert return_tensors == "pt"

        encoded = []

        for text in texts:
            # Produce deterministic fake tokenization:
            # BOS + content tokens + EOS.
            content_len = max(1, min(3, len(text) % 4 + 1))
            ids = [self.bos_token_id]
            ids.extend([5 + (i % 3) for i in range(content_len)])
            ids.append(self.eos_token_id)

            if truncation:
                ids = ids[:max_length]

            encoded.append(ids)

        max_len = max(len(ids) for ids in encoded) if padding else None

        padded_ids = []
        attention_masks = []

        for ids in encoded:
            if padding and max_len is not None:
                pad_len = max_len - len(ids)
                padded = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            else:
                padded = ids
                mask = [1] * len(ids)

            padded_ids.append(padded)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


class TinyModel(torch.nn.Module):
    """Minimal model stub returning deterministic hidden states."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size

        # Deterministic hidden state based on token ids.
        values = input_ids.float().unsqueeze(-1).repeat(1, 1, hidden_size)
        offsets = torch.arange(hidden_size, device=input_ids.device).float()
        hidden = values + offsets

        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def tiny_modernmolbert_dir(tmp_path: Path) -> Path:
    """Placeholder path used by monkeypatched loaders."""
    model_dir = tmp_path / "tiny-modernmolbert"
    model_dir.mkdir()
    return model_dir


@pytest.fixture(autouse=True)
def patch_modernmolbert_loaders(monkeypatch):
    """Avoid loading a real tokenizer/model checkpoint in unit tests."""

    def fake_tokenizer_from_pretrained(path):
        return TinyTokenizer()

    def fake_model_from_pretrained(path):
        return TinyModel(hidden_size=8)

    monkeypatch.setattr(
        "modernmolbert.eval.featurizers.modernmolbert_selfies.APETokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "modernmolbert.eval.featurizers.modernmolbert_selfies.AutoModel.from_pretrained",
        fake_model_from_pretrained,
    )


def test_modernmolbert_selfies_featurizer_valid_and_invalid_smiles(
    tiny_modernmolbert_dir,
):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
    )

    batch = featurizer.featurize_smiles(["CCO", "not_a_smiles"])

    assert batch.valid_mask.tolist() == [True, False]
    assert batch.X.shape == (1, 8)
    assert batch.X.ndim == 2
    assert batch.X.dtype == np.float32
    batch.check(n_inputs=2)


def test_modernmolbert_selfies_featurizer_all_invalid_smiles(tiny_modernmolbert_dir):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
    )

    batch = featurizer.featurize_smiles(["not_a_smiles"])

    assert batch.valid_mask.tolist() == [False]
    assert batch.X.shape == (0, 8)
    assert batch.X.ndim == 2
    assert batch.X.dtype == np.float32
    batch.check(n_inputs=1)


def test_modernmolbert_selfies_featurizer_multiple_valid_smiles_batches(
    tiny_modernmolbert_dir,
):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
    )

    batch = featurizer.featurize_smiles(["CCO", "CCN", "c1ccccc1"])

    assert batch.valid_mask.tolist() == [True, True, True]
    assert batch.X.shape == (3, 8)
    assert batch.X.dtype == np.float32
    assert np.isfinite(batch.X).all()
    batch.check(n_inputs=3)


def test_modernmolbert_selfies_featurizer_preserves_input_order_in_valid_mask(
    tiny_modernmolbert_dir,
):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
    )

    batch = featurizer.featurize_smiles(
        ["not_a_smiles", "CCO", "also_not_smiles", "CCN"]
    )

    assert batch.valid_mask.tolist() == [False, True, False, True]
    assert batch.X.shape == (2, 8)
    batch.check(n_inputs=4)


def test_modernmolbert_selfies_featurizer_cls_pooling(tiny_modernmolbert_dir):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
        pooling="cls",
    )

    batch = featurizer.featurize_smiles(["CCO"])

    assert batch.valid_mask.tolist() == [True]
    assert batch.X.shape == (1, 8)
    assert batch.X.dtype == np.float32
    batch.check(n_inputs=1)


def test_modernmolbert_selfies_featurizer_rejects_unknown_pooling(
    tiny_modernmolbert_dir,
):
    with pytest.raises(ValueError, match="pooling|Unsupported"):
        ModernMolBERTSelfiesFeaturizer(
            model_dir=tiny_modernmolbert_dir,
            tokenizer_path=tiny_modernmolbert_dir,
            max_seq_length=32,
            batch_size=2,
            device="cpu",
            pooling="not_a_pooling_strategy",  # type: ignore[arg-type]
        )


def test_mean_pooling_excludes_special_tokens(tiny_modernmolbert_dir):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
        pooling="mean",
    )

    hidden = torch.tensor(
        [
            [
                [100.0, 100.0],  # BOS, should be excluded
                [1.0, 3.0],  # content
                [3.0, 5.0],  # content
                [200.0, 200.0],  # EOS, should be excluded
                [300.0, 300.0],  # PAD, should be excluded
            ]
        ]
    )

    input_ids = torch.tensor(
        [
            [
                featurizer.tokenizer.bos_token_id,
                5,
                6,
                featurizer.tokenizer.eos_token_id,
                featurizer.tokenizer.pad_token_id,
            ]
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)

    pooled = mean_pool_excluding_token_ids(
        last_hidden_state=hidden,
        attention_mask=attention_mask,
        input_ids=input_ids,
        excluded_token_ids=featurizer._special_token_ids(),
    )

    expected = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(pooled, expected)


def test_mean_pooling_falls_back_to_attention_mask_when_no_content_tokens(
    tiny_modernmolbert_dir,
):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
        pooling="mean",
    )

    hidden = torch.tensor(
        [
            [
                [1.0, 3.0],  # BOS
                [3.0, 5.0],  # EOS
                [9.0, 9.0],  # PAD, attention mask excludes
            ]
        ]
    )

    input_ids = torch.tensor(
        [
            [
                featurizer.tokenizer.bos_token_id,
                featurizer.tokenizer.eos_token_id,
                featurizer.tokenizer.pad_token_id,
            ]
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    pooled = mean_pool_excluding_token_ids(
        last_hidden_state=hidden,
        attention_mask=attention_mask,
        input_ids=input_ids,
        excluded_token_ids=featurizer._special_token_ids(),
    )

    expected = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(pooled, expected)


def test_modernmolbert_selfies_featurizer_records_metadata(tiny_modernmolbert_dir):
    featurizer = ModernMolBERTSelfiesFeaturizer(
        name="modernmolbert_pilot_test",
        model_dir=tiny_modernmolbert_dir,
        tokenizer_path=tiny_modernmolbert_dir,
        max_seq_length=32,
        batch_size=2,
        device="cpu",
        pooling="mean",
    )

    batch = featurizer.featurize_smiles(["CCO", "not_a_smiles"])

    assert batch.metadata["featurizer"] == "modernmolbert_pilot_test"
    assert batch.metadata["backend"] == "modernmolbert_selfies"
    assert batch.metadata["pooling"] == "mean"
    assert batch.metadata["max_seq_length"] == 32
    assert batch.metadata["n_inputs"] == 2
    assert batch.metadata["n_valid"] == 1
    assert batch.metadata["invalid_fraction"] == pytest.approx(0.5)
