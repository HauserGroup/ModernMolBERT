# tests/test_eval_modernmolbert_smiles.py
"""SMILES-native path of the ModernMolBERT featurizer.

The SELFIES path of the same featurizer is covered in
test_eval_modernmolbert_selfies.py; here we assert the SMILES checkpoint feeds raw
SMILES to the tokenizer with no SMILES->SELFIES round-trip.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import modernmolbert.eval.featurizers.modernmolbert_selfies as mm_selfies
from modernmolbert.eval.featurizers.modernmolbert_selfies import (
    ModernMolBERTSelfiesFeaturizer,
)


class RecordingTokenizer:
    """Tokenizer stub that records the exact strings it was asked to encode."""

    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 4

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, texts, *, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        self.seen.extend(texts)
        ids = [[self.bos_token_id, 5, self.eos_token_id] for _ in texts]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
        }


class TinyModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden_size = self.config.hidden_size
        values = input_ids.float().unsqueeze(-1).repeat(1, 1, hidden_size)
        return SimpleNamespace(last_hidden_state=values)


def _make_featurizer(tmp_path: Path, monkeypatch, tokenizer: RecordingTokenizer):
    model_dir = tmp_path / "tiny-smiles"
    model_dir.mkdir()
    monkeypatch.setattr(
        mm_selfies, "_load_ape_tokenizer", lambda path, representation="SELFIES": tokenizer
    )
    monkeypatch.setattr(
        "modernmolbert.eval.featurizers.modernmolbert_selfies.AutoModel.from_pretrained",
        lambda path, **kwargs: TinyModel(hidden_size=8),
    )
    return ModernMolBERTSelfiesFeaturizer(
        model_dir=model_dir,
        tokenizer_path=model_dir,
        max_seq_length=32,
        batch_size=4,
        device="cpu",
        representation="SMILES",
    )


def test_smiles_featurizer_passes_raw_smiles_to_tokenizer(tmp_path, monkeypatch):
    tokenizer = RecordingTokenizer()
    featurizer = _make_featurizer(tmp_path, monkeypatch, tokenizer)

    batch = featurizer.featurize_smiles(["CCO", "c1ccccc1"])

    # Raw SMILES reach the tokenizer unchanged (SELFIES encoder would give "[C][C][O]").
    assert tokenizer.seen == ["CCO", "c1ccccc1"]
    assert batch.valid_mask.tolist() == [True, True]
    assert batch.X.shape == (2, 8)
    assert batch.X.dtype == np.float32
    assert batch.metadata["representation"] == "SMILES"
    assert batch.metadata["backend"] == "modernmolbert_smiles"


def test_smiles_featurizer_never_imports_selfies(tmp_path, monkeypatch):
    # Poison the selfies module: if the SMILES path tried to convert, it would raise.
    poison = SimpleNamespace(encoder=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setitem(sys.modules, "selfies", poison)

    tokenizer = RecordingTokenizer()
    featurizer = _make_featurizer(tmp_path, monkeypatch, tokenizer)

    batch = featurizer.featurize_smiles(["CCO"])
    assert tokenizer.seen == ["CCO"]
    assert batch.valid_mask.tolist() == [True]


def test_smiles_featurizer_marks_empty_and_none_invalid(tmp_path, monkeypatch):
    tokenizer = RecordingTokenizer()
    featurizer = _make_featurizer(tmp_path, monkeypatch, tokenizer)

    batch = featurizer.featurize_smiles(["CCO", "", None])
    assert batch.valid_mask.tolist() == [True, False, False]
    assert tokenizer.seen == ["CCO"]
    assert batch.X.shape == (1, 8)
    batch.check(n_inputs=3)
