from pathlib import Path

import pytest
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

from modernmolbert.ape_tokenizer import APETokenizer
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


def _tokenizer_for_representation(
    representation: str,
) -> tuple[APEPreTrainedTokenizer, str]:
    tokenizer = APEPreTrainedTokenizer(representation=representation)
    if representation == "SELFIES":
        tokenizer.vocabulary = {
            **tokenizer.special_tokens,
            "[C]": 5,
            "[O]": 6,
            "[C][C]": 7,
        }
        text = "[C][C][O]"
    else:
        tokenizer.vocabulary = {
            **tokenizer.special_tokens,
            "C": 5,
            "O": 6,
            "CC": 7,
            "CCO": 8,
        }
        text = "CCO"

    tokenizer.update_reverse_vocabulary()
    return tokenizer, text


def test_ape_tokenizer_outputs_feed_transformers_models_for_selfies_and_smiles(
    tmp_path: Path,
):
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model_dir = tmp_path / "bert"
    BertModel(config).save_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    for representation in ("SELFIES", "SMILES"):
        tokenizer, text = _tokenizer_for_representation(representation)

        batch = tokenizer(text, add_special_tokens=True, return_tensors="pt")

        assert batch["input_ids"].ndim == 2
        assert batch["attention_mask"].ndim == 2
        assert batch["input_ids"].shape[0] == 1

        with torch.no_grad():
            output = model(**batch)

        assert output.last_hidden_state.shape[:2] == batch["input_ids"].shape
        assert torch.isfinite(output.last_hidden_state).all()


def test_ape_pretrained_tokenizer_matches_legacy_tokenizer_for_selfies_and_smiles():
    for representation in ("SELFIES", "SMILES"):
        hf_tokenizer, text = _tokenizer_for_representation(representation)
        with pytest.warns(DeprecationWarning):
            legacy = APETokenizer(representation=representation)
        legacy.vocabulary = hf_tokenizer.vocabulary
        legacy.update_reverse_vocabulary()
        hf_tokenizer = APEPreTrainedTokenizer(
            vocab=legacy.vocabulary,
            representation=representation,
            bos_token=legacy.bos_token,
            eos_token=legacy.eos_token,
            unk_token=legacy.unk_token,
            pad_token=legacy.pad_token,
            mask_token=legacy.mask_token,
        )

        assert hf_tokenizer.encode(text, add_special_tokens=False) == legacy.encode(
            text,
            add_special_tokens=False,
        )
        assert hf_tokenizer.encode(text, add_special_tokens=True) == legacy.encode(
            text,
            add_special_tokens=True,
        )


def test_auto_tokenizer_loads_custom_ape_tokenizer_for_selfies_and_smiles(tmp_path: Path):
    for representation in ("SELFIES", "SMILES"):
        tokenizer, text = _tokenizer_for_representation(representation)
        tokenizer_dir = tmp_path / representation.lower()
        tokenizer.save_pretrained(str(tokenizer_dir))

        loaded = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
        )

        assert loaded.__class__.__name__ == "APEPreTrainedTokenizer"
        assert loaded.representation == representation
        assert loaded.encode(text, add_special_tokens=False) == tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        batch = loaded(text, add_special_tokens=True, return_tensors="pt")
        assert batch["input_ids"].ndim == 2
        assert batch["attention_mask"].shape == batch["input_ids"].shape
