from pathlib import Path

import torch
from transformers import AutoModel, BertConfig, BertModel

from modernmolbert.ape_tokenizer import APETokenizer


def _tokenizer_for_representation(representation: str) -> tuple[APETokenizer, str]:
    tokenizer = APETokenizer(representation=representation)
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
