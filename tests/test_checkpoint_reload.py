from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, ModernBertConfig

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    copy_tokenizer_artifacts,
    file_sha256,
    write_tokenizer_metadata,
)


def _tiny_tokenizer() -> APEPreTrainedTokenizer:
    tok = APEPreTrainedTokenizer()
    tok.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
    }
    tok.special_tokens = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
    }
    tok.update_reverse_vocabulary()
    return tok


def test_end_to_end_save_and_reload_with_tokenizer_artifacts(tmp_path: Path):
    tokenizer = _tiny_tokenizer()

    vocab_path = tmp_path / "selfies_ape_tokenizer.json"
    tokenizer.save_vocabulary_file(vocab_path)
    metadata_path = tmp_path / "selfies_ape_tokenizer.metadata.json"
    write_tokenizer_metadata(
        metadata_path,
        {
            "representation": "SELFIES",
            "tokenizer_sha256": file_sha256(vocab_path),
            "tokenizer_path": str(vocab_path),
        },
    )

    config = ModernBertConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = AutoModelForMaskedLM.from_config(config)

    batch = tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
    batch = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    out = model(**batch, labels=labels)
    out.loss.backward()

    output_dir = tmp_path / "run"
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_model_dir))

    copy_tokenizer_artifacts(vocab_path, metadata_path, output_dir, final_model_dir)

    for expected in [
        "vocab.json",
        "tokenizer_metadata.json",
        "ape_tokenizer/vocab.json",
        "ape_tokenizer/tokenizer_config.json",
        "ape_tokenizer/special_tokens_map.json",
        "ape_tokenizer/tokenization_ape.py",
    ]:
        assert (final_model_dir / expected).exists()
    assert not (final_model_dir / "tokenizer.json").exists()
    assert not (final_model_dir / "tokenizer_config.json").exists()

    reloaded_model = AutoModelForMaskedLM.from_pretrained(str(final_model_dir))
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        str(final_model_dir / "ape_tokenizer"),
        trust_remote_code=True,
    )

    eval_batch = reloaded_tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
    eval_batch = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in eval_batch.items()}
    with torch.no_grad():
        logits = reloaded_model(**eval_batch).logits

    assert torch.isfinite(logits).all()
    assert logits.shape[0] == 1
    assert logits.shape[1] == eval_batch["input_ids"].shape[1]

    auto_batch = reloaded_tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
    assert auto_batch["input_ids"].shape == eval_batch["input_ids"].shape
