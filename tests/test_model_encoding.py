"""Model encoder output tests.

These tests verify that a trained checkpoint produces finite embeddings.
They are skipped automatically when no trained model exists under runs/*/final_model.
No environment variable is required — just train a model first.
"""

import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

_TEXTS = [
    "[C]",
    "[O]",
    "[C][C][O]",
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
]


def test_existing_minimal_model_encoder_output(existing_minimal_model: Path) -> None:
    """Verify that a trained checkpoint produces finite encoder embeddings.

    Tests the base ModernBERT encoder output (last_hidden_state + mean-pooled
    embedding), not the MLM logits. Skipped if no trained model is found.
    """
    verbose = os.environ.get("MODERNMOLBERT_TEST_VERBOSE") == "1"

    model = AutoModel.from_pretrained(existing_minimal_model)
    model.eval()

    tokenizer_dir = existing_minimal_model / "ape_tokenizer"
    if tokenizer_dir.exists():
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tok = APEPreTrainedTokenizer()
        tok.load_vocabulary_file(existing_minimal_model / "vocab.json")

    if verbose:
        print(f"\n[encoder-test] model_dir={existing_minimal_model}")
        print(f"[encoder-test] vocab_size={model.config.vocab_size}")
        print(f"[encoder-test] hidden_size={model.config.hidden_size}")
        print(f"[encoder-test] num_layers={model.config.num_hidden_layers}")

    unk_id = tok.vocabulary[str(tok.unk_token)]

    for text in _TEXTS:
        batch = tok(text, add_special_tokens=True, return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        token_strings = tok.convert_ids_to_tokens(ids)
        unk_positions = [i for i, t in enumerate(ids) if t == unk_id]
        assert not unk_positions, (text, ids, token_strings, unk_positions)

        with torch.no_grad():
            out = model(**batch)

        hidden = out.last_hidden_state

        assert hidden.ndim == 3, text
        assert hidden.shape == (
            1,
            batch["input_ids"].shape[1],
            model.config.hidden_size,
        ), text
        assert torch.isfinite(hidden).all(), text

        # Mean-pooled molecular embedding.
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        assert pooled.shape == (1, model.config.hidden_size), text
        assert torch.isfinite(pooled).all(), text

        if verbose:
            print(
                f"[encoder-test] text={text!r}"
                f" | tokens={token_strings}"
                f" | ids={ids}"
                f" | seq_len={hidden.shape[1]}"
                f" | pooled_norm={pooled.norm().item():.4f}"
            )
