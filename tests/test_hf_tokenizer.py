"""Tests for the HuggingFace Hub-hosted APE SELFIES tokenizer.

Enable with:
    HF_TOKEN=<token> pytest tests/test_hf_tokenizer.py -q -s

Skipped automatically when HF_TOKEN is not set, to avoid requiring network access in CI.
"""

import os

import pytest

HF_TOKENIZER_REPO = "HauserGroup/ApeTokenizer-SELFIES"

_SELFIES_EXAMPLES = [
    "[C][C][O]",
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    "[O][=C][Branch1][C][O][C][C][O]",
]


@pytest.mark.network
def test_hf_ape_tokenizer_encodes_and_decodes_selfies() -> None:
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("Set HF_TOKEN to enable Hub tokenizer tests.")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        HF_TOKENIZER_REPO, token=os.environ["HF_TOKEN"], trust_remote_code=True
    )

    assert tok.vocab_size > 0
    for attr in ("pad_token_id", "mask_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
        assert isinstance(getattr(tok, attr, None), int), attr

    for selfies in _SELFIES_EXAMPLES:
        ids = tok(selfies, add_special_tokens=True)["input_ids"]
        assert len(ids) >= 3
        assert tok.unk_token_id not in ids, selfies
        assert tok.decode(ids, skip_special_tokens=True).strip(), selfies
