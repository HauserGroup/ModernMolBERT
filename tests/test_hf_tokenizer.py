"""Tests for the HuggingFace Hub-hosted APE SELFIES tokenizer.

Enable with:
    HF_TOKEN=<token> pytest tests/test_hf_tokenizer.py -q -s

Skipped automatically when the HF tokenizer hub tests flag is not set,
to avoid requiring network access in CI.
"""

import os

import pytest


HF_TOKENIZER_REPO = "HauserGroup/ApeTokenizer-SELFIES"

_SELFIES_EXAMPLES = [
    "[C][C][O]",
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    "[C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C]",
    "[O][=C][Branch1][C][O][C][C][O]",
]


def _hub_enabled() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def _hub_tokenizer():
    """Load the tokenizer from the Hub (network call)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        HF_TOKENIZER_REPO,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )


@pytest.mark.network
def test_hf_ape_tokenizer_loads_from_hub() -> None:
    """Tokenizer loads from Hub and exposes a positive vocab_size."""
    if not _hub_enabled():
        pytest.skip("Set HF_TOKEN to enable Hub tokenizer tests.")

    tok = _hub_tokenizer()

    print(f"\n  repo:       {HF_TOKENIZER_REPO}")
    print(f"  class:      {type(tok).__name__}")
    print(f"  vocab_size: {tok.vocab_size}")

    assert tok.vocab_size > 0, f"Expected positive vocab_size, got {tok.vocab_size}"


@pytest.mark.network
def test_hf_ape_tokenizer_has_required_special_tokens() -> None:
    """Tokenizer exposes pad, mask, bos, eos and unk tokens."""
    if not _hub_enabled():
        pytest.skip("Set HF_TOKEN to enable Hub tokenizer tests.")

    tok = _hub_tokenizer()

    for attr in ("pad_token_id", "mask_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
        value = getattr(tok, attr, None)
        print(f"  {attr}: {value}")
        assert value is not None, f"{attr} is None"
        assert isinstance(value, int), f"{attr} should be int, got {type(value)}"


@pytest.mark.network
@pytest.mark.parametrize("selfies", _SELFIES_EXAMPLES)
def test_hf_ape_tokenizer_encodes_selfies_without_unk(selfies: str) -> None:
    """Each example SELFIES encodes with zero unknown tokens."""
    if not _hub_enabled():
        pytest.skip("Set HF_TOKEN to enable Hub tokenizer tests.")

    tok = _hub_tokenizer()
    unk_id = tok.unk_token_id

    encoded = tok(selfies, add_special_tokens=True, return_tensors=None)
    ids = encoded["input_ids"]

    unk_count = sum(1 for x in ids if x == unk_id)
    unk_rate = unk_count / max(1, len(ids))

    print(f"\n  selfies:   {selfies}")
    print(f"  ids:       {ids}")
    print(f"  length:    {len(ids)}")
    print(f"  unk_count: {unk_count}  (rate={unk_rate:.3f})")

    assert len(ids) >= 3, f"Expected at least BOS + 1 content token + EOS, got {ids}"
    assert unk_count == 0, (
        f"Unexpected UNK tokens in encoded SELFIES.\n"
        f"  selfies: {selfies}\n"
        f"  ids: {ids}\n"
        f"  unk_id={unk_id}, unk_count={unk_count}"
    )


@pytest.mark.network
def test_hf_ape_tokenizer_roundtrip_decode() -> None:
    """Encoded tokens decode back to non-empty strings for each example."""
    if not _hub_enabled():
        pytest.skip("Set HF_TOKEN to enable Hub tokenizer tests.")

    tok = _hub_tokenizer()

    for selfies in _SELFIES_EXAMPLES:
        ids = tok(selfies, add_special_tokens=False, return_tensors=None)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)

        print(f"\n  selfies:  {selfies}")
        print(f"  decoded:  {decoded!r}")

        assert decoded.strip(), f"Decoded string is empty for selfies={selfies!r}, ids={ids}"
