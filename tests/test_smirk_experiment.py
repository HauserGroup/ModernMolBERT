"""SMIRK pilot compatibility checks; skipped in the main dependency environment."""

from pathlib import Path

import pytest

from modernmolbert.tokenization.load import load_checkpoint_tokenizer, load_verified_tokenizer
from modernmolbert.utils import (
    compute_tokenization_stats,
    copy_tokenizer_artifacts,
    resolve_special_ids,
)

pytest.importorskip("smirk")

TOKENIZER = (
    Path(__file__).resolve().parents[1] / "tokenizer/experimental_smirk_v1/smirk_smiles.json"
)
EXAMPLES = ["CCO", "Cl[Pt@SP1](Cl)([NH3])[NH3]", "[O-][99Tc](=O)(=O)=O.[Na+]"]


def test_smirk_bundle_and_checkpoint_reload(tmp_path: Path) -> None:
    tokenizer, metadata, vocab_path, metadata_path = load_verified_tokenizer(TOKENIZER)
    special_ids = resolve_special_ids(tokenizer)
    assert metadata["algorithm"] == "SMIRK"
    assert metadata["representation"] == "SMILES"
    assert special_ids == metadata["special_ids"]
    assert len(set(special_ids.values())) == 5
    for smiles in EXAMPLES:
        ids = tokenizer(smiles)["input_ids"]
        assert ids[0] == special_ids["bos_token"]
        assert ids[-1] == special_ids["eos_token"]
        assert special_ids["unk_token"] not in ids
        assert "".join(tokenizer.tokenize(smiles)) == smiles
        assert tokenizer.decode(ids, skip_special_tokens=True) == smiles

    stats = compute_tokenization_stats(tokenizer, EXAMPLES, 384, special_ids)
    assert stats["unk_rate"] == 0
    assert stats["silent_loss_rate"] == 0

    final_dir = tmp_path / "final_model"
    copy_tokenizer_artifacts(vocab_path, metadata_path, tmp_path, final_dir, 384)
    reloaded, representation = load_checkpoint_tokenizer(final_dir)
    assert representation == "SMILES"
    for smiles in EXAMPLES:
        assert reloaded(smiles)["input_ids"] == tokenizer(smiles)["input_ids"]
