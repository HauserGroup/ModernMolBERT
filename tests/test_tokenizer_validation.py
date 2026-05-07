from pathlib import Path

from modernmolbert.ape_tokenizer import APETokenizer
from modernmolbert.utils import (
    SELFIES_REPRESENTATION,
    compute_tokenization_stats,
    file_sha256,
    metadata_path_for_vocab,
    resolve_special_ids,
    sample_jsonl_sequences,
    validate_selfies_sample_shape,
    write_tokenizer_metadata,
)


def _tiny_tokenizer() -> APETokenizer:
    tok = APETokenizer()
    tok.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[=Branch1]": 7,
        "[=O]": 8,
        "[=C]": 9,
        "[Ring1]": 10,
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


def test_fixture_sampling_and_selfies_shape_validation():
    fixture = Path(__file__).resolve().parent / "fixtures" / "tiny_pubchem_like.jsonl"
    seqs = sample_jsonl_sequences(fixture, SELFIES_REPRESENTATION, n=3)
    assert len(seqs) == 3
    validate_selfies_sample_shape(seqs)


def test_tokenization_stats_and_metadata_helpers(tmp_path: Path):
    tokenizer = _tiny_tokenizer()
    special_ids = resolve_special_ids(tokenizer)

    sequences = ["[C][C][O]", "[C][C][=Branch1][C][=O][O]"]
    stats = compute_tokenization_stats(
        tokenizer=tokenizer,
        sequences=sequences,
        max_seq_length=32,
        special_ids=special_ids,
    )

    assert stats["unk_rate"] <= 0.001
    assert stats["mean_len"] > 0
    assert stats["truncation_rate"] == 0.0

    vocab_path = tmp_path / "selfies_ape_tokenizer.json"
    tokenizer.save_vocabulary(str(vocab_path))
    metadata_path = metadata_path_for_vocab(vocab_path)
    write_tokenizer_metadata(
        metadata_path,
        {
            "representation": "SELFIES",
            "tokenizer_sha256": file_sha256(vocab_path),
            "tokenizer_path": str(vocab_path),
        },
    )

    assert metadata_path.exists()
