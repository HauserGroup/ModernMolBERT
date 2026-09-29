from pathlib import Path


from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import assert_representation_compatible
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    collect_corpus_for_tokenizer,
    compute_tokenization_stats,
    find_local_dataset,
    eligible_token_ids,
    file_sha256,
    ignored_special_token_ids,
    metadata_path_for_vocab,
    resolve_special_ids,
    sample_jsonl_sequences,
    validate_selfies_sample_shape,
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


def _incomplete_vocab_tokenizer() -> APEPreTrainedTokenizer:
    # Valid SELFIES vocab that is missing every token in the test sample
    # ([C], [O]), so encoding those sequences yields all-<unk>. (A vocab with
    # bare non-bracket tokens like "C" is now rejected outright by the strict
    # SELFIES pre-tokenizer, so it can no longer stand in for a "broken" vocab.)
    tok = APEPreTrainedTokenizer()
    tok.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[N]": 5,
        "[P]": 6,
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
    tokenizer.save_vocabulary_file(vocab_path)
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


def test_unk_rate_counts_unknown_tokens_when_unk_is_special():
    tokenizer = _incomplete_vocab_tokenizer()
    special_ids = resolve_special_ids(tokenizer)

    stats = compute_tokenization_stats(
        tokenizer=tokenizer,
        sequences=["[C][C][O]"],
        max_seq_length=64,
        special_ids=special_ids,
    )

    # [C] and [O] are absent from the vocab, so every token is unknown.
    assert stats["unk_rate"] > 0.5

    encoded = tokenizer("[C][C][O]", add_special_tokens=True)["input_ids"]
    eligible = eligible_token_ids(encoded, special_ids)
    assert len(eligible) == 3
    assert sum(1 for x in eligible if x == special_ids["unk_token"]) == 3


def test_ethanol_gate_fails_when_selfies_symbols_are_unknown():
    tokenizer = _incomplete_vocab_tokenizer()
    special_ids = resolve_special_ids(tokenizer)

    try:
        assert_representation_compatible(tokenizer, special_ids, SELFIES_REPRESENTATION)
        raise AssertionError("Expected ethanol gate to fail for broken tokenizer")
    except ValueError as exc:
        assert "unk_rate" in str(exc)


def test_selfies_encoding_does_not_split_brackets():

    tok = APEPreTrainedTokenizer()

    tok.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[C][C]": 7,
    }

    tok.update_reverse_vocabulary()

    ids = tok.encode("[C][C][O]", add_special_tokens=True)

    assert ids == [0, 7, 6, 2]

    assert tok.vocabulary["<unk>"] not in ids


def test_ignored_special_token_ids_excludes_unk_token():
    special_ids = {
        "bos_token": 0,
        "pad_token": 1,
        "eos_token": 2,
        "unk_token": 3,
        "mask_token": 4,
    }

    ignored = ignored_special_token_ids(special_ids)

    assert special_ids["pad_token"] in ignored
    assert special_ids["bos_token"] in ignored
    assert special_ids["eos_token"] in ignored
    assert special_ids["mask_token"] in ignored
    assert special_ids["unk_token"] not in ignored


def test_find_local_dataset_raises_for_invalid_explicit_dir(tmp_path: Path):
    missing = tmp_path / "not_a_dataset"
    missing.mkdir(parents=True, exist_ok=True)

    try:
        find_local_dataset(data_dir=missing, dataset_name=PUBCHEM10M_DATASET)
        raise AssertionError("Expected FileNotFoundError for explicit invalid data_dir")
    except FileNotFoundError as exc:
        assert "dataset_info.json" in str(exc)


def test_collect_corpus_passes_data_files(monkeypatch):
    captured: dict[str, object] = {}

    class _DummyStream:
        def __iter__(self):
            return iter(
                [
                    {"SELFIES": "[C][C][O]"},
                    {"SELFIES": "[C][O][C]"},
                ]
            )

    def _fake_get_streaming_dataset(*args, **kwargs):
        captured.update(kwargs)
        return _DummyStream()

    monkeypatch.setattr(
        "modernmolbert.utils.get_streaming_dataset",
        _fake_get_streaming_dataset,
    )

    corpus = collect_corpus_for_tokenizer(
        dataset_name=PUBCHEM10M_DATASET,
        representation="SELFIES",
        n=2,
        seed=13,
        buffer_size=100,
        data_files="/tmp/data/*.parquet",
    )

    assert corpus == ["[C][C][O]", "[C][O][C]"]
    assert captured.get("data_files") == "/tmp/data/*.parquet"
