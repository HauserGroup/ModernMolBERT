import os
from pathlib import Path

import pytest

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.validate_tokenizer import _assert_ethanol_not_unknown
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    ZINC20_CHEMBL36_DATASET,
    ZINC20_DATASET,
    collect_corpus_for_tokenizer,
    compute_tokenization_stats,
    find_local_dataset,
    get_streaming_dataset,
    eligible_token_ids,
    file_sha256,
    infer_selfies_column,
    infer_validation_split,
    ignored_special_token_ids,
    metadata_path_for_vocab,
    normalize_sequence,
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


def _broken_bracket_tokenizer() -> APEPreTrainedTokenizer:
    tok = APEPreTrainedTokenizer()
    tok.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "C": 5,
        "O": 6,
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
    tokenizer = _broken_bracket_tokenizer()
    special_ids = resolve_special_ids(tokenizer)

    stats = compute_tokenization_stats(
        tokenizer=tokenizer,
        sequences=["[C][C][O]"],
        max_seq_length=64,
        special_ids=special_ids,
    )

    # Expect unknowns from bracket characters to be counted.
    assert stats["unk_rate"] > 0.5

    encoded = tokenizer("[C][C][O]", add_special_tokens=True)["input_ids"]
    eligible = eligible_token_ids(encoded, special_ids)
    assert len(eligible) == 3
    assert sum(1 for x in eligible if x == special_ids["unk_token"]) == 3


def test_ethanol_gate_fails_when_selfies_symbols_are_unknown():
    tokenizer = _broken_bracket_tokenizer()
    special_ids = resolve_special_ids(tokenizer)

    try:
        _assert_ethanol_not_unknown(tokenizer, special_ids)
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


def test_infer_selfies_column_for_pubchem_and_zinc20():
    assert infer_selfies_column(PUBCHEM10M_DATASET, None) == "SELFIES"
    assert infer_selfies_column(ZINC20_DATASET, None) == "SELFIES"
    # zinc20_chembl36 uses lowercase "selfies" column
    assert infer_selfies_column(ZINC20_CHEMBL36_DATASET, None) == "selfies"
    assert infer_selfies_column(PUBCHEM10M_DATASET, "my_col") == "my_col"


def test_infer_selfies_column_for_local_metadata(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "data" / "pretrain" / "chembl36_selfies"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.parquet").write_text("placeholder", encoding="utf-8")
    (dataset_dir / "metadata.json").write_text(
        '{"selfies_column": "selfies"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("modernmolbert.utils.repo_root", lambda: tmp_path)

    assert infer_selfies_column("data/pretrain/chembl36_selfies", None) == "selfies"


def test_infer_validation_split_for_pubchem_and_zinc20():
    assert infer_validation_split(PUBCHEM10M_DATASET, None) is None
    assert infer_validation_split(ZINC20_DATASET, None) == "validation"
    assert infer_validation_split(ZINC20_CHEMBL36_DATASET, None) is None
    assert infer_validation_split(PUBCHEM10M_DATASET, "dev") == "dev"


def test_normalize_sequence_supports_pubchem_and_zinc20_column_names():
    assert normalize_sequence({"SELFIES": "[C][O]"}, "SELFIES") == "[C][O]"
    assert normalize_sequence({"selfies": "[C][O]"}, "selfies") == "[C][O]"
    assert normalize_sequence({"SELFIES": "   "}, "SELFIES") is None


def test_find_local_dataset_raises_for_invalid_explicit_dir(tmp_path: Path):
    missing = tmp_path / "not_a_dataset"
    missing.mkdir(parents=True, exist_ok=True)

    try:
        find_local_dataset(data_dir=missing, dataset_name=PUBCHEM10M_DATASET)
        raise AssertionError("Expected FileNotFoundError for explicit invalid data_dir")
    except FileNotFoundError as exc:
        assert "dataset_info.json" in str(exc) or "parquet" in str(exc)


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


def test_find_local_dataset_resolves_path_like_dataset_name(tmp_path: Path, monkeypatch):
    dataset_name = "chembl36_selfies_tmp_local_test"
    dataset_dir = tmp_path / "data" / "pretrain" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.parquet").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("modernmolbert.utils.repo_root", lambda: tmp_path)

    resolved = find_local_dataset(dataset_name=f"data/pretrain/{dataset_name}")

    assert resolved == dataset_dir


def test_find_local_dataset_path_like_fails_fast_when_missing(monkeypatch):
    monkeypatch.setattr("modernmolbert.utils.repo_root", lambda: Path("/tmp/does-not-exist"))

    try:
        find_local_dataset(dataset_name="data/pretrain/missing")
        raise AssertionError("Expected FileNotFoundError for missing local dataset path")
    except FileNotFoundError as exc:
        assert "Local dataset path not found" in str(exc)


def test_get_streaming_dataset_uses_local_parquet_split(tmp_path: Path, monkeypatch):
    import pandas as pd

    dataset_name = "chembl36_selfies_tmp_stream_test"
    dataset_dir = tmp_path / "data" / "pretrain" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SELFIES": ["[C][C][O]", "[C][O][C]"]}).to_parquet(
        dataset_dir / "train.parquet",
        index=False,
    )

    monkeypatch.setattr("modernmolbert.utils.repo_root", lambda: tmp_path)

    ds = get_streaming_dataset(
        dataset_name=f"data/pretrain/{dataset_name}",
        seed=13,
        buffer_size=10,
        split="train",
    )
    first = next(iter(ds))

    assert "SELFIES" in first


@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TOKENIZER_TESTS") != "1",
    reason="Optional slow integration test; set RUN_SLOW_TOKENIZER_TESTS=1 to run.",
)
def test_collect_corpus_from_local_parquet_dataset(tmp_path: Path, monkeypatch):
    import pandas as pd

    dataset_name = "chembl36_selfies_tmp_corpus_test"
    dataset_dir = tmp_path / "data" / "pretrain" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SELFIES": ["[C][C][O]", "[C][O][C]", "[C][C][C]"]}).to_parquet(
        dataset_dir / "train.parquet",
        index=False,
    )

    monkeypatch.setattr("modernmolbert.utils.repo_root", lambda: tmp_path)

    corpus = collect_corpus_for_tokenizer(
        dataset_name=f"data/pretrain/{dataset_name}",
        representation="SELFIES",
        n=2,
        seed=13,
        buffer_size=10,
    )

    assert len(corpus) == 2
    assert all(seq.startswith("[") for seq in corpus)
