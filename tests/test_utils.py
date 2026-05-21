"""Tests for modernmolbert.utils — private helpers and pure functions not covered elsewhere."""

import json
from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    _available_local_parquet_splits,
    _is_local_dataset_dir,
    _local_dataset_matches_request,
    _looks_like_path,
    _normalized_name,
    _resolve_dataset_name_as_local_path,
    _split_parquet_files,
    assert_metadata_representation,
    collect_local_parquet_corpus,
    encode_sequence,
    filter_zinc20_chembl36_by_source,
    token_id,
    tokenizer_vocab_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer(vocab: dict[str, int] | None = None) -> APEPreTrainedTokenizer:
    tok = APEPreTrainedTokenizer()
    tok.vocabulary = vocab or {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[N]": 7,
    }
    tok.special_tokens = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    tok.update_reverse_vocabulary()
    return tok


# ---------------------------------------------------------------------------
# _normalized_name
# ---------------------------------------------------------------------------


def test_normalized_name_strips_non_alphanumeric_and_lowercases() -> None:

    assert _normalized_name("PubChem10M_SMILES_SELFIES") == "pubchem10msmilesselfies"
    assert _normalized_name("my-dataset/v2") == "mydatasetv2"
    assert _normalized_name("ABC123") == "abc123"


# ---------------------------------------------------------------------------
# _looks_like_path
# ---------------------------------------------------------------------------


def test_looks_like_path_absolute() -> None:
    assert _looks_like_path("/home/user/data") is True


def test_looks_like_path_relative_with_slash() -> None:
    assert _looks_like_path("data/something") is True


def test_looks_like_path_dot_prefix() -> None:
    assert _looks_like_path("./local_data") is True


def test_looks_like_path_plain_hf_repo() -> None:
    assert _looks_like_path("mikemayuare/PubChem10M_SMILES_SELFIES") is True


def test_looks_like_path_plain_name_no_slash() -> None:
    assert _looks_like_path("pubchem10m") is False


# ---------------------------------------------------------------------------
# _is_local_dataset_dir
# ---------------------------------------------------------------------------


def test_is_local_dataset_dir_with_dataset_info_json(tmp_path: Path) -> None:
    (tmp_path / "dataset_info.json").write_text("{}", encoding="utf-8")
    assert _is_local_dataset_dir(tmp_path) is True


def test_is_local_dataset_dir_with_parquet_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    df.to_parquet(tmp_path / "train.parquet")
    assert _is_local_dataset_dir(tmp_path) is True


def test_is_local_dataset_dir_empty_dir(tmp_path: Path) -> None:
    assert _is_local_dataset_dir(tmp_path) is False


def test_is_local_dataset_dir_nonexistent(tmp_path: Path) -> None:
    assert _is_local_dataset_dir(tmp_path / "missing") is False


# ---------------------------------------------------------------------------
# _local_dataset_matches_request
# ---------------------------------------------------------------------------


def test_local_dataset_matches_request_by_dir_name(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pubchem10m"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"dataset_name": "pubchem10m"}), encoding="utf-8"
    )
    assert _local_dataset_matches_request(dataset_dir, "PubChem10M_SMILES_SELFIES") is True


def test_local_dataset_matches_request_no_info_file(tmp_path: Path) -> None:
    d = tmp_path / "myds"
    d.mkdir()
    assert _local_dataset_matches_request(d, "myds") is False


def test_local_dataset_matches_request_mismatch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "zinc20"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"dataset_name": "zinc20"}), encoding="utf-8"
    )
    assert _local_dataset_matches_request(dataset_dir, "pubchem10m") is False


# ---------------------------------------------------------------------------
# _available_local_parquet_splits
# ---------------------------------------------------------------------------


def test_available_splits_finds_train(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    df.to_parquet(tmp_path / "train.parquet")
    splits = _available_local_parquet_splits(tmp_path)
    assert "train" in splits


def test_available_splits_finds_validation_via_alias(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    df.to_parquet(tmp_path / "validation.parquet")
    splits = _available_local_parquet_splits(tmp_path)
    assert "valid" in splits or "validation" in splits


def test_available_splits_sharded_train(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    df.to_parquet(tmp_path / "train-00001-of-00002.parquet")
    splits = _available_local_parquet_splits(tmp_path)
    assert "train" in splits


def test_available_splits_empty_dir(tmp_path: Path) -> None:
    assert _available_local_parquet_splits(tmp_path) == set()


# ---------------------------------------------------------------------------
# _split_parquet_files
# ---------------------------------------------------------------------------


def test_split_parquet_files_single_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    expected = tmp_path / "train.parquet"
    df.to_parquet(expected)
    files = _split_parquet_files(tmp_path, "train")
    assert files == [expected]


def test_split_parquet_files_sharded(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    shard1 = tmp_path / "train-00001.parquet"
    shard2 = tmp_path / "train-00002.parquet"
    df.to_parquet(shard1)
    df.to_parquet(shard2)
    files = _split_parquet_files(tmp_path, "train")
    assert set(files) == {shard1, shard2}


def test_split_parquet_files_alias_valid_finds_validation(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    val_file = tmp_path / "validation.parquet"
    df.to_parquet(val_file)
    files = _split_parquet_files(tmp_path, "valid")
    assert val_file in files


def test_split_parquet_files_missing_returns_empty(tmp_path: Path) -> None:
    assert _split_parquet_files(tmp_path, "test") == []


# ---------------------------------------------------------------------------
# _resolve_dataset_name_as_local_path
# ---------------------------------------------------------------------------


def test_resolve_dataset_name_finds_local_parquet_dir(tmp_path: Path) -> None:
    df = pd.DataFrame({"SELFIES": ["[C][O]"]})
    df.to_parquet(tmp_path / "train.parquet")

    result = _resolve_dataset_name_as_local_path(str(tmp_path))
    assert result == tmp_path


def test_resolve_dataset_name_returns_none_for_hf_repo_name() -> None:
    result = _resolve_dataset_name_as_local_path("mikemayuare/PubChem10M")
    assert result is None


def test_resolve_dataset_name_returns_none_for_nonexistent_path() -> None:
    result = _resolve_dataset_name_as_local_path("/absolutely/does/not/exist/xyz123")
    assert result is None


# ---------------------------------------------------------------------------
# collect_local_parquet_corpus
# ---------------------------------------------------------------------------


def test_collect_local_parquet_corpus_returns_sequences(tmp_path: Path) -> None:
    df = pd.DataFrame({"SELFIES": ["[C][O]", "[C][N]", "[O][C]"]})
    df.to_parquet(tmp_path / "train.parquet")

    corpus = collect_local_parquet_corpus(
        directory=tmp_path,
        representation="SELFIES",
        n=3,
        seed=0,
    )
    assert len(corpus) == 3
    assert all(isinstance(s, str) for s in corpus)


def test_collect_local_parquet_corpus_respects_n(tmp_path: Path) -> None:
    df = pd.DataFrame({"SELFIES": [f"[C][O]{i}" for i in range(20)]})
    df.to_parquet(tmp_path / "train.parquet")

    corpus = collect_local_parquet_corpus(
        directory=tmp_path,
        representation="SELFIES",
        n=5,
        seed=0,
    )
    assert len(corpus) == 5


def test_collect_local_parquet_corpus_raises_on_missing_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"smiles": ["CCO"]})
    df.to_parquet(tmp_path / "train.parquet")

    with pytest.raises(ValueError, match="does not contain column"):
        collect_local_parquet_corpus(
            directory=tmp_path,
            representation="SELFIES",
            n=1,
            seed=0,
        )


def test_collect_local_parquet_corpus_raises_on_missing_split(tmp_path: Path) -> None:
    df = pd.DataFrame({"SELFIES": ["[C][O]"]})
    df.to_parquet(tmp_path / "train.parquet")

    with pytest.raises(ValueError, match="no split"):
        collect_local_parquet_corpus(
            directory=tmp_path,
            representation="SELFIES",
            n=1,
            seed=0,
            split="test",
        )


# ---------------------------------------------------------------------------
# tokenizer_vocab_size
# ---------------------------------------------------------------------------


def test_tokenizer_vocab_size_via_get_vocab() -> None:
    tok = _make_tokenizer()
    size = tokenizer_vocab_size(tok)
    assert size == len(tok.vocabulary)


# ---------------------------------------------------------------------------
# token_id
# ---------------------------------------------------------------------------


def test_token_id_known_token_returns_correct_id() -> None:
    tok = _make_tokenizer()
    assert token_id(tok, "[C]") == tok.vocabulary["[C]"]


def test_token_id_special_token() -> None:
    tok = _make_tokenizer()
    assert token_id(tok, "<pad>") == tok.vocabulary["<pad>"]


# ---------------------------------------------------------------------------
# encode_sequence
# ---------------------------------------------------------------------------


def test_encode_sequence_returns_lists_not_tensors() -> None:
    tok = _make_tokenizer()
    result = encode_sequence(tok, "[C][O]", max_seq_length=32)
    assert isinstance(result["input_ids"], list)
    assert isinstance(result["attention_mask"], list)
    assert all(isinstance(x, int) for x in result["input_ids"])


def test_encode_sequence_includes_bos_and_eos() -> None:
    tok = _make_tokenizer()
    ids = encode_sequence(tok, "[C][O]", max_seq_length=64)["input_ids"]
    assert ids[0] == tok.vocabulary["<s>"]
    assert ids[-1] == tok.vocabulary["</s>"]


def test_encode_sequence_truncates_at_max_length() -> None:
    tok = _make_tokenizer()
    long_selfies = "[C][O]" * 20
    ids = encode_sequence(tok, long_selfies, max_seq_length=8)["input_ids"]
    assert len(ids) <= 8


def test_encode_sequence_no_truncation_when_none() -> None:
    tok = _make_tokenizer()
    selfies = "[C][O][N]"
    ids_full = encode_sequence(tok, selfies, max_seq_length=None)["input_ids"]
    ids_trunc = encode_sequence(tok, selfies, max_seq_length=4)["input_ids"]
    assert len(ids_full) >= len(ids_trunc)


# ---------------------------------------------------------------------------
# assert_metadata_representation
# ---------------------------------------------------------------------------


def test_assert_metadata_representation_matching_passes() -> None:
    assert_metadata_representation({"representation": "SELFIES"}, "SELFIES")


def test_assert_metadata_representation_case_insensitive() -> None:
    assert_metadata_representation({"representation": "selfies"}, "SELFIES")


def test_assert_metadata_representation_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        assert_metadata_representation({"representation": "SMILES"}, "SELFIES")


def test_assert_metadata_representation_missing_raises() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        assert_metadata_representation({}, "SELFIES")


# ---------------------------------------------------------------------------
# filter_zinc20_chembl36_by_source
# ---------------------------------------------------------------------------


def test_filter_zinc20_all_returns_unchanged() -> None:
    from datasets import Dataset

    rows = [
        {"id": "ZINC001", "selfies": "[C]"},
        {"id": "CHEMBL001", "selfies": "[O]"},
    ]
    ds = Dataset.from_list(rows).to_iterable_dataset()
    result = list(filter_zinc20_chembl36_by_source(ds, source="all"))
    assert len(result) == 2


def test_filter_zinc20_keeps_only_zinc_ids() -> None:
    from datasets import Dataset

    rows = [
        {"id": "ZINC001", "selfies": "[C]"},
        {"id": "CHEMBL001", "selfies": "[O]"},
        {"id": "ZINC002", "selfies": "[N]"},
    ]
    ds = Dataset.from_list(rows).to_iterable_dataset()
    result = list(filter_zinc20_chembl36_by_source(ds, source="zinc"))
    ids = [r["id"] for r in result]
    assert all(i.startswith("ZINC") for i in ids)
    assert len(ids) == 2


def test_filter_zinc20_keeps_only_chembl_ids() -> None:
    from datasets import Dataset

    rows = [
        {"id": "ZINC001", "selfies": "[C]"},
        {"id": "CHEMBL001", "selfies": "[O]"},
        {"id": "CHEMBL002", "selfies": "[N]"},
    ]
    ds = Dataset.from_list(rows).to_iterable_dataset()
    result = list(filter_zinc20_chembl36_by_source(ds, source="chembl"))
    ids = [r["id"] for r in result]
    assert all(i.startswith("CHEMBL") for i in ids)
    assert len(ids) == 2
