"""Tests for modernmolbert.utils — private helpers and pure functions not covered elsewhere."""

import json
from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    _local_dataset_matches_request,
    _resolve_dataset_name_as_local_path,
    _split_parquet_files,
    assert_metadata_representation,
    collect_local_parquet_corpus,
    encode_sequence,
    filter_zinc20_chembl36_by_source,
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
# _local_dataset_matches_request
# ---------------------------------------------------------------------------


def test_local_dataset_matches_request_by_dir_name(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pubchem10m"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"dataset_name": "pubchem10m"}), encoding="utf-8"
    )
    assert _local_dataset_matches_request(dataset_dir, "PubChem10M_SMILES_SELFIES") is True


def test_local_dataset_matches_request_mismatch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "zinc20"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"dataset_name": "zinc20"}), encoding="utf-8"
    )
    assert _local_dataset_matches_request(dataset_dir, "pubchem10m") is False


# ---------------------------------------------------------------------------
# _split_parquet_files
# ---------------------------------------------------------------------------


def test_split_parquet_files_sharded(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    shard1 = tmp_path / "train-00001.parquet"
    shard2 = tmp_path / "train-00002.parquet"
    df.to_parquet(shard1)
    df.to_parquet(shard2)
    files = _split_parquet_files(tmp_path, "train")
    assert set(files) == {shard1, shard2}


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


# ---------------------------------------------------------------------------
# encode_sequence
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# assert_metadata_representation
# ---------------------------------------------------------------------------


def test_assert_metadata_representation_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        assert_metadata_representation({"representation": "SMILES"}, "SELFIES")


# ---------------------------------------------------------------------------
# filter_zinc20_chembl36_by_source
# ---------------------------------------------------------------------------


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
