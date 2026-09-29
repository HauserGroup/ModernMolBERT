from argparse import Namespace

import pandas as pd
import pytest

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.train_ape_tokenizer import collect_corpus_primitives, validate_args
from modernmolbert.train_selfies_ape_modernbert import (
    assert_corpus_only_vocab,
    corpus_only_training_parquet,
)
from modernmolbert.utils import file_sha256


def test_full_corpus_scan_adds_rare_training_symbols_and_component_dots(tmp_path):
    source = tmp_path / "train.parquet"
    pd.DataFrame({"selfies": ["[C].[O]", "[135I][C]"]}).to_parquet(source)
    primitives, n_rows = collect_corpus_primitives(source, "selfies")
    assert n_rows == 2
    assert primitives == [".", "[135I]", "[C]", "[O]"]
    tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
    tokenizer.add_tokens_to_vocabulary(primitives)
    assert tokenizer.tokenize("[C].[O]") == ["[C]", ".", "[O]"]
    assert tokenizer.unk_token_id not in tokenizer.encode("[135I][C]", add_special_tokens=False)


def test_full_corpus_scan_rejects_malformed_strings(tmp_path):
    source = tmp_path / "train.parquet"
    pd.DataFrame({"selfies": ["[C]missing[O]"]}).to_parquet(source)
    with pytest.raises(ValueError, match="Malformed SELFIES"):
        collect_corpus_primitives(source, "selfies")


def test_corpus_only_scan_rejects_benchmark_symbol_injection():
    args = Namespace(
        tokenizer_train_size=10,
        max_vocab_size=100,
        min_freq_for_merge=2,
        shuffle_buffer_size=10,
        representation="SELFIES",
        extra_vocab_selfies_path=None,
        extra_vocab_symbols_path="benchmark_symbols.txt",
        corpus_primitive_parquet="train.parquet",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_args(args)


def test_encoder_requires_scanned_uninjected_tokenizer_metadata(tmp_path):
    source = tmp_path / "train.parquet"
    source.write_bytes(b"frozen training input")
    clean = {
        "corpus_primitive_scan": {"sha256": file_sha256(source), "n_rows": 100},
        "extra_vocab_symbols_requested": 0,
        "extra_vocab_symbols_added": 0,
    }
    assert_corpus_only_vocab(clean, source)
    with pytest.raises(ValueError, match="corpus_primitive_scan"):
        assert_corpus_only_vocab({}, source)
    with pytest.raises(ValueError, match="requested extra"):
        assert_corpus_only_vocab({**clean, "extra_vocab_symbols_requested": 1}, source)
    with pytest.raises(ValueError, match="different training Parquet"):
        assert_corpus_only_vocab(
            {**clean, "corpus_primitive_scan": {"sha256": "a" * 64, "n_rows": 100}}, source
        )


def test_encoder_rejects_auto_discovered_arrow_dataset(tmp_path, monkeypatch):
    import modernmolbert.train_selfies_ape_modernbert as train

    (tmp_path / "train.parquet").touch()
    args = Namespace(
        data_dir=None, data_files=None, dataset_name=str(tmp_path), train_split="train"
    )
    monkeypatch.setattr(train, "find_local_dataset", lambda **_: tmp_path / "arrow")
    with pytest.raises(ValueError, match="Arrow dataset"):
        corpus_only_training_parquet(args)
    args.data_files = str(tmp_path / "train.parquet")
    assert corpus_only_training_parquet(args) == tmp_path / "train.parquet"
