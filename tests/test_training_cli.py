import argparse
import sys

import pytest
import torch
from datasets import Dataset
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

from modernmolbert.train_ape_tokenizer import parse_args as parse_ape_args
from modernmolbert.train_selfies_ape_modernbert import (
    build_modernbert_config,
    make_eval_dataset,
    make_train_iterable_dataset,
    parse_args as parse_train_args,
    sequence_bucket,
    validate_args,
)


class _Argv:
    def __init__(self, *args: str):
        self._args = ["prog", *args]
        self._old = []

    def __enter__(self):
        self._old = sys.argv
        sys.argv = self._args

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self._old


def test_parse_train_args_accepts_no_bf16():
    with _Argv("--output_dir", "tmp/run", "--no-bf16"):
        args = parse_train_args()

    assert args.bf16 is False


def test_molecular_model_config_uses_vocabulary_boundaries_for_cls_and_sep(monkeypatch):
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.AutoConfig.from_pretrained",
        lambda *_args: ModernBertConfig(),
    )
    config = build_modernbert_config(
        argparse.Namespace(model_size="small", max_seq_length=128),
        vocab_size=588,
        special_ids={
            "pad_token": 1,
            "bos_token": 0,
            "eos_token": 2,
            "unk_token": 3,
            "mask_token": 4,
        },
    )
    assert config.cls_token_id == config.bos_token_id == 0
    assert config.sep_token_id == config.eos_token_id == 2
    assert max(config.cls_token_id, config.sep_token_id, config.pad_token_id) < 588


def test_parse_ape_args_accepts_data_files():
    with _Argv("--data_files", "data/*.parquet"):
        args = parse_ape_args()

    assert args.data_files == "data/*.parquet"


def test_validate_args_rejects_unsupported_cuda_bf16(monkeypatch):
    args = argparse.Namespace(
        max_seq_length=128,
        mlm_probability=0.15,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        val_split_mod=100,
        val_split_bucket=0,
        device_backend="cuda",
        eval_size=1000,
        max_eval_batches=0,
        load_best_model_at_end=True,
        save_steps=500,
        eval_steps=500,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="Use --no-bf16"):
        validate_args(args, backend="cuda")


def test_pretokenized_rows_use_stable_hash_split(monkeypatch):
    rows = [
        {"input_ids": [0, 5, 2]},
        {"input_ids": [0, 6, 2]},
    ]
    validation_bucket = sequence_bucket("0,5,2", 100)
    args = argparse.Namespace(
        dataset_name="pretok",
        train_split="train",
        validation_split=None,
        use_validation_split=False,
        selfies_column="SELFIES",
        data_dir=None,
        data_files=None,
        seed=13,
        shuffle_buffer_size=100,
        max_seq_length=8,
        eval_size=4,
        max_eval_batches=0,
        per_device_eval_batch_size=2,
        val_split_mod=100,
        val_split_bucket=validation_bucket,
    )

    def _fake_stream(*args, **kwargs):
        return Dataset.from_list(rows).to_iterable_dataset()

    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.get_streaming_dataset",
        _fake_stream,
    )

    train_rows = list(make_train_iterable_dataset(args, tokenizer=None))  # type: ignore[arg-type]
    eval_dataset = make_eval_dataset(args, tokenizer=None)  # type: ignore[arg-type]

    assert [row["input_ids"] for row in train_rows] == [[0, 6, 2]]
    assert eval_dataset["input_ids"] == [[0, 5, 2]]


def test_global_train_shuffle_covers_all_rows_without_source_order_bias(tmp_path):
    source = tmp_path / "train.parquet"
    Dataset.from_dict({"input_ids": [[0, i + 5, 2] for i in range(100)]}).to_parquet(source)
    args = argparse.Namespace(
        output_dir=str(tmp_path / "run"),
        dataset_name=str(tmp_path),
        train_split="train",
        data_dir=None,
        data_files=None,
        global_train_shuffle=True,
        seed=42,
        shuffle_buffer_size=10,
        selfies_column="selfies",
        use_validation_split=True,
        max_seq_length=8,
    )

    rows = list(make_train_iterable_dataset(args, tokenizer=None))  # type: ignore[arg-type]
    observed = [row["input_ids"][1] - 5 for row in rows]
    assert sorted(observed) == list(range(100))
    assert observed != list(range(100))
    assert any(index >= 20 for index in observed[:10])
    rows_again = list(make_train_iterable_dataset(args, tokenizer=None))  # type: ignore[arg-type]
    assert [row["input_ids"][1] for row in rows_again] == [row["input_ids"][1] for row in rows]
    stream = make_train_iterable_dataset(args, tokenizer=None)  # type: ignore[arg-type]
    stream.set_epoch(1)
    assert [row["input_ids"][1] for row in stream][:10] != [row["input_ids"][1] for row in rows][
        :10
    ]
