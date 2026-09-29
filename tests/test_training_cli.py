import argparse
import sys

import pytest
import torch
from datasets import Dataset
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

from modernmolbert.train_selfies_ape_modernbert import (
    build_modernbert_config,
    make_eval_dataset,
    make_train_iterable_dataset,
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
