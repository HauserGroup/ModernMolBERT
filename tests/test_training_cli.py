import argparse
import sys

import pytest
import torch
from datasets import Dataset

from modernmolbert.train_ape_tokenizer import parse_args as parse_ape_args
from modernmolbert.train_selfies_ape_modernbert import (
    make_eval_dataset,
    make_train_iterable_dataset,
    parse_args as parse_train_args,
    resolve_dataset_args,
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


def test_representation_defaults_to_selfies():
    with _Argv("--output_dir", "tmp/run"):
        args = resolve_dataset_args(parse_train_args())

    assert args.representation == "SELFIES"
    # Column is dataset-inferred; molecule_column mirrors the resolved column.
    assert args.molecule_column == args.selfies_column
    assert str(args.tokenizer_vocab_path).endswith("selfies_ape_tokenizer.json")


def test_representation_smiles_resolves_column_and_tokenizer():
    with _Argv(
        "--output_dir",
        "tmp/run",
        "--representation",
        "SMILES",
        "--molecule_column",
        "smiles_canonical_clean",
        "--dataset_name",
        "data/pretrain/chembl36_selfies",
    ):
        args = resolve_dataset_args(parse_train_args())

    assert args.representation == "SMILES"
    assert args.selfies_column == "smiles_canonical_clean"
    assert args.molecule_column == "smiles_canonical_clean"
    assert str(args.tokenizer_vocab_path).endswith("smiles_ape_tokenizer.json")


def test_selfies_column_alias_still_resolves():
    with _Argv("--output_dir", "tmp/run", "--selfies_column", "my_col"):
        args = resolve_dataset_args(parse_train_args())

    assert args.selfies_column == "my_col"


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
