import argparse
import sys

import pytest
import torch

from modernmolbert.train_ape_tokenizer import parse_args as parse_ape_args
from modernmolbert.train_selfies_ape_modernbert import (
    parse_args as parse_train_args,
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
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="Use --no-bf16"):
        validate_args(args, backend="cuda")
