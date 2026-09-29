import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from modernmolbert.train_selfies_ape_modernbert import (
    make_eval_dataset,
    make_train_iterable_dataset,
)

_FREEZE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_training_order.py"
_FREEZE_SPEC = importlib.util.spec_from_file_location("freeze_training_order", _FREEZE_SCRIPT)
assert _FREEZE_SPEC is not None and _FREEZE_SPEC.loader is not None
_FREEZE_MODULE: Any = importlib.util.module_from_spec(_FREEZE_SPEC)
_FREEZE_SPEC.loader.exec_module(_FREEZE_MODULE)
freeze_order = _FREEZE_MODULE.freeze_order


class _Tokenizer:
    def __call__(self, sequence: str, **_kwargs):
        index = int(sequence.rsplit("_", 1)[1])
        return {"input_ids": [0, index + 5, 2]}


def _args(root: Path, column: str, order: Path, validation_ids: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name=str(root),
        data_dir=None,
        data_files=str(root / "train.parquet"),
        train_split="train",
        validation_split="valid",
        use_validation_split=True,
        molecule_column=column,
        output_dir=str(root / f"out_{column}"),
        seed=42,
        global_train_shuffle=True,
        train_order_path=order,
        validation_row_ids_path=validation_ids,
        shuffle_buffer_size=8,
        max_seq_length=16,
        eval_size=4,
        max_eval_batches=0,
        per_device_eval_batch_size=2,
    )


def test_frozen_row_order_is_identical_across_representations(tmp_path: Path) -> None:
    Dataset.from_dict(
        {
            "smiles_canonical_clean": [f"smiles_{i}" for i in range(16)],
            "selfies": [f"selfies_{i}" for i in range(16)],
        }
    ).to_parquet(str(tmp_path / "train.parquet"))
    Dataset.from_dict(
        {
            "smiles_canonical_clean": [f"smiles_{i}" for i in range(8)],
            "selfies": [f"selfies_{i}" for i in range(8)],
        }
    ).to_parquet(str(tmp_path / "valid.parquet"))
    order = tmp_path / "train_order.npy"
    validation_ids = tmp_path / "valid_ids.npy"
    freeze_order(tmp_path / "train.parquet", order, seed=42)
    freeze_order(tmp_path / "valid.parquet", validation_ids, seed=42, take=4)

    for column in ("smiles_canonical_clean", "selfies"):
        args = _args(tmp_path, column, order, validation_ids)
        train = list(make_train_iterable_dataset(args, _Tokenizer()))  # type: ignore[arg-type]
        observed = [int(row[column].rsplit("_", 1)[1]) for row in train]
        assert observed == np.load(order).tolist()

        validation = make_eval_dataset(args, _Tokenizer())  # type: ignore[arg-type]
        assert [ids[1] - 5 for ids in validation["input_ids"]] == np.load(validation_ids).tolist()

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        freeze_order(tmp_path / "train.parquet", order, seed=42)
