from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.common.types import EmbeddedDataset


def _split_to_list(split: object) -> list[int]:
    """Convert a split index container to a plain list of integer indices."""
    if split is None:
        return []

    if isinstance(split, list):
        return [int(i) for i in split]

    tolist = getattr(split, "tolist", None)
    if callable(tolist):
        values = tolist()
        if isinstance(values, Iterable):
            return [int(i) for i in values]
        scalar_value = cast(Any, values)
        return [int(scalar_value)]

    if isinstance(split, Iterable):
        return [int(i) for i in split]

    scalar_split = cast(Any, split)
    return [int(scalar_split)]


def get_train_data(dataset: EmbeddedDataset) -> tuple[np.ndarray, np.ndarray]:
    train_split = _split_to_list(dataset.splits.get("train", []))
    valid_split = _split_to_list(dataset.splits.get("valid", []))

    train_indices = train_split + valid_split

    if not train_indices:
        raise ValueError("No training indices found. Expected at least a non-empty 'train' split.")

    X = dataset.X[train_indices].astype(np.float32, copy=False)
    y = dataset.y_np[train_indices]

    return X, y


def get_test_data(dataset: EmbeddedDataset) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset.splits["test"], list):
        test_split = dataset.splits["test"]
    else:
        test_split = dataset.splits["test"].tolist()
    X = dataset.X[test_split].astype(np.float32, copy=False)
    return X, dataset.y_np[test_split]
