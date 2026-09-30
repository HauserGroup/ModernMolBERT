from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import logging as log
import pandas as pd
import numpy as np
import json
import base64
import sys


from typing import Literal


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        # A tensor can only exist if torch is already imported, so avoid importing it here.
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(o, torch.Tensor):
            data_b64 = base64.b64encode(o.cpu().numpy().tobytes()).decode("utf-8")
            return dict(__torch_tensor__=data_b64, dtype=str(o.dtype).split(".")[-1], shape=o.shape)
        if isinstance(o, np.ndarray):
            data_b64 = base64.b64encode(np.ascontiguousarray(o).data).decode("utf-8")
            return dict(__ndarray__=data_b64, dtype=str(o.dtype), shape=o.shape)
        elif isinstance(o, pd.DataFrame):
            return dict(__dataframe__=o.to_dict(orient="split"))
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray or pandas DataFrame
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray or DataFrame
    :return: (ndarray or DataFrame) if input was an encoded ndarray or DataFrame
    """
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = base64.b64decode(dct["__ndarray__"])
        return np.frombuffer(data, dct["dtype"]).reshape(dct["shape"])
    elif isinstance(dct, dict) and "__dataframe__" in dct:
        return pd.DataFrame(**dct["__dataframe__"])
    elif isinstance(dct, dict) and "__torch_tensor__" in dct:
        import torch

        data = base64.b64decode(dct["__torch_tensor__"])
        as_tensor = getattr(torch, "as_" + "tensor")
        return as_tensor(np.frombuffer(data, dct["dtype"]).reshape(dct["shape"]))
    return dct


@dataclass
class EmbeddingConfig:
    raw_directory: str
    embedded_directory: str
    predictions_directory: str
    prepared_directory: str
    max_invalid_embeddings: int
    max_samples: int | None = None


@dataclass
class Dataset:
    name: str
    task: Literal["classification", "regression"]
    data: Any
    splits: Any

    @property
    def labels(self) -> pd.DataFrame:
        id_names = {"drug_id", "mol_id", "id", "split"}
        id_cols = [x for x in self.data.columns if x.lower() in id_names]

        return self.data.drop(columns=(["smiles", "graph"] + id_cols), errors="ignore")

    def serialize_legacy(self, path):
        import json

        obj = {"name": self.name, "task": self.task, "data": self.data, "splits": self.splits}
        with open(path, "w") as f:
            json.dump(obj, f, cls=NumpyEncoder)

    @classmethod
    def deserialize_legacy(cls, path):
        import json

        with open(path) as f:
            return cls(**json.load(f, object_hook=json_numpy_obj_hook))


@dataclass
class EmbeddedDataset:
    name: str
    task: Literal["classification", "regression"]
    embedder: str
    splits: Any
    X: np.ndarray
    y: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def remove_failed_embeddings(self) -> int:
        """Performs validation of missing values, drop failed molecules and return number of invalid samples.

        Returns:
            int: number of invalid samples
        """
        print(f"Shape of X: {self.X.shape}")
        if len(self.X.shape) == 1:
            log.warning(f"Dataset '{self.name}' has only one dimension in X, reshaping to 2D.")
            self.X = self.X.reshape(self.y.shape[0], -1)
        valid_indices = np.where(~np.isnan(self.X).any(axis=1))[0]
        invalid_count = self.X.shape[0] - valid_indices.shape[0]
        split_to_idx = {k: i for i, k in enumerate(self.splits.keys())}
        if invalid_count == 0:
            return 0

        log.warning(f"Found {invalid_count} invalid samples in the dataset '{self.name}'.")
        splits_raw = -1 * np.ones(self.X.shape[0], dtype=int)
        for split_name, indices in self.splits.items():
            splits_raw[indices] = split_to_idx[split_name]
        self.X = self.X[valid_indices]
        self.y = self.y.iloc[valid_indices]
        splits_raw = splits_raw[valid_indices]

        self.splits = {k: np.where(splits_raw == i)[0].tolist() for k, i in split_to_idx.items()}
        log.info(f"Removed {invalid_count} invalid samples from the dataset '{self.name}'")
        return invalid_count

    @cached_property
    def y_np(self) -> np.ndarray:
        if "split" in self.y.columns:
            self.y = self.y.drop(columns=["split"])

        arr = self.y.to_numpy()
        if arr.shape[1] == 1:
            return arr.flatten()
        return arr

    @classmethod
    def deserialize_legacy(cls, path):
        import json

        with open(path) as f:
            return cls(**json.load(f, object_hook=json_numpy_obj_hook))


@dataclass
class HeadResult:
    embedder: str
    dataset_name: str
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    cv_score: float
    model: str
    hyperparams: dict[str, Any]
    test_source_row_indices: np.ndarray | None = None
    prepared_data_sha256: str | None = None


@dataclass
class EvaluationResult:
    embedder: str
    metric_name: str
    metric_value: float
    cv_metric_value: float
    model: str
    hyperparams: dict[str, Any]
