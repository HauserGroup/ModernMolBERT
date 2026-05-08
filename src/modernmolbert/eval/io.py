import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hash_strings(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()


def hash_dataframe_smiles(frame: pd.DataFrame, smiles_column: str) -> str:
    values = [str(x) for x in frame[smiles_column].tolist()]
    return hash_strings(values)


def save_numpy(path: str | Path, arr: np.ndarray) -> None:
    np.save(path, arr)


def load_numpy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)
