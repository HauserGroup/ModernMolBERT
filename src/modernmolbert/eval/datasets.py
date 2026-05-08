from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class EvalDataset:
    name: str
    task_names: list[str]
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    smiles_column: str
    label_columns: list[str]
    task_type: TaskType

    def check(self) -> None:
        for split_name, df in [
            ("train", self.train),
            ("valid", self.valid),
            ("test", self.test),
        ]:
            if self.smiles_column not in df.columns:
                raise ValueError(
                    f"{split_name} missing smiles column {self.smiles_column!r}"
                )
            for col in self.label_columns:
                if col not in df.columns:
                    raise ValueError(f"{split_name} missing label column {col!r}")


MOLECULENET_TASKS: dict[str, dict] = {
    "bbbp": {
        "loader": "load_bbbp",
        "task_type": "classification",
    },
    "hiv": {
        "loader": "load_hiv",
        "task_type": "classification",
    },
    "tox21": {
        "loader": "load_tox21",
        "task_type": "classification",
    },
    "esol": {
        "loader": "load_delaney",
        "task_type": "regression",
    },
    "freesolv": {
        "loader": "load_freesolv",
        "task_type": "regression",
    },
    "lipo": {
        "loader": "load_lipo",
        "task_type": "regression",
    },
}


def _dc_dataset_to_frame(ds, task_names: list[str]) -> pd.DataFrame:
    """Convert a DeepChem Dataset split to a pandas DataFrame.

    DeepChem datasets usually expose:
      - ids: often SMILES strings
      - y: labels, shape [n, n_tasks]
      - w: weights/masks, shape [n, n_tasks]
    """
    smiles = list(ds.ids)
    y = np.asarray(ds.y)
    w = np.asarray(ds.w) if getattr(ds, "w", None) is not None else np.ones_like(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)

    frame = pd.DataFrame({"smiles": smiles})

    for i, task in enumerate(task_names):
        frame[task] = y[:, i]
        frame[f"{task}__weight"] = w[:, i]

    return frame


def load_moleculenet(
    name: str,
    splitter: str = "scaffold",
    reload: bool = True,
    data_dir: str | None = None,
    save_dir: str | None = None,
) -> EvalDataset:
    """Load a MoleculeNet task through DeepChem and convert to EvalDataset."""
    if name not in MOLECULENET_TASKS:
        raise ValueError(
            f"Unknown MoleculeNet task {name!r}. Known: {sorted(MOLECULENET_TASKS)}"
        )

    try:
        import deepchem as dc
    except ImportError as e:
        raise ImportError(
            "DeepChem is required for MoleculeNet loading. "
            "Install with `uv add deepchem` or add it to an eval dependency group."
        ) from e

    spec = MOLECULENET_TASKS[name]
    loader = getattr(dc.molnet, spec["loader"])  # type: ignore[attr-defined]

    # Use Raw featurizer because we only need SMILES strings / ids.
    tasks, datasets, _transformers = loader(
        featurizer="Raw",
        splitter=splitter,
        reload=reload,
        data_dir=data_dir,
        save_dir=save_dir,
    )

    train_ds, valid_ds, test_ds = datasets
    task_names = list(tasks)

    eval_ds = EvalDataset(
        name=name,
        task_names=task_names,
        train=_dc_dataset_to_frame(train_ds, task_names),
        valid=_dc_dataset_to_frame(valid_ds, task_names),
        test=_dc_dataset_to_frame(test_ds, task_names),
        smiles_column="smiles",
        label_columns=task_names,
        task_type=spec["task_type"],
    )
    eval_ds.check()
    return eval_ds
