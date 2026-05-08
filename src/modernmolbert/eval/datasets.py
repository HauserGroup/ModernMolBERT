from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class EvalDataset:
    """A benchmark task with fixed train/valid/test splits."""

    name: str
    task_type: TaskType
    task_names: list[str]
    train: pd.DataFrame
    valid: pd.DataFrame | None
    test: pd.DataFrame
    smiles_column: str = "smiles"

    @property
    def label_columns(self) -> list[str]:
        return self.task_names

    def check(self) -> None:
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(f"Unknown task_type: {self.task_type!r}")

        if not self.task_names:
            raise ValueError("EvalDataset must contain at least one task")

        for split_name, frame in self.iter_splits(include_valid=True):
            if self.smiles_column not in frame.columns:
                raise ValueError(
                    f"{split_name} split is missing smiles column "
                    f"{self.smiles_column!r}"
                )

            for task in self.task_names:
                if task not in frame.columns:
                    raise ValueError(
                        f"{split_name} split is missing label column {task!r}"
                    )

    def iter_splits(
        self,
        *,
        include_valid: bool = True,
    ):
        yield "train", self.train
        if include_valid and self.valid is not None:
            yield "valid", self.valid
        yield "test", self.test


def load_csv_eval_dataset(
    *,
    name: str,
    task_type: TaskType,
    task_names: list[str],
    train_csv: str | Path,
    test_csv: str | Path,
    valid_csv: str | Path | None = None,
    smiles_column: str = "smiles",
) -> EvalDataset:
    """Load an EvalDataset from explicit CSV split files."""

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    valid = pd.read_csv(valid_csv) if valid_csv is not None else None

    dataset = EvalDataset(
        name=name,
        task_type=task_type,
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
    )
    dataset.check()
    return dataset


def load_single_csv_with_split_column(
    *,
    name: str,
    task_type: TaskType,
    task_names: list[str],
    csv_path: str | Path,
    split_column: str = "split",
    smiles_column: str = "smiles",
    train_value: str = "train",
    valid_value: str = "valid",
    test_value: str = "test",
) -> EvalDataset:
    """Load an EvalDataset from one CSV containing a split column."""

    frame = pd.read_csv(csv_path)
    if split_column not in frame.columns:
        raise ValueError(f"Missing split column {split_column!r}")

    train = frame.loc[frame[split_column] == train_value].reset_index(drop=True)
    valid_frame = frame.loc[frame[split_column] == valid_value].reset_index(drop=True)
    test = frame.loc[frame[split_column] == test_value].reset_index(drop=True)

    valid = valid_frame if len(valid_frame) > 0 else None

    dataset = EvalDataset(
        name=name,
        task_type=task_type,
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
    )
    dataset.check()
    return dataset


_MOLECULENET_LOADERS: dict[str, tuple[str, TaskType]] = {
    "bbbp": ("load_bbbp", "classification"),
    "hiv": ("load_hiv", "classification"),
    "tox21": ("load_tox21", "classification"),
    "esol": ("load_delaney", "regression"),
    "freesolv": ("load_sampl", "regression"),
    "lipo": ("load_lipo", "regression"),
}


def load_moleculenet(
    dataset_name: str,
    splitter: str = "scaffold",
) -> EvalDataset:
    """Load a standard MoleculeNet benchmark via deepchem.

    Parameters
    ----------
    dataset_name:
        One of "bbbp", "hiv", "tox21", "esol", "freesolv", "lipo".
    splitter:
        Deepchem splitter name, e.g. "scaffold" or "random".
    """
    try:
        import deepchem as dc
    except ImportError as e:
        raise ImportError(
            "deepchem is required for MoleculeNet loading. "
            "Install it with: uv add --group eval deepchem"
        ) from e

    if dataset_name not in _MOLECULENET_LOADERS:
        raise ValueError(
            f"Unknown MoleculeNet dataset {dataset_name!r}. "
            f"Known datasets: {sorted(_MOLECULENET_LOADERS)}"
        )

    loader_name, task_type = _MOLECULENET_LOADERS[dataset_name]
    loader = getattr(dc.molnet, loader_name)  # type: ignore[attr-defined]
    tasks, (train_dc, valid_dc, test_dc), _ = loader(splitter=splitter)

    def _to_dataframe(dc_dataset) -> pd.DataFrame:
        df = pd.DataFrame({"smiles": dc_dataset.ids})
        for i, task in enumerate(tasks):
            df[task] = dc_dataset.y[:, i]
            # Include weight column so _valid_label_mask can filter missing labels.
            if dc_dataset.w is not None:
                df[f"{task}__weight"] = dc_dataset.w[:, i]
        return df

    return EvalDataset(
        name=dataset_name,
        task_type=task_type,
        task_names=list(tasks),
        train=_to_dataframe(train_dc),
        valid=_to_dataframe(valid_dc) if valid_dc is not None else None,
        test=_to_dataframe(test_dc),
        smiles_column="smiles",
    )
