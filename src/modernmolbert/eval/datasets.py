"""Contributing:


dataset = load_csv_eval_dataset(
    name="my_assay",
    task_type="classification",
    task_names="active",
    train_csv="data/my_assay/train.csv",
    valid_csv="data/my_assay/valid.csv",
    test_csv="data/my_assay/test.csv",
    smiles_column="smiles",
)

"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Literal
from collections.abc import Sequence

import pandas as pd


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class EvalDataset:
    """A benchmark dataset with fixed train/valid/test splits.

    This class is intentionally simple. It stores already-prepared tabular
    data and tells the evaluation runner which columns contain molecular
    representations and labels.

    SMILES featurizers should use ``smiles_column``.
    SELFIES featurizers should use ``selfies_column``.
    """

    name: str
    task_type: TaskType
    task_names: list[str]
    train: pd.DataFrame
    valid: pd.DataFrame | None
    test: pd.DataFrame
    smiles_column: str = "smiles"
    selfies_column: str = "selfies"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label_columns(self) -> list[str]:
        return self.task_names

    def check(self) -> None:
        """Validate dataset structure before evaluation."""

        if not self.name:
            raise ValueError("Dataset name must be non-empty")

        if self.task_type not in {"classification", "regression"}:
            raise ValueError(f"Unknown task_type: {self.task_type!r}")

        if not self.task_names:
            raise ValueError("EvalDataset must contain at least one task")

        for split_name, frame in self.iter_splits(include_valid=True):
            if len(frame) == 0:
                raise ValueError(f"{split_name} split is empty")

            if self.smiles_column not in frame.columns:
                raise ValueError(
                    f"{split_name} split is missing SMILES column "
                    f"{self.smiles_column!r}"
                )

            # SELFIES is optional because not all datasets are necessarily
            # prepared for SELFIES evaluation. ModernMolBERT-SELFIES loaders
            # can check this explicitly when needed.
            for task in self.task_names:
                if task not in frame.columns:
                    raise ValueError(
                        f"{split_name} split is missing label column {task!r}"
                    )

    def check_selfies_available(self) -> None:
        """Validate that every split contains the configured SELFIES column."""

        for split_name, frame in self.iter_splits(include_valid=True):
            if self.selfies_column not in frame.columns:
                raise ValueError(
                    f"{split_name} split is missing SELFIES column "
                    f"{self.selfies_column!r}"
                )

    def iter_splits(
        self,
        *,
        include_valid: bool = True,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        yield "train", self.train

        if include_valid and self.valid is not None:
            yield "valid", self.valid

        yield "test", self.test


def load_csv_eval_dataset(
    *,
    name: str,
    task_type: TaskType,
    task_names: str | Sequence[str],
    train_csv: str | Path,
    test_csv: str | Path,
    valid_csv: str | Path | None = None,
    smiles_column: str = "smiles",
    selfies_column: str = "selfies",
) -> EvalDataset:
    """Load an EvalDataset from explicit CSV split files."""

    train = read_table(train_csv)
    valid = read_table(valid_csv) if valid_csv is not None else None
    test = read_table(test_csv)

    dataset = EvalDataset(
        name=name,
        task_type=task_type,
        task_names=normalize_task_names(task_names),
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata={},
    )
    dataset.check()
    return dataset


def load_single_csv_with_split_column(
    *,
    name: str,
    task_type: TaskType,
    task_names: str | Sequence[str],
    csv_path: str | Path,
    split_column: str = "split",
    smiles_column: str = "smiles",
    selfies_column: str = "selfies",
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
        task_names=normalize_task_names(task_names),
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata={},
    )
    dataset.check()
    return dataset


def load_prepared_moleculenet_dataset(
    *,
    dataset_dir: str | Path,
    eval_split: str = "test",
    smiles_column: str = "smiles_canonical",
    selfies_column: str = "selfies",
    merge_train_valid: bool = False,
) -> EvalDataset:
    """Load a sanitized MoleculeNet dataset prepared by prepare_moleculenet.

    Expected layout:

        dataset_dir/
          metadata.json
          train.parquet
          valid.parquet
          test.parquet

    The metadata file should contain at least:

        {
          "name": "...",
          "task_type": "classification" | "regression",
          "tasks": [...]
        }
    """

    dataset_dir = Path(dataset_dir)

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be either 'valid' or 'test'")

    train_path = dataset_dir / "train.parquet"
    valid_path = dataset_dir / "valid.parquet"
    eval_path = dataset_dir / f"{eval_split}.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval split: {eval_path}")

    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path) if valid_path.exists() else None
    test = pd.read_parquet(eval_path)

    if merge_train_valid and valid is not None:
        train = pd.concat([train, valid], ignore_index=True)
        valid = None

    task_type = metadata.get("task_type")
    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"metadata.json has invalid task_type: {task_type!r}. "
            "Expected 'classification' or 'regression'."
        )

    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("metadata.json must contain a non-empty list field 'tasks'")

    dataset_metadata = dict(metadata)

    dataset_metadata.update(
        {
            "eval_split": eval_split,
            "dataset_dir": str(dataset_dir),
        }
    )

    dataset = EvalDataset(
        name=str(metadata.get("name", dataset_dir.name)),
        task_type=task_type,
        task_names=[str(task) for task in tasks],
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata=dataset_metadata,
    )
    dataset.check()
    return dataset


def normalize_task_names(task_names: str | Sequence[str]) -> list[str]:

    if isinstance(task_names, str):
        return [task_names]

    out = [str(task) for task in task_names]

    if not out:
        raise ValueError("task_names must contain at least one task")

    return out


def read_table(path: str | Path) -> pd.DataFrame:

    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    raise ValueError(
        f"Unsupported table format for {path}. Use .csv, .tsv, .parquet, or .pq."
    )


def make_eval_dataset_from_splits(
    *,
    name: str,
    task_type: TaskType,
    task_names: str | Sequence[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    valid: pd.DataFrame | None = None,
    smiles_column: str = "smiles",
    selfies_column: str = "selfies",
    metadata: dict[str, Any] | None = None,
) -> EvalDataset:
    dataset = EvalDataset(
        name=name,
        task_type=task_type,
        task_names=normalize_task_names(task_names),
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata={} if metadata is None else dict(metadata),
    )
    dataset.check()
    return dataset
