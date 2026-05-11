"""Dataset loading utilities for frozen molecular representation benchmarks.

All dataset sources are normalized into an `EvalDataset`, which stores fixed
train/valid/test pandas DataFrames plus the column names used for molecular
representations and labels.

Contributing a simple CSV/Parquet dataset:

```python
dataset = load_table_eval_dataset(
    name="my_assay",
    task_type="classification",
    task_names="active",
    train_path="data/my_assay/train.csv",
    valid_path="data/my_assay/valid.csv",
    test_path="data/my_assay/test.csv",
    smiles_column="smiles",
)
```
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal
from collections.abc import Iterator

import pandas as pd


TaskType = Literal["classification", "regression"]


def normalize_task_type(task_type: str) -> TaskType:

    if task_type == "classification":
        return "classification"

    if task_type == "regression":
        return "regression"

    raise ValueError(
        f"Unknown task_type: {task_type!r}. Expected 'classification' or 'regression'."
    )


def normalize_task_names(task_names: str | Sequence[str]) -> list[str]:
    """Normalize one task name or a sequence of task names to a non-empty list."""

    if isinstance(task_names, str):
        out = [task_names]

    else:
        out = [str(task) for task in task_names]

    out = [task.strip() for task in out]

    if not out or any(not task for task in out):
        raise ValueError("task_names must contain at least one non-empty task")

    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate task names found: {out}")

    return out


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a tabular dataset split from CSV, TSV, or Parquet."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Table file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    raise ValueError(
        f"Unsupported table format for {path}. Use .csv, .tsv, .txt, .parquet, or .pq."
    )


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

        normalize_task_type(self.task_type)

        if not self.task_names:
            raise ValueError("EvalDataset must contain at least one task")

        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError(f"Duplicate task names found: {self.task_names}")

        for split_name, frame in self.iter_splits(include_valid=True):
            if len(frame) == 0:
                raise ValueError(f"{split_name} split is empty")

            if self.smiles_column not in frame.columns:
                raise ValueError(
                    f"{split_name} split is missing SMILES column {self.smiles_column!r}"
                )

            # SELFIES is optional because not all datasets are necessarily
            # prepared for SELFIES evaluation. ModernMolBERT-SELFIES loaders
            # can check this explicitly when needed.
            for task in self.task_names:
                if task not in frame.columns:
                    raise ValueError(f"{split_name} split is missing label column {task!r}")

    def check_selfies_available(self) -> None:
        """Validate that every split contains the configured SELFIES column."""

        for split_name, frame in self.iter_splits(include_valid=True):
            if self.selfies_column not in frame.columns:
                raise ValueError(
                    f"{split_name} split is missing SELFIES column {self.selfies_column!r}"
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
    """Load an EvalDataset from explicit CSV split files.

    Deprecated naming: this function now supports CSV/TSV/Parquet via read_table.
    Prefer load_table_eval_dataset in new code.
    """

    return load_table_eval_dataset(
        name=name,
        task_type=task_type,
        task_names=task_names,
        train_path=train_csv,
        valid_path=valid_csv,
        test_path=test_csv,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
    )


def load_single_table_with_split_column(
    *,
    name: str,
    task_type: TaskType | str,
    task_names: str | Sequence[str],
    table_path: str | Path,
    split_column: str = "split",
    smiles_column: str = "smiles",
    selfies_column: str = "selfies",
    train_value: str = "train",
    valid_value: str = "valid",
    test_value: str = "test",
) -> EvalDataset:
    """Load an EvalDataset from one table containing a split column."""

    frame = read_table(table_path)

    if split_column not in frame.columns:
        raise ValueError(f"Missing split column {split_column!r}")

    train = frame.loc[frame[split_column] == train_value].reset_index(drop=True)
    valid_frame = frame.loc[frame[split_column] == valid_value].reset_index(drop=True)
    test = frame.loc[frame[split_column] == test_value].reset_index(drop=True)

    valid = valid_frame if len(valid_frame) > 0 else None

    return make_eval_dataset_from_splits(
        name=name,
        task_type=task_type,
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata={
            "source": "table_with_split_column",
            "table_path": str(table_path),
            "split_column": split_column,
            "train_value": train_value,
            "valid_value": valid_value,
            "test_value": test_value,
        },
    )


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
    """Load an EvalDataset from one CSV containing a split column.

    Deprecated naming: this function now supports CSV/TSV/Parquet via read_table.
    Prefer load_single_table_with_split_column in new code.
    """

    return load_single_table_with_split_column(
        name=name,
        task_type=task_type,
        task_names=task_names,
        table_path=csv_path,
        split_column=split_column,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        train_value=train_value,
        valid_value=valid_value,
        test_value=test_value,
    )


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
            "source": "prepared_moleculenet",
            "eval_split": eval_split,
            "dataset_dir": str(dataset_dir),
            "merge_train_valid": merge_train_valid,
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


def normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize optional metadata to a plain mutable dictionary."""

    if metadata is None:
        return {}

    return {str(key): value for key, value in metadata.items()}


def make_eval_dataset_from_splits(
    *,
    name: str,
    task_type: TaskType | str,
    task_names: str | Sequence[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    valid: pd.DataFrame | None = None,
    smiles_column: str = "smiles",
    selfies_column: str = "selfies",
    metadata: Mapping[str, Any] | None = None,
) -> EvalDataset:
    """Construct and validate an EvalDataset from already-loaded split frames."""

    dataset = EvalDataset(
        name=name,
        task_type=normalize_task_type(str(task_type)),
        task_names=normalize_task_names(task_names),
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata=normalize_metadata(metadata),
    )

    dataset.check()

    return dataset


def load_table_eval_dataset(
    *,
    name: str,
    task_type: TaskType | str,
    task_names: str | Sequence[str],
    train_path: str | Path,
    test_path: str | Path,
    valid_path: str | Path | None = None,
    smiles_column: str = "smiles",
    selfies_column: str = "selfies",
) -> EvalDataset:
    """Load an EvalDataset from explicit train/valid/test table files.

    Supported file formats are CSV, TSV, TXT, Parquet, and PQ.

    """

    train = read_table(train_path)

    valid = read_table(valid_path) if valid_path is not None else None

    test = read_table(test_path)

    return make_eval_dataset_from_splits(
        name=name,
        task_type=task_type,
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column=smiles_column,
        selfies_column=selfies_column,
        metadata={
            "source": "table_splits",
            "train_path": str(train_path),
            "valid_path": str(valid_path) if valid_path is not None else None,
            "test_path": str(test_path),
        },
    )


def load_eval_dataset_from_config(config: Mapping[str, Any]) -> EvalDataset:
    """Load an EvalDataset from a suite/dataset config dictionary.

    Supported loaders:
    - table_splits
    - table_with_split_column
    - prepared_moleculenet
    """

    loader = str(config.get("loader", ""))

    if loader == "table_splits":
        return load_table_eval_dataset(
            name=str(config["name"]),
            task_type=str(config["task_type"]),
            task_names=config["task_names"],
            train_path=config["train_path"],
            valid_path=config.get("valid_path"),
            test_path=config["test_path"],
            smiles_column=str(config.get("smiles_column", "smiles")),
            selfies_column=str(config.get("selfies_column", "selfies")),
        )

    if loader == "table_with_split_column":
        return load_single_table_with_split_column(
            name=str(config["name"]),
            task_type=str(config["task_type"]),
            task_names=config["task_names"],
            table_path=config["table_path"],
            split_column=str(config.get("split_column", "split")),
            smiles_column=str(config.get("smiles_column", "smiles")),
            selfies_column=str(config.get("selfies_column", "selfies")),
            train_value=str(config.get("train_value", "train")),
            valid_value=str(config.get("valid_value", "valid")),
            test_value=str(config.get("test_value", "test")),
        )

    if loader == "prepared_moleculenet":
        return load_prepared_moleculenet_dataset(
            dataset_dir=config["dataset_dir"],
            eval_split=str(config.get("eval_split", "test")),
            smiles_column=str(config.get("smiles_column", "smiles_canonical")),
            selfies_column=str(config.get("selfies_column", "selfies")),
            merge_train_valid=bool(config.get("merge_train_valid", False)),
        )

    raise ValueError(
        f"Unknown dataset loader {loader!r}. "
        "Expected one of: 'table_splits', 'table_with_split_column', "
        "'prepared_moleculenet'."
    )
