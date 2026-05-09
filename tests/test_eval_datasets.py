import json
from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.eval.datasets import (
    load_eval_dataset_from_config,
    load_prepared_moleculenet_dataset,
    load_single_table_with_split_column,
    load_table_eval_dataset,
    normalize_task_names,
    read_table,
)

# ---------------------------------------------------------------------------
# normalize_task_names
# ---------------------------------------------------------------------------


def test_normalize_task_names_single_string() -> None:
    assert normalize_task_names("label") == ["label"]


def test_normalize_task_names_sequence() -> None:
    assert normalize_task_names(["a", "b"]) == ["a", "b"]


def test_normalize_task_names_duplicate_fails() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_task_names(["label", "label"])


def test_normalize_task_names_empty_string_fails() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_task_names([""])


def test_normalize_task_names_empty_list_fails() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_task_names([])


# ---------------------------------------------------------------------------
# read_table
# ---------------------------------------------------------------------------


def test_read_table_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"smiles": ["CCO"], "label": [1]}).to_csv(csv_path, index=False)

    df = read_table(csv_path)

    assert list(df.columns) == ["smiles", "label"]
    assert len(df) == 1


def test_read_table_parquet(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]}).to_parquet(
        pq_path, index=False
    )

    df = read_table(pq_path)

    assert list(df.columns) == ["smiles", "label"]
    assert len(df) == 2


def test_read_table_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nonexistent.csv")


def test_read_table_unsupported_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "data.xyz"
    p.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported"):
        read_table(p)


# ---------------------------------------------------------------------------
# load_table_eval_dataset
# ---------------------------------------------------------------------------


def _write_split_csv(path: Path, n: int = 4) -> None:
    pd.DataFrame(
        {
            "smiles": ["CCO"] * n,
            "selfies": ["[C][C][O]"] * n,
            "label": [0, 1] * (n // 2),
        }
    ).to_csv(path, index=False)


def test_load_table_eval_dataset_basic(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_split_csv(train_path)
    _write_split_csv(test_path)

    ds = load_table_eval_dataset(
        name="my_assay",
        task_type="classification",
        task_names="label",
        train_path=train_path,
        test_path=test_path,
    )

    assert ds.name == "my_assay"
    assert ds.task_type == "classification"
    assert ds.task_names == ["label"]
    assert len(ds.train) == 4
    assert len(ds.test) == 4
    assert ds.valid is None


def test_load_table_eval_dataset_with_valid(tmp_path: Path) -> None:
    for name in ("train.csv", "valid.csv", "test.csv"):
        _write_split_csv(tmp_path / name)

    ds = load_table_eval_dataset(
        name="assay_with_valid",
        task_type="classification",
        task_names="label",
        train_path=tmp_path / "train.csv",
        valid_path=tmp_path / "valid.csv",
        test_path=tmp_path / "test.csv",
    )

    assert ds.valid is not None
    assert len(ds.valid) == 4


# ---------------------------------------------------------------------------
# load_single_table_with_split_column
# ---------------------------------------------------------------------------


def _write_split_column_table(path: Path) -> None:
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CO", "CN", "C"],
            "selfies": ["[C][C][O]", "[C][C][N]", "[C][O]", "[C][N]", "[C]"],
            "label": [0, 1, 0, 1, 0],
            "split": ["train", "train", "valid", "test", "test"],
        }
    ).to_csv(path, index=False)


def test_load_single_table_with_split_column(tmp_path: Path) -> None:
    table_path = tmp_path / "data.csv"
    _write_split_column_table(table_path)

    ds = load_single_table_with_split_column(
        name="split_col_ds",
        task_type="classification",
        task_names="label",
        table_path=table_path,
    )

    assert ds.name == "split_col_ds"
    assert len(ds.train) == 2
    assert ds.valid is not None
    assert len(ds.valid) == 1
    assert len(ds.test) == 2


def test_load_single_table_with_split_column_missing_split_col_raises(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "data.csv"
    pd.DataFrame({"smiles": ["CCO"], "label": [0]}).to_csv(table_path, index=False)

    with pytest.raises(ValueError, match="Missing split column"):
        load_single_table_with_split_column(
            name="ds",
            task_type="classification",
            task_names="label",
            table_path=table_path,
        )


# ---------------------------------------------------------------------------
# load_prepared_moleculenet_dataset
# ---------------------------------------------------------------------------


def _write_moleculenet_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "bbbp"
    dataset_dir.mkdir()

    metadata = {
        "name": "bbbp",
        "task_type": "classification",
        "tasks": ["p_np"],
        "source": "moleculenet",
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    for split in ("train", "valid", "test"):
        pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCN"],
                "selfies": ["[C][C][O]", "[C][C][N]"],
                "p_np": [0, 1],
            }
        ).to_parquet(dataset_dir / f"{split}.parquet", index=False)

    return dataset_dir


def test_load_prepared_moleculenet_dataset_preserves_metadata(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_moleculenet_dir(tmp_path)

    ds = load_prepared_moleculenet_dataset(dataset_dir=dataset_dir)

    assert ds.name == "bbbp"
    assert ds.task_type == "classification"
    assert ds.task_names == ["p_np"]
    assert ds.metadata["source"] == "moleculenet"
    assert ds.metadata["eval_split"] == "test"
    assert ds.metadata["dataset_dir"] == str(dataset_dir)


def test_load_prepared_moleculenet_dataset_eval_split_train_fails(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_moleculenet_dir(tmp_path)

    with pytest.raises(ValueError, match="eval_split"):
        load_prepared_moleculenet_dataset(dataset_dir=dataset_dir, eval_split="train")


# ---------------------------------------------------------------------------
# load_eval_dataset_from_config — prepared_moleculenet loader
# ---------------------------------------------------------------------------


def test_load_eval_dataset_from_config_prepared_moleculenet(tmp_path: Path) -> None:
    dataset_dir = _write_moleculenet_dir(tmp_path)

    config = {
        "loader": "prepared_moleculenet",
        "dataset_dir": str(dataset_dir),
        "eval_split": "test",
    }

    ds = load_eval_dataset_from_config(config)

    assert ds.name == "bbbp"
    assert ds.task_names == ["p_np"]
    assert isinstance(ds.train, pd.DataFrame)
    assert isinstance(ds.test, pd.DataFrame)


def test_load_eval_dataset_from_config_unknown_loader_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset loader"):
        load_eval_dataset_from_config({"loader": "nonexistent_loader"})
