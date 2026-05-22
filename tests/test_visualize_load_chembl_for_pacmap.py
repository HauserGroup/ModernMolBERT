from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.visualize.load_chembl_for_pacmap import (
    load_chembl_selfies,
    write_subset,
)


def _minimal_parquet(tmp_path: Path, *, rows: list[dict]) -> Path:
    path = tmp_path / "chembl.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


_BASE_ROWS = [
    {
        "chembl_id": "CHEMBL1",
        "smiles_canonical_clean": "CCO",
        "selfies": "[C][C][O]",
        "alogp": 1.0,
        "is_valid": True,
    },
    {
        "chembl_id": "CHEMBL2",
        "smiles_canonical_clean": "CCN",
        "selfies": "[C][C][N]",
        "alogp": 0.5,
        "is_valid": True,
    },
    {
        "chembl_id": "CHEMBL3",
        "smiles_canonical_clean": "CN",
        "selfies": "[C][N]",
        "alogp": -1.0,
        "is_valid": False,
    },
]


def test_load_filters_invalid_rows(tmp_path: Path) -> None:
    path = _minimal_parquet(tmp_path, rows=_BASE_ROWS)
    df = load_chembl_selfies(path, property_column="alogp", only_valid=True)
    assert (df["is_valid"] == True).all()  # noqa: E712
    assert len(df) == 2


def test_load_includes_invalid_when_disabled(tmp_path: Path) -> None:
    path = _minimal_parquet(tmp_path, rows=_BASE_ROWS)
    df = load_chembl_selfies(path, property_column="alogp", only_valid=False)
    assert len(df) == 3


def test_load_drops_null_selfies(tmp_path: Path) -> None:
    rows = _BASE_ROWS + [
        {
            "chembl_id": "CHEMBL4",
            "smiles_canonical_clean": "CO",
            "selfies": None,
            "alogp": 2.0,
            "is_valid": True,
        }
    ]
    path = _minimal_parquet(tmp_path, rows=rows)
    df = load_chembl_selfies(path, property_column="alogp", only_valid=False)
    assert df["selfies"].notna().all()


def test_load_drops_null_property(tmp_path: Path) -> None:
    rows = _BASE_ROWS + [
        {
            "chembl_id": "CHEMBL5",
            "smiles_canonical_clean": "CO",
            "selfies": "[C][O]",
            "alogp": None,
            "is_valid": True,
        }
    ]
    path = _minimal_parquet(tmp_path, rows=rows)
    df = load_chembl_selfies(path, property_column="alogp", only_valid=False)
    assert df["alogp"].notna().all()


def test_load_sample_size_limits_rows(tmp_path: Path) -> None:
    rows = [
        {
            "chembl_id": f"CHEMBL{i}",
            "smiles_canonical_clean": "CCO",
            "selfies": "[C][C][O]",
            "alogp": float(i),
            "is_valid": True,
        }
        for i in range(50)
    ]
    path = _minimal_parquet(tmp_path, rows=rows)
    df = load_chembl_selfies(path, property_column="alogp", sample_size=10, seed=0)
    assert len(df) == 10


def test_load_sample_is_deterministic(tmp_path: Path) -> None:
    rows = [
        {
            "chembl_id": f"CHEMBL{i}",
            "smiles_canonical_clean": "CCO",
            "selfies": "[C][C][O]",
            "alogp": float(i),
            "is_valid": True,
        }
        for i in range(50)
    ]
    path = _minimal_parquet(tmp_path, rows=rows)
    df_a = load_chembl_selfies(path, property_column="alogp", sample_size=10, seed=42)
    df_b = load_chembl_selfies(path, property_column="alogp", sample_size=10, seed=42)
    assert list(df_a["chembl_id"]) == list(df_b["chembl_id"])


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_chembl_selfies(tmp_path / "nonexistent.parquet", property_column="alogp")


def test_load_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"chembl_id": ["CHEMBL1"]}).to_parquet(path)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_chembl_selfies(path, property_column="alogp")


def test_write_subset_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]", "[O]"], "alogp": [1.0, 2.0]})
    out = tmp_path / "out.parquet"
    write_subset(df, out)
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == 2


def test_write_subset_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]"], "alogp": [1.0]})
    out = tmp_path / "out.csv"
    write_subset(df, out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert "selfies" in loaded.columns


def test_write_subset_unsupported_suffix_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Unsupported output suffix"):
        write_subset(df, tmp_path / "out.json")
