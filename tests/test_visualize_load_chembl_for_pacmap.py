from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.visualize.load_chembl_for_pacmap import (
    load_chembl_selfies,
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


def test_load_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"chembl_id": ["CHEMBL1"]}).to_parquet(path)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_chembl_selfies(path, property_column="alogp")
