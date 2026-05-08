import json
from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.eval.datasets import load_prepared_moleculenet_dataset
from modernmolbert.eval.moleculenet import canonicalize_and_selfies, sanitize_frame


def test_canonicalize_and_selfies_valid_smiles() -> None:
    canonical, selfies, error = canonicalize_and_selfies("CCO")

    assert error is None
    assert canonical is not None
    assert selfies is not None
    assert "[C]" in selfies
    assert "[O]" in selfies


def test_canonicalize_and_selfies_invalid_smiles() -> None:
    canonical, selfies, error = canonicalize_and_selfies("not_a_smiles")

    assert canonical is None
    assert selfies is None
    assert error is not None


def test_sanitize_frame_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "smiles_raw": ["CCO", "not_a_smiles"],
            "label": [1.0, 0.0],
        }
    )

    out = sanitize_frame(df)

    assert list(out.columns[:5]) == [
        "smiles_raw",
        "smiles_canonical",
        "selfies",
        "is_valid",
        "sanitize_error",
    ]

    assert out["is_valid"].tolist() == [True, False]
    assert out.loc[0, "selfies"] is not None
    assert pd.isna(out.loc[1, "selfies"])


def test_load_prepared_moleculenet_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "esol"
    dataset_dir.mkdir()

    metadata = {
        "name": "esol",
        "task_type": "regression",
        "tasks": ["measured log solubility in mols per litre"],
    }

    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )

    train = pd.DataFrame(
        {
            "smiles_raw": ["CCO", "CCN"],
            "smiles_canonical": ["CCO", "CCN"],
            "selfies": ["[C][C][O]", "[C][C][N]"],
            "is_valid": [True, True],
            "sanitize_error": [None, None],
            "measured log solubility in mols per litre": [0.1, 0.2],
        }
    )
    valid = train.copy()
    test = train.copy()

    train.to_parquet(dataset_dir / "train.parquet", index=False)
    valid.to_parquet(dataset_dir / "valid.parquet", index=False)
    test.to_parquet(dataset_dir / "test.parquet", index=False)

    ds = load_prepared_moleculenet_dataset(dataset_dir=dataset_dir)

    assert ds.name == "esol"
    assert ds.task_type == "regression"
    assert ds.task_names == ["measured log solubility in mols per litre"]
    assert ds.smiles_column == "smiles_canonical"
    assert ds.selfies_column == "selfies"
    assert len(ds.train) == 2
    assert len(ds.test) == 2


@pytest.mark.model
def test_deepchem_esol_prepare_smoke(tmp_path: Path) -> None:
    """Optional DeepChem integration smoke test.

    Enable with:
        MODERNMOLBERT_RUN_DEEPCHEM_TESTS=1 uv run pytest tests/test_eval_moleculenet.py -q -s
    """
    import os

    if os.environ.get("MODERNMOLBERT_RUN_DEEPCHEM_TESTS") != "1":
        pytest.skip("Set MODERNMOLBERT_RUN_DEEPCHEM_TESTS=1 to run DeepChem test.")

    from modernmolbert.eval.moleculenet import ALL_SPECS, prepare_dataset

    prepare_dataset(
        spec=ALL_SPECS["esol"],
        output_root=tmp_path / "prepared",
        deepchem_data_dir=tmp_path / "deepchem_raw",
        deepchem_save_dir=tmp_path / "deepchem_processed",
        split="scaffold",
        keep_invalid=False,
    )

    dataset_dir = tmp_path / "prepared" / "esol"
    assert (dataset_dir / "metadata.json").exists()
    assert (dataset_dir / "train.parquet").exists()
    assert (dataset_dir / "valid.parquet").exists()
    assert (dataset_dir / "test.parquet").exists()

    ds = load_prepared_moleculenet_dataset(dataset_dir=dataset_dir)
    assert len(ds.train) > 0
    assert len(ds.test) > 0
