import pandas as pd
import pytest

from modernmolbert.data.chembl36 import (
    ChemBL36SelfiesPrepConfig,
    canonicalize_and_selfies,
    prepare_chembl36_frame,
    split_by_hash,
)


def test_canonicalize_and_selfies_valid() -> None:
    out = canonicalize_and_selfies("CCO")

    assert out["is_valid"] is True
    assert out["smiles_canonical_clean"] == "CCO"
    assert out["selfies"] is not None
    assert out["sanitize_error"] is None


def test_canonicalize_and_selfies_invalid() -> None:
    out = canonicalize_and_selfies("not_a_smiles")

    assert out["is_valid"] is False
    assert out["selfies"] is None
    assert out["sanitize_error"] is not None


def test_prepare_chembl36_frame_filters_and_adds_selfies() -> None:
    frame = pd.DataFrame(
        {
            "chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
            "canonical_smiles": ["CCO", "not_a_smiles", "CCN", "C"],
            "standard_inchi_key": ["a", "b", "c", "d"],
            "molecule_type": [
                "Small molecule",
                "Small molecule",
                "Small molecule",
                "Small molecule",
            ],
            "heavy_atoms": [3, 5, 3, 1],
            "mw_freebase": [46.0, 100.0, 45.0, 16.0],
        }
    )

    config = ChemBL36SelfiesPrepConfig()
    out, stats = prepare_chembl36_frame(frame, config=config, return_stats=True)

    assert len(out) == 2
    assert out["is_valid"].all()
    assert "selfies" in out.columns
    assert "split_key" in out.columns
    assert stats["rows_after_dedupe"] == 4
    assert stats["rows_valid_after_conversion"] == 3
    assert stats["rows_after_filters"] == 2
    assert stats["sanitize_error_counts"]["failed_basic_filters"] == 1


def test_prepare_chembl36_frame_dedupes_clean_split_keys() -> None:
    frame = pd.DataFrame(
        {
            "chembl_id": ["CHEMBL1", "CHEMBL1_DUP"],
            "canonical_smiles": ["CCO", "OCC"],
            "standard_inchi_key": ["same", "same"],
            "molecule_type": ["Small molecule", "Small molecule"],
            "heavy_atoms": [3, 3],
            "mw_freebase": [46.0, 46.0],
        }
    )

    out = prepare_chembl36_frame(frame, config=ChemBL36SelfiesPrepConfig())

    assert len(out) == 1
    assert out.loc[0, "split_key"] == "same"


def test_prepare_chembl36_frame_requires_smiles_column() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_chembl36_frame(pd.DataFrame({"x": ["CCO"]}), config=ChemBL36SelfiesPrepConfig())


def test_split_by_hash_returns_non_overlapping_splits() -> None:
    frame = pd.DataFrame(
        {
            "split_key": [f"mol_{i}" for i in range(1000)],
            "selfies": ["[C]" for _ in range(1000)],
        }
    )

    train, valid, test = split_by_hash(
        frame,
        key_column="split_key",
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=13,
    )

    assert len(train) > 0
    assert len(valid) > 0
    assert len(test) > 0

    train_keys = set(train["split_key"])
    valid_keys = set(valid["split_key"])
    test_keys = set(test["split_key"])

    assert train_keys.isdisjoint(valid_keys)
    assert train_keys.isdisjoint(test_keys)
    assert valid_keys.isdisjoint(test_keys)


def test_split_by_hash_changes_with_seed() -> None:
    frame = pd.DataFrame({"split_key": [f"mol_{i}" for i in range(1000)]})

    train_a, valid_a, test_a = split_by_hash(
        frame,
        key_column="split_key",
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=13,
    )
    train_b, valid_b, test_b = split_by_hash(
        frame,
        key_column="split_key",
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=14,
    )

    assert set(train_a["split_key"]) != set(train_b["split_key"])
    assert set(valid_a["split_key"]) != set(valid_b["split_key"])
    assert set(test_a["split_key"]) != set(test_b["split_key"])
