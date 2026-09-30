from pathlib import Path

import pandas as pd

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


def test_prepare_chembl36_frame_filters_and_adds_selfies(tmp_path: Path) -> None:
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

    config = ChemBL36SelfiesPrepConfig(output_dir=tmp_path / "chembl36")
    out, stats = prepare_chembl36_frame(frame, config=config, return_stats=True)

    assert len(out) == 2
    assert out["is_valid"].to_numpy().all()
    assert "selfies" in out.columns
    assert "split_key" in out.columns
    assert stats["rows_after_dedupe"] == 4
    assert stats["rows_valid_after_conversion"] == 3
    assert stats["rows_after_filters"] == 2
    assert stats["sanitize_error_counts"]["failed_basic_filters"] == 1

    resumed, resumed_stats = prepare_chembl36_frame(frame, config=config, return_stats=True)
    pd.testing.assert_frame_equal(out, resumed)
    assert resumed_stats == stats

    changed_config = ChemBL36SelfiesPrepConfig(output_dir=config.output_dir, min_heavy_atoms=1)
    changed, _ = prepare_chembl36_frame(frame, config=changed_config, return_stats=True)
    assert len(changed) == 3


def test_prepare_chembl36_frame_dedupes_clean_split_keys(tmp_path: Path) -> None:
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

    out = prepare_chembl36_frame(
        frame,
        config=ChemBL36SelfiesPrepConfig(output_dir=tmp_path / "chembl36"),
    )

    assert len(out) == 1
    assert out.loc[0, "split_key"] == "same"


def test_split_by_hash_default_returns_train_valid_only() -> None:
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
        seed=13,
    )

    assert len(train) > 0
    assert len(valid) > 0
    assert test is None

    train_keys = set(train["split_key"])
    valid_keys = set(valid["split_key"])

    assert train_keys.isdisjoint(valid_keys)
    assert len(train_keys | valid_keys) == len(frame)


def test_split_by_hash_can_create_non_overlapping_test_split() -> None:
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
    assert test is not None
    assert len(test) > 0

    train_keys = set(train["split_key"])
    valid_keys = set(valid["split_key"])
    test_keys = set(test["split_key"])

    assert train_keys.isdisjoint(valid_keys)
    assert train_keys.isdisjoint(test_keys)
    assert valid_keys.isdisjoint(test_keys)
    assert len(train_keys | valid_keys | test_keys) == len(frame)
