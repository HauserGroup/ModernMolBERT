import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.datasets import load_prepared_moleculenet_dataset
from modernmolbert.eval.moleculenet import (
    ALL_SPECS,
    canonicalize_and_selfies,
    collect_preparation_versions,
    compute_duplicate_stats,
    compute_scaffold_stats,
    compute_split_overlap_stats,
    compute_task_label_stats,
    deepchem_dataset_to_frame,
    grouped_random_split_frame,
    prepare_dataset,
    prepare_many,
    sanitize_frame,
    split_sanitized_frame,
)


REQUIRED_PREPARED_COLUMNS = [
    "molnet_row_id",
    "deepchem_id",
    "smiles_raw",
    "smiles_canonical",
    "selfies",
    "is_valid",
    "sanitize_error",
]


def _write_prepared_dataset(
    dataset_dir: Path,
    *,
    name: str = "toy",
    task_type: str = "classification",
    tasks: list[str] | None = None,
) -> None:
    tasks = tasks or ["label"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "name": name,
        "task_type": task_type,
        "tasks": tasks,
        "preferred_metric": "roc_auc" if task_type == "classification" else "rmse",
        "split": "scaffold",
        "split_seed": 13,
        "split_fractions": {"train": 0.8, "valid": 0.1, "test": 0.1},
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    frame = pd.DataFrame(
        {
            "molnet_row_id": [0, 1, 2, 3],
            "deepchem_id": ["a", "b", "c", "d"],
            "smiles_raw": ["CCO", "CCN", "CCC", "CO"],
            "smiles_canonical": ["CCO", "CCN", "CCC", "CO"],
            "selfies": ["[C][C][O]", "[C][C][N]", "[C][C][C]", "[C][O]"],
            "is_valid": [True, True, True, True],
            "sanitize_error": [None, None, None, None],
            tasks[0]: [0.0, 1.0, 1.0, 0.0],
        }
    )

    frame.iloc[:2].to_parquet(dataset_dir / "train.parquet", index=False)
    frame.iloc[2:3].to_parquet(dataset_dir / "valid.parquet", index=False)
    frame.iloc[3:].to_parquet(dataset_dir / "test.parquet", index=False)


class DummyDeepChemDataset:
    X = np.array(["CCO", "CCN", "CCC"], dtype=object)
    ids = np.array(["id_a", "id_b", "id_c"], dtype=object)
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    w = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )


def test_canonicalize_and_selfies_valid_smiles() -> None:
    canonical, selfies, error = canonicalize_and_selfies("CCO")

    assert error is None
    assert canonical == "CCO"
    assert selfies is not None
    assert "[C]" in selfies
    assert "[O]" in selfies


def test_canonicalize_and_selfies_invalid_smiles() -> None:
    canonical, selfies, error = canonicalize_and_selfies("not_a_smiles")

    assert canonical is None
    assert selfies is None
    assert error is not None


def test_sanitize_frame_adds_expected_columns_and_marks_invalid() -> None:
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
    assert out.loc[0, "smiles_canonical"] == "CCO"
    assert out.loc[0, "selfies"] is not None
    assert pd.isna(out.loc[0, "sanitize_error"])
    assert pd.isna(out.loc[1, "smiles_canonical"])
    assert pd.isna(out.loc[1, "selfies"])
    assert isinstance(out.loc[1, "sanitize_error"], str)


def test_deepchem_dataset_to_frame_preserves_ids_and_missing_labels() -> None:
    frame = deepchem_dataset_to_frame(
        dataset=DummyDeepChemDataset(),
        tasks=["task_a", "task_b"],
    )

    assert list(frame.columns[:3]) == ["molnet_row_id", "deepchem_id", "smiles_raw"]
    assert frame["molnet_row_id"].tolist() == [0, 1, 2]
    assert frame["deepchem_id"].tolist() == ["id_a", "id_b", "id_c"]
    assert frame["smiles_raw"].tolist() == ["CCO", "CCN", "CCC"]

    assert frame.loc[0, "task_a"] == 1.0
    assert pd.isna(frame.loc[0, "task_b"])

    assert pd.isna(frame.loc[1, "task_a"])
    assert frame.loc[1, "task_b"] == 1.0
    assert frame.loc[2, "task_a"] == 1.0
    assert frame.loc[2, "task_b"] == 1.0


def test_compute_task_label_stats_classification_detects_one_class_splits() -> None:
    splits = {
        "train": pd.DataFrame({"label": [0.0, 0.0, 1.0, 1.0, np.nan]}),
        "valid": pd.DataFrame({"label": [0.0, 0.0, np.nan]}),
        "test": pd.DataFrame({"label": [1.0, 1.0]}),
    }

    stats = compute_task_label_stats(
        splits=splits,
        tasks=["label"],
        task_type="classification",
    )

    assert stats["train"]["label"]["n_rows"] == 5
    assert stats["train"]["label"]["n_observed"] == 4
    assert stats["train"]["label"]["n_missing"] == 1
    assert stats["train"]["label"]["class_counts"] == {"0.0": 2, "1.0": 2}
    assert stats["train"]["label"]["n_classes_observed"] == 2
    assert stats["valid"]["label"]["n_classes_observed"] == 1
    assert stats["test"]["label"]["n_classes_observed"] == 1


def test_compute_task_label_stats_regression_summarizes_observed_values() -> None:
    splits = {
        "train": pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0]}),
        "valid": pd.DataFrame({"y": [np.nan]}),
    }

    stats = compute_task_label_stats(
        splits=splits,
        tasks=["y"],
        task_type="regression",
    )

    train = stats["train"]["y"]
    assert train["n_rows"] == 4
    assert train["n_observed"] == 3
    assert train["n_missing"] == 1
    assert train["min"] == 1.0
    assert train["max"] == 4.0
    assert train["mean"] == pytest.approx(7.0 / 3.0)

    valid = stats["valid"]["y"]
    assert valid["n_observed"] == 0
    assert valid["mean"] is None
    assert valid["std"] is None


def test_compute_duplicate_stats_reports_canonical_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "smiles_canonical": ["CCO", "CCO", "CCN", "CCC", None, "CCC"],
        }
    )

    stats = compute_duplicate_stats(frame)

    assert stats["n_valid_rows"] == 5
    assert stats["n_unique_canonical_smiles"] == 3
    assert stats["n_duplicate_rows"] == 2
    assert stats["duplicate_fraction"] == pytest.approx(2 / 5)
    assert stats["top_10_duplicate_group_sizes"][:2] == [2, 2]


def test_grouped_random_split_keeps_duplicate_canonical_smiles_together() -> None:
    frame = pd.DataFrame(
        {
            "smiles_canonical": ["CCO", "CCO", "CCN", "CCC", "CO", "CN", "CN"],
            "is_valid": [True] * 7,
            "label": [0, 0, 1, 1, 0, 1, 1],
        }
    )

    splits = grouped_random_split_frame(
        frame,
        group_column="smiles_canonical",
        seed=13,
        frac_train=0.5,
        frac_valid=0.25,
        frac_test=0.25,
    )

    overlap = compute_split_overlap_stats(splits, key="smiles_canonical")

    assert overlap["train_valid"]["n_overlap"] == 0
    assert overlap["train_test"]["n_overlap"] == 0
    assert overlap["valid_test"]["n_overlap"] == 0


def test_split_sanitized_frame_random_groups_duplicates_by_default() -> None:
    frame = pd.DataFrame(
        {
            "smiles_canonical": ["CCO", "CCO", "CCN", "CCC", "CO", "CN", "CN"],
            "is_valid": [True] * 7,
            "label": [0, 0, 1, 1, 0, 1, 1],
        }
    )

    splits = split_sanitized_frame(
        frame,
        split="random",
        seed=13,
        frac_train=0.5,
        frac_valid=0.25,
        frac_test=0.25,
    )

    overlap = compute_split_overlap_stats(splits, key="smiles_canonical")

    assert overlap["train_valid"]["n_overlap"] == 0
    assert overlap["train_test"]["n_overlap"] == 0
    assert overlap["valid_test"]["n_overlap"] == 0


def test_compute_split_overlap_stats_reports_examples() -> None:
    splits = {
        "train": pd.DataFrame({"smiles_canonical": ["CCO", "CCN"]}),
        "valid": pd.DataFrame({"smiles_canonical": ["CCO", "CCC"]}),
        "test": pd.DataFrame({"smiles_canonical": ["CO"]}),
    }

    stats = compute_split_overlap_stats(splits, key="smiles_canonical")

    assert stats["train_valid"]["n_overlap"] == 1
    assert stats["train_valid"]["examples"] == ["CCO"]
    assert stats["train_test"]["n_overlap"] == 0
    assert stats["valid_test"]["n_overlap"] == 0


def test_scaffold_split_raises_on_empty_valid_or_test() -> None:
    frame = pd.DataFrame(
        {
            "smiles_canonical": ["CCO"],
            "selfies": ["[C][C][O]"],
            "is_valid": [True],
        }
    )

    with pytest.raises(RuntimeError, match="Scaffold split produced an empty valid or test split"):
        split_sanitized_frame(
            frame,
            split="scaffold",
            seed=13,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )


def test_compute_scaffold_stats_returns_basic_summary() -> None:
    frame = pd.DataFrame(
        {
            "smiles_canonical": ["CCO", "CCN", "c1ccccc1", "c1ccccc1O"],
        }
    )

    stats = compute_scaffold_stats(frame)

    assert stats["n_scaffolds"] >= 1
    assert stats["largest_scaffold_size"] >= 1
    assert 0.0 < stats["largest_scaffold_fraction"] <= 1.0
    assert isinstance(stats["top_10_scaffold_sizes"], list)


def test_collect_preparation_versions_has_expected_keys() -> None:
    versions = collect_preparation_versions()

    for key in ["deepchem", "rdkit", "selfies", "pandas", "numpy"]:
        assert key in versions


def test_load_prepared_moleculenet_dataset_reads_metadata_and_splits(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "esol"
    _write_prepared_dataset(
        dataset_dir,
        name="esol",
        task_type="regression",
        tasks=["measured log solubility in mols per litre"],
    )

    ds = load_prepared_moleculenet_dataset(dataset_dir=dataset_dir)

    assert ds.name == "esol"
    assert ds.task_type == "regression"
    assert ds.task_names == ["measured log solubility in mols per litre"]
    assert ds.smiles_column == "smiles_canonical"
    assert ds.selfies_column == "selfies"
    assert len(ds.train) == 2
    assert len(ds.test) == 1
    assert ds.metadata["eval_split"] == "test"


def test_load_prepared_moleculenet_dataset_can_use_valid_as_eval_split(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tox21"
    _write_prepared_dataset(
        dataset_dir,
        name="tox21",
        task_type="classification",
        tasks=["nr-ar"],
    )

    ds = load_prepared_moleculenet_dataset(dataset_dir=dataset_dir, eval_split="valid")

    assert ds.metadata["eval_split"] == "valid"
    assert len(ds.test) == 1
    assert ds.test.iloc[0]["smiles_canonical"] == "CCC"


def test_prepare_many_writes_root_manifest_with_shared_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_prepare_dataset(**kwargs):
        dataset_out = kwargs["output_root"] / kwargs["spec"].name
        dataset_out.mkdir(parents=True)
        (dataset_out / "metadata.json").write_text("{}\n", encoding="utf-8")
        calls.append(kwargs)
        return dataset_out

    monkeypatch.setattr(
        "modernmolbert.eval.moleculenet.prepare_dataset",
        fake_prepare_dataset,
    )

    output_root = tmp_path / "prepared"
    prepare_many(
        dataset_names=["esol", "bbbp"],
        output_root=output_root,
        deepchem_data_dir=tmp_path / "raw",
        deepchem_save_dir=tmp_path / "processed",
        split="random",
        seed=17,
        frac_train=0.7,
        frac_valid=0.2,
        frac_test=0.1,
        keep_invalid=False,
    )

    assert len(calls) == 2
    assert calls[0]["seed"] == 17
    assert calls[0]["frac_train"] == 0.7
    assert calls[0]["frac_valid"] == 0.2
    assert calls[0]["frac_test"] == 0.1

    manifest_path = output_root / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"] == "random"
    assert manifest["split_seed"] == 17
    assert manifest["split_fractions"] == {
        "train": 0.7,
        "valid": 0.2,
        "test": 0.1,
    }
    assert [item["name"] for item in manifest["datasets"]] == ["esol", "bbbp"]


@pytest.mark.model
def test_deepchem_esol_prepare_smoke(tmp_path: Path) -> None:
    """Optional DeepChem integration smoke test.

    Enable with:
        MODERNMOLBERT_RUN_DEEPCHEM_TESTS=1 uv run pytest tests/test_eval_moleculenet.py -q -s
    """
    import os

    if os.environ.get("MODERNMOLBERT_RUN_DEEPCHEM_TESTS") != "1":
        pytest.skip("Set MODERNMOLBERT_RUN_DEEPCHEM_TESTS=1 to run DeepChem test.")

    prepare_dataset(
        spec=ALL_SPECS["esol"],
        output_root=tmp_path / "prepared",
        deepchem_data_dir=tmp_path / "deepchem_raw",
        deepchem_save_dir=tmp_path / "deepchem_processed",
        split="scaffold",
        seed=13,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        keep_invalid=False,
    )

    dataset_dir = tmp_path / "prepared" / "esol"
    assert (dataset_dir / "metadata.json").exists()
    assert (dataset_dir / "train.parquet").exists()
    assert (dataset_dir / "valid.parquet").exists()
    assert (dataset_dir / "test.parquet").exists()
    assert (tmp_path / "prepared" / "manifest.json").exists() is False

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["split_seed"] == 13
    assert metadata["split_fractions"] == {"train": 0.8, "valid": 0.1, "test": 0.1}
    assert "label_stats" in metadata
    assert "duplicate_stats" in metadata
    assert "split_overlap_stats" in metadata
    assert "versions" in metadata

    ds = load_prepared_moleculenet_dataset(dataset_dir=dataset_dir)
    assert len(ds.train) > 0
    assert len(ds.test) > 0
