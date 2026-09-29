import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import embedding_cost_worker
from measure_embedding_cost import (
    benchmark_test_smiles,
    measurement_specs,
    parse_checkpoints,
    sample_molecules,
    summarise,
)
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset


def write_prepared(directory: Path, name: str, smiles: list[str], test_rows: list[int]) -> None:
    train_rows = [i for i in range(len(smiles)) if i not in test_rows]
    Dataset(
        name=name,
        task="classification",
        data=pd.DataFrame({"smiles": smiles, "label": [0, 1] * (len(smiles) // 2)}),
        splits={"train": train_rows, "valid": [], "test": test_rows},
    ).serialize_legacy(directory / f"{name}.json")


def test_test_smiles_are_distinct_across_datasets(tmp_path: Path) -> None:
    write_prepared(tmp_path, "a", ["CCO", "CCN", "CCC", "c1ccccc1"], [2, 3])
    write_prepared(tmp_path, "b", ["CCC", "O", "N", "CC"], [0, 1])
    config = tmp_path / "datasets.yaml"
    config.write_text(
        yaml.safe_dump({"datasets": {"clf_a": {"name": "a"}, "clf_b": {"name": "b"}}})
    )

    smiles, hashes, n_rows = benchmark_test_smiles(config, tmp_path)

    assert smiles == ["CCC", "O", "c1ccccc1"]
    assert n_rows == 4
    assert set(hashes) == {"a", "b"} and all(len(value) == 64 for value in hashes.values())


def test_sample_is_seeded_and_capped() -> None:
    smiles = [f"C{'C' * i}O" for i in range(20)]
    first = sample_molecules(smiles, 5, seed=42)
    assert first == sample_molecules(smiles, 5, seed=42)
    assert len(first) == 5 and len(set(first)) == 5
    assert sorted(sample_molecules(smiles, None, seed=0)) == sorted(smiles)


def test_checkpoint_arguments_need_unique_names() -> None:
    assert parse_checkpoints(["small=runs/a", "base=runs/b"]) == {
        "small": Path("runs/a"),
        "base": Path("runs/b"),
    }
    with pytest.raises(ValueError, match="Duplicate"):
        parse_checkpoints(["small=runs/a", "small=runs/b"])
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_checkpoints(["runs/a"])


def test_fingerprints_are_timed_once_on_cpu() -> None:
    specs = measurement_specs(
        {"small": Path("runs/a")},
        fingerprints=True,
        devices=["cpu", "mps"],
        batch_sizes=[32, 128],
        max_seq_length=128,
    )
    fingerprints = [spec for spec in specs if spec["kind"] == "fingerprint"]
    checkpoints = [spec for spec in specs if spec["kind"] == "checkpoint"]
    assert [spec["device"] for spec in fingerprints] == ["cpu", "cpu"]
    assert {(spec["device"], spec["batch_size"]) for spec in checkpoints} == {
        ("cpu", 32),
        ("cpu", 128),
        ("mps", 32),
        ("mps", 128),
    }


def test_summary_counts_all_inputs_in_throughput() -> None:
    rows = [
        {
            "name": "small",
            "kind": "checkpoint",
            "device": "cpu",
            "batch_size": 32,
            "n_molecules": 100,
            "n_valid": 90,
            "n_truncated": 1,
            "end_to_end_seconds": seconds,
            "first_pass_seconds": 2 * seconds,
            "conversion_seconds": 0.1,
            "tokenisation_seconds": 0.2,
            "features_seconds": 0.5,
            "import_seconds": 1.0,
            "load_seconds": 0.3,
            "rss_start_bytes": 10,
            "peak_rss_bytes": peak,
            "peak_mps_driver_bytes": None,
            "n_parameters": 1000,
            "feature_dim": 512,
        }
        for seconds, peak in ((1.0, 50), (2.0, 70), (4.0, 60))
    ]
    fingerprint = {
        **rows[0],
        "name": "ECFP4-binary",
        "kind": "fingerprint",
        "batch_size": None,
        "feature_dim": 2048,
    }
    summary = summarise(pd.DataFrame([*rows, fingerprint])).set_index("name")

    assert summary.loc["small", "repeats"] == 3
    assert summary.loc["small", "median_molecules_per_second"] == 50.0
    assert summary.loc["small", "min_molecules_per_second"] == 25.0
    assert summary.loc["small", "median_first_pass_molecules_per_second"] == 25.0
    assert summary.loc["small", "max_peak_rss_bytes"] == 70
    assert summary.loc["small", "float32_bytes_per_molecule"] == 2048
    assert summary.loc["ECFP4-binary", "repeats"] == 1


def test_fingerprint_worker_reports_parse_failures() -> None:
    spec = {"name": "ECFP4-count", "kind": "fingerprint", "device": "cpu", "batch_size": None}
    result = embedding_cost_worker.measure_fingerprint(spec, ["CCO", "not_a_smiles", "c1ccccc1"])
    assert result["n_molecules"] == 3
    assert result["n_valid"] == 2
    assert result["feature_dim"] == embedding_cost_worker.ECFP_BITS
    peak, after_load = result["peak_rss_bytes"], result["rss_after_load_bytes"]
    assert isinstance(peak, int) and isinstance(after_load, int)
    assert peak >= after_load


def test_worker_prints_json_last(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    molecules = tmp_path / "molecules.txt"
    molecules.write_text("CCO\nCCN\n")
    spec = {"name": "ECFP4-binary", "kind": "fingerprint", "device": "cpu", "batch_size": None}
    embedding_cost_worker.main([json.dumps(spec), str(molecules)])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["n_valid"] == 2
    assert result["tokenisation_seconds"] is None
    assert result["rss_start_bytes"] > 0


@pytest.mark.model
def test_checkpoint_worker_matches_featurizer(existing_minimal_model: Path) -> None:
    spec = {
        "name": "minimal",
        "kind": "checkpoint",
        "device": "cpu",
        "batch_size": 2,
        "max_seq_length": 128,
        "path": str(existing_minimal_model),
    }
    result = embedding_cost_worker.measure_checkpoint(spec, ["CCO", "c1ccccc1", "CC(=O)O"])
    assert result["n_valid"] == result["n_valid_stage_split"]
    seconds = result["end_to_end_seconds"]
    assert isinstance(seconds, float) and np.isfinite(seconds)
