from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import embedding_cost_worker
from measure_embedding_cost import (
    measurement_specs,
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


def test_remote_code_is_opt_in_per_encoder() -> None:
    specs = measurement_specs(
        {},
        fingerprints=False,
        devices=["cpu"],
        batch_sizes=[32],
        max_seq_length=128,
        transformers={"ChemBERTa-2": "org/chemberta", "MoLFormer": "org/molformer"},
        selfies_transformers={"SELFormer": "org/selformer"},
        trust_remote_code=["MoLFormer"],
    )
    by_name = {spec["name"]: spec for spec in specs}
    assert by_name["ChemBERTa-2"]["input"] == "smiles"
    assert by_name["SELFormer"]["input"] == "selfies"
    assert [name for name, spec in by_name.items() if spec["trust_remote_code"]] == ["MoLFormer"]
    with pytest.raises(ValueError, match="unique"):
        measurement_specs(
            {"x": "runs/a"},
            fingerprints=False,
            devices=["cpu"],
            batch_sizes=[32],
            max_seq_length=128,
            transformers={"x": "org/x"},
        )


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
