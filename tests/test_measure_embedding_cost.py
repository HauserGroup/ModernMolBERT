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
    parse_named,
    run_worker,
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


def test_named_arguments_need_unique_names() -> None:
    assert parse_named(["small=runs/a", "base=org/model"]) == {
        "small": "runs/a",
        "base": "org/model",
    }
    with pytest.raises(ValueError, match="Duplicate"):
        parse_named(["small=runs/a", "small=runs/b"])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_named(["ECFP4-count=runs/a"])
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_named(["runs/a"])


def test_fingerprints_are_timed_once_on_cpu() -> None:
    specs = measurement_specs(
        {"small": "runs/a"},
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


@pytest.mark.model
def test_transformer_worker_runs_offline_on_a_cached_encoder(tmp_path: Path) -> None:
    from huggingface_hub import try_to_load_from_cache

    model = "DeepChem/ChemBERTa-10M-MLM"
    if not isinstance(try_to_load_from_cache(model, "config.json"), str):
        pytest.skip(f"{model} is not in the local Hugging Face cache")
    molecules = tmp_path / "molecules.txt"
    molecules.write_text("CCO\nc1ccccc1\nCC(=O)O\n")
    spec = {
        "name": "cached",
        "kind": "transformer",
        "device": "cpu",
        "batch_size": 2,
        "max_seq_length": 128,
        "path": model,
        "input": "smiles",
        "trust_remote_code": False,
    }
    result = run_worker(spec, molecules)
    assert result["n_valid"] == 3
    assert result["feature_dim"] == 384
    assert result["conversion_seconds"] is not None
