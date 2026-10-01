import pytest
import json
from pathlib import Path

from scripts import run_revision_factorial_v1 as launcher


def test_launcher_rejects_occupied_gpu(monkeypatch):
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "106138, /opt/lab/envs/esmfold2-py312/bin/python\n",
    )
    with pytest.raises(RuntimeError, match="already in use"):
        launcher.check_gpu_available()


def test_launcher_accepts_idle_gpu(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_args, **_kwargs: "")
    launcher.check_gpu_available()


def test_five_commands_share_frozen_recipe_and_manifest():
    spec = launcher.read_spec()
    assert set(spec["runs"]) == {
        "small_ape_selfies",
        "small_ape_smiles",
        "small_bpe_selfies",
        "small_bpe_smiles",
        "base_ape_selfies",
    }
    for run_id, run in spec["runs"].items():
        command = launcher.command_for(run_id, None, Path("campaign.json"), spec)
        assert command[command.index("--campaign_manifest") + 1] == "campaign.json"
        assert command[command.index("--model_size") + 1] == run["model_size"]
        assert command[command.index("--molecule_column") + 1] == run["molecule_column"]
        assert command[command.index("--max_steps") + 1] == "30000"
        assert command[command.index("--train_order_path") + 1].endswith("train_order_seed42.npy")
        assert "--no-load_best_model_at_end" in command


def test_manifest_rejects_another_code_revision(tmp_path: Path, monkeypatch):
    spec = {"runs": {"model": {}}, "frozen_files": {}}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("spec", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "SPEC", spec_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda args, **_kwargs: "b" * 40 if "rev-parse" in args else "",
    )
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "code_commit": "a" * 40,
                "spec_sha256": launcher.file_sha256(spec_path),
                "run_ids": ["model"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        launcher.check_campaign_manifest(manifest, spec)


def test_manifest_rejects_changed_frozen_input(tmp_path: Path, monkeypatch):
    frozen = tmp_path / "train.parquet"
    frozen.write_bytes(b"original")
    spec = {"runs": {"model": {}}, "frozen_files": {"train.parquet": launcher.file_sha256(frozen)}}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("spec", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "SPEC", spec_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda args, **_kwargs: "a" * 40 if "rev-parse" in args else "",
    )
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "code_commit": "a" * 40,
                "spec_sha256": launcher.file_sha256(spec_path),
                "run_ids": ["model"],
                "frozen_files_sha256": spec["frozen_files"],
            }
        ),
        encoding="utf-8",
    )
    launcher.check_campaign_manifest(manifest, spec)
    frozen.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="missing or changed"):
        launcher.check_campaign_manifest(manifest, spec)
