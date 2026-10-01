import json
from pathlib import Path

import pytest

from modernmolbert.utils import file_sha256
from scripts import run_revision_common_scoring as scoring


def test_scoring_command_requires_fresh_verified_common_cohort(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scoring, "ROOT", tmp_path)
    campaign = tmp_path / "campaign.json"
    campaign.write_text('{"campaign": "test"}', encoding="utf-8")
    monkeypatch.setattr(scoring, "CAMPAIGN", campaign)
    config = tmp_path / "datasets.yaml"
    config.write_text("datasets:\n  clf_example:\n    name: example\n", encoding="utf-8")
    monkeypatch.setattr(scoring, "CONFIG", config)
    embedding = tmp_path / "data/embedded/example/REVISION_COMMON_small_ape_selfies.joblib"
    embedding.parent.mkdir(parents=True)
    embedding.write_bytes(b"frozen embedding")
    prepared = tmp_path / "data/prepared/example.json"
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"frozen prepared rows")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "campaign_manifest_sha256": file_sha256(campaign),
                "run_ids": sorted(scoring.RUN_IDS),
                "common_prefix": "REVISION_COMMON_",
                "tasks": {
                    "example": {
                        "prepared_sha256": file_sha256(prepared),
                        "models": {
                            "small_ape_selfies": {"common_embedding_sha256": file_sha256(embedding)}
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def run_cmd() -> list[str]:
        return scoring.command_for(
            manifest_path=manifest,
            run_id="small_ape_selfies",
            task="example",
            output_root=tmp_path / "scores",
            n_jobs=4,
        )

    command = run_cmd()
    assert command[-1].endswith("scores/small_ape_selfies/example.csv")
    assert "--resume" in command
    assert command[command.index("--missing-labels") + 1] == "as-negative"

    embedding.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        run_cmd()
    embedding.write_bytes(b"frozen embedding")
    output = Path(command[-1])
    output.parent.mkdir(parents=True)
    output.write_text("older result", encoding="utf-8")
    assert run_cmd() == command

    seed43_embedding = embedding.with_name("REVISION_COMMON_s43_small_ape_selfies.joblib")
    seed43_embedding.write_bytes(b"seed 43 common embedding")
    seed43_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    seed43_manifest["seed"] = 43
    seed43_manifest["common_prefix"] = "REVISION_COMMON_s43_"
    seed43_manifest["tasks"]["example"]["models"]["small_ape_selfies"][
        "common_embedding_sha256"
    ] = file_sha256(seed43_embedding)
    manifest.write_text(json.dumps(seed43_manifest), encoding="utf-8")
    seed43 = scoring.command_for(
        manifest_path=manifest,
        run_id="small_ape_selfies",
        task="example",
        output_root=tmp_path / "seed43_scores",
        n_jobs=4,
        seed=43,
        campaign_path=campaign,
    )
    assert seed43[seed43.index("--embedder") + 1] == "REVISION_COMMON_s43_small_ape_selfies"
