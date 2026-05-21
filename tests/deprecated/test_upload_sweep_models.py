import json
from pathlib import Path

from modernmolbert.upload_model import upload_model_to_hub
from modernmolbert.upload_sweep_models import (
    build_upload_plans,
    repo_id_for_run,
    slugify_run_name,
    write_manifest,
)


def _make_final_model(run_dir: Path) -> None:
    final_model = run_dir / "final_model"
    final_model.mkdir(parents=True)
    (final_model / "model.safetensors").write_bytes(b"weights")
    (final_model / "config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "run_args.json").write_text("{}\n", encoding="utf-8")


def test_slugify_run_name_matches_sweep_names() -> None:
    assert slugify_run_name("mask_standard__mlm_0p15__lr_1e-4") == (
        "mask-standard-mlm-0p15-lr-1e-4"
    )


def test_repo_id_for_run_appends_readable_slug() -> None:
    assert (
        repo_id_for_run(
            "HauserGroup/ModernMolBERT-small-chembl36",
            "mask_span__mlm_0p20__lr_4e-4",
        )
        == "HauserGroup/ModernMolBERT-small-chembl36-mask-span-mlm-0p20-lr-4e-4"
    )


def test_build_upload_plans_discovers_final_models_and_skips_incomplete(tmp_path) -> None:
    complete = tmp_path / "mask_standard__mlm_0p15__lr_1e-4"
    _make_final_model(complete)

    incomplete = tmp_path / "mask_span__mlm_0p20__lr_2e-4"
    incomplete.mkdir()
    (incomplete / "run_args.json").write_text("{}\n", encoding="utf-8")

    plans, skipped = build_upload_plans(
        run_root=tmp_path,
        repo_prefix="org/model",
        checkpoint="final",
    )

    assert len(plans) == 1
    assert plans[0].run_dir == complete
    assert plans[0].source_dir == complete / "final_model"
    assert plans[0].repo_id == "org/model-mask-standard-mlm-0p15-lr-1e-4"
    assert skipped == [
        {
            "run_dir": str(incomplete),
            "reason": f"model.safetensors not found in {incomplete / 'final_model'}",
        }
    ]


def test_write_manifest_records_results_and_skips(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"

    write_manifest(
        path=manifest,
        run_root=tmp_path,
        checkpoint="final",
        dry_run=True,
        results=[{"repo_id": "org/model-run", "uploaded": False}],
        skipped=[{"run_dir": str(tmp_path / "bad"), "reason": "missing"}],
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == "final"
    assert payload["dry_run"] is True
    assert payload["results"][0]["repo_id"] == "org/model-run"
    assert payload["skipped"][0]["reason"] == "missing"


class _ExplodingApi:
    def create_repo(self, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("dry run should not create repos")

    def upload_folder(self, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("dry run should not upload folders")


class _RecordingApi:
    def __init__(self) -> None:
        self.create_repo_calls = []
        self.upload_folder_calls = []
        self.uploaded_files = []

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def upload_folder(self, **kwargs):
        folder_path = Path(kwargs["folder_path"])
        self.uploaded_files = sorted(path.name for path in folder_path.iterdir())
        self.upload_folder_calls.append(
            {key: value for key, value in kwargs.items() if key != "folder_path"}
        )


def _make_hub_ready_run(run_dir: Path) -> None:
    _make_final_model(run_dir)
    (run_dir / "trainer_state.json").write_text(
        '{"best_metric": 0.5, "best_global_step": 10}\n', encoding="utf-8"
    )
    ape = run_dir / "final_model" / "ape_tokenizer"
    ape.mkdir()
    for name in ("vocab.json", "tokenizer_config.json", "special_tokens_map.json"):
        (ape / name).write_text("{}\n", encoding="utf-8")
    (ape / "tokenization_ape.py").write_text("# tokenizer shim\n", encoding="utf-8")


def test_upload_model_to_hub_dry_run_stages_without_api_calls(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    _make_hub_ready_run(run_dir)

    result = upload_model_to_hub(
        run_dir=run_dir,
        repo_id="org/model-run-a",
        private=True,
        dry_run=True,
        api=_ExplodingApi(),
    )

    assert result["uploaded"] is False
    assert result["repo_id"] == "org/model-run-a"
    assert result["private"] is True
    assert result["staged_files"] == [
        "README.md",
        "config.json",
        "model.safetensors",
        "run_args.json",
        "special_tokens_map.json",
        "tokenization_ape.py",
        "tokenizer_config.json",
        "trainer_state.json",
        "vocab.json",
    ]


def test_upload_model_to_hub_uses_injected_api_without_network(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    _make_hub_ready_run(run_dir)
    api = _RecordingApi()

    result = upload_model_to_hub(
        run_dir=run_dir,
        repo_id="org/model-run-a",
        private=True,
        commit_message="Upload test model",
        token="unused-token",
        dry_run=False,
        api=api,
    )

    assert result["uploaded"] is True
    assert api.create_repo_calls == [
        {
            "repo_id": "org/model-run-a",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.upload_folder_calls == [
        {
            "repo_id": "org/model-run-a",
            "repo_type": "model",
            "commit_message": "Upload test model",
        }
    ]
    assert api.uploaded_files == result["staged_files"]
