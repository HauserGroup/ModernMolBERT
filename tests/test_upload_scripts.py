import hashlib
import json
from pathlib import Path

import pytest

from modernmolbert import upload_dataset, upload_model, upload_tokenizer


class _RecordingApi:
    def __init__(self) -> None:
        self.create_repo_calls: list[dict] = []
        self.upload_folder_calls: list[dict] = []

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def upload_folder(self, **kwargs):
        self.upload_folder_calls.append(kwargs)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_dataset_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "train.parquet").write_bytes(b"train")
    (path / "valid.parquet").write_bytes(b"valid")
    (path / "example.tsv").write_text("smiles\tselfies\n", encoding="utf-8")
    _write_json(
        path / "metadata.json",
        {
            "dataset_name": "lukaskim/ChEMBL-36",
            "representation": "SELFIES",
            "config": {
                "min_heavy_atoms": 3,
                "max_heavy_atoms": 100,
                "max_mw": 1000.0,
                "valid_fraction": 0.01,
            },
            "preparation_stats": {
                "input_rows": 10,
                "rows_after_dedupe": 9,
                "rows_valid_after_conversion": 8,
                "rows_after_filters": 7,
                "dropped_invalid_or_filtered": 3,
            },
            "row_counts": {
                "train": 6,
                "valid": 1,
                "prepared_total": 7,
            },
            "versions": {"datasets": "4.8.5"},
        },
    )
    return path


def _make_vocab_and_metadata(tmp_path: Path) -> tuple[Path, Path]:
    vocab_path = tmp_path / "vocab.json"
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
    }
    _write_json(vocab_path, vocab)

    digest = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "vocab.metadata.json"
    _write_json(
        metadata_path,
        {
            "representation": "SELFIES",
            "max_merge_pieces": 2,
            "min_freq_for_merge": 3000,
            "tokenizer_train_size": "2M",
            "vocab_size": len(vocab),
            "tokenizer_sha256": digest,
            "special_ids": {
                "bos_token": 0,
                "pad_token": 1,
                "eos_token": 2,
                "unk_token": 3,
                "mask_token": 4,
            },
        },
    )
    return vocab_path, metadata_path


def test_write_collator_config_writes_expected_defaults(tmp_path: Path) -> None:
    out_dir = tmp_path / "staging"
    out_dir.mkdir()

    upload_model.write_collator_config(out_dir, "hetero_span")

    payload = json.loads((out_dir / "collator_config.json").read_text(encoding="utf-8"))
    assert payload["masking_strategy"] == "hetero_span"
    assert payload["mlm_probability"] == 0.20
    assert payload["span_p_geom"] == 0.4
    assert payload["span_max_length"] == 6
    assert payload["heteroatom_start_weight"] == 2.0
    assert "Change masking_strategy" in payload["_note"]


def test_upload_dataset_to_hub_dry_run_stages_files(tmp_path: Path) -> None:
    dataset_dir = _make_dataset_dir(tmp_path / "dataset")

    result = upload_dataset.upload_dataset_to_hub(
        dataset_dir=dataset_dir,
        repo_id="org/chembl36-selfies",
        dry_run=True,
        keep_staging_dir=tmp_path / "staging",
    )

    assert result["uploaded"] is False
    assert sorted(result["staged_files"]) == [
        "README.md",
        "example.tsv",
        "metadata.json",
        "train.parquet",
        "valid.parquet",
    ]


def test_upload_dataset_to_hub_uses_injected_api(tmp_path: Path) -> None:
    dataset_dir = _make_dataset_dir(tmp_path / "dataset")
    api = _RecordingApi()

    result = upload_dataset.upload_dataset_to_hub(
        dataset_dir=dataset_dir,
        repo_id="org/chembl36-selfies",
        private=True,
        commit_message="Upload dataset test",
        dry_run=False,
        api=api,  # type: ignore
    )

    assert result["uploaded"] is True
    assert api.create_repo_calls == [
        {
            "repo_id": "org/chembl36-selfies",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert len(api.upload_folder_calls) == 1
    assert api.upload_folder_calls[0]["repo_id"] == "org/chembl36-selfies"
    assert api.upload_folder_calls[0]["repo_type"] == "dataset"
    assert api.upload_folder_calls[0]["commit_message"] == "Upload dataset test"


def test_verify_metadata_accepts_valid_payload(tmp_path: Path) -> None:
    vocab_path, metadata_path = _make_vocab_and_metadata(tmp_path)
    metadata = upload_tokenizer.load_metadata(metadata_path)

    representation = upload_tokenizer.verify_metadata(metadata, vocab_path)

    assert representation == "SELFIES"


def test_upload_tokenizer_to_hub_dry_run_stages_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab_path, metadata_path = _make_vocab_and_metadata(tmp_path)
    staging_dir = tmp_path / "staging-tokenizer"

    class _FakeTokenizer:
        def __init__(self, *, representation: str, model_max_length: int) -> None:
            self.representation = representation
            self.model_max_length = model_max_length

        def load_vocabulary_file(self, vocab_file: str) -> None:
            self._vocab_file = Path(vocab_file)

        def save_pretrained(self, out_dir: str) -> None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "vocab.json").write_text(
                self._vocab_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
            _write_json(
                out / "tokenizer_config.json",
                {
                    "representation": self.representation,
                    "model_max_length": self.model_max_length,
                },
            )
            _write_json(out / "special_tokens_map.json", {"mask_token": "<mask>"})

    monkeypatch.setattr(upload_tokenizer, "APEPreTrainedTokenizer", _FakeTokenizer)

    called = {"verify_saved": False}

    def _fake_verify_saved_tokenizer(*args, **kwargs):
        called["verify_saved"] = True

    def _fake_write_readme(staging: Path, repo_id: str) -> None:
        (staging / "README.md").write_text(f"# {repo_id}\n", encoding="utf-8")

    monkeypatch.setattr(upload_tokenizer, "verify_saved_tokenizer", _fake_verify_saved_tokenizer)
    monkeypatch.setattr(upload_tokenizer, "_write_readme", _fake_write_readme)

    upload_tokenizer.upload_tokenizer_to_hub(
        repo_id="org/ape-tokenizer",
        vocab_path=vocab_path,
        metadata_path=metadata_path,
        staging_dir=staging_dir,
        dry_run=True,
    )

    assert called["verify_saved"] is True
    assert sorted(path.name for path in staging_dir.iterdir()) == [
        "README.md",
        "ape_tokenizer_metadata.json",
        "metadata.json",
        "special_tokens_map.json",
        "tokenization_ape.py",
        "tokenizer_config.json",
        "tokenizer_metadata.json",
        "vocab.json",
    ]
