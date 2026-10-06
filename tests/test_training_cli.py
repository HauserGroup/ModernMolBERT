import argparse
import json
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
from datasets import Dataset
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

from modernmolbert.train_selfies_ape_modernbert import (
    adjust_args_for_backend,
    build_modernbert_config,
    compute_metrics,
    _encode_without_truncation,
    finalize_run_identity,
    make_eval_dataset,
    make_train_iterable_dataset,
    prepare_run_directory,
    _run_input_hashes,
    parse_args,
    preprocess_logits_for_metrics,
    sequence_bucket,
    validate_args,
)
from modernmolbert.utils import load_run_args


class _Argv:
    def __init__(self, *args: str):
        self._args = ["prog", *args]
        self._old = []

    def __enter__(self):
        self._old = sys.argv
        sys.argv = self._args

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self._old


def test_debug_keeps_frozen_validation_population():
    args = argparse.Namespace(
        debug=True,
        eval_size=4096,
        validation_row_ids_path=Path("validation_rows.npy"),
        max_steps=30_000,
        logging_steps=100,
        eval_steps=5000,
        save_steps=5000,
        tokenizer_validation_samples=10_000,
        bf16=True,
        fp16=False,
        num_workers=4,
    )
    adjusted = adjust_args_for_backend(args, "cuda")
    assert adjusted.eval_size == 4096
    assert adjusted.max_steps == 200
    assert adjusted.eval_steps == adjusted.save_steps == 50


def test_molecular_model_config_uses_vocabulary_boundaries_for_cls_and_sep(monkeypatch):
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.AutoConfig.from_pretrained",
        lambda *_args: ModernBertConfig(),
    )
    config = build_modernbert_config(
        argparse.Namespace(model_size="small", max_seq_length=128),
        vocab_size=588,
        special_ids={
            "pad_token": 1,
            "bos_token": 0,
            "eos_token": 2,
            "unk_token": 3,
            "mask_token": 4,
        },
    )
    assert config.cls_token_id == config.bos_token_id == 0
    assert config.sep_token_id == config.eos_token_id == 2
    assert max(config.cls_token_id, config.sep_token_id, config.pad_token_id) < 588


def test_validate_args_rejects_unsupported_cuda_bf16(monkeypatch):
    args = argparse.Namespace(
        max_seq_length=128,
        mlm_probability=0.15,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        val_split_mod=100,
        val_split_bucket=0,
        device_backend="cuda",
        eval_size=1000,
        max_eval_batches=0,
        load_best_model_at_end=True,
        save_steps=500,
        eval_steps=500,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="Use --no-bf16"):
        validate_args(args, backend="cuda")


def test_frozen_train_order_requires_global_shuffle():
    args = argparse.Namespace(train_order_path=Path("order.npy"), global_train_shuffle=False)
    with pytest.raises(ValueError, match="requires --global_train_shuffle"):
        validate_args(args, backend="cpu")


def test_explicit_adamw_recipe_is_available_to_launcher():
    with _Argv(
        "--output_dir",
        "run",
        "--tokenizer_vocab_path",
        "vocab.json",
        "--optim",
        "adamw_torch",
        "--adam_beta1",
        "0.9",
        "--adam_beta2",
        "0.999",
        "--adam_epsilon",
        "1e-8",
    ):
        args = parse_args()
    assert (args.optim, args.adam_beta1, args.adam_beta2, args.adam_epsilon) == (
        "adamw_torch",
        0.9,
        0.999,
        1e-8,
    )


def test_frozen_validation_row_ids_requires_validation_split():
    args = argparse.Namespace(
        train_order_path=None,
        validation_row_ids_path=Path("ids.npy"),
        use_validation_split=False,
    )
    with pytest.raises(ValueError, match="requires --use_validation_split"):
        validate_args(args, backend="cpu")


def test_explicit_train_file_also_hashes_validation_file(tmp_path: Path, monkeypatch):
    train = tmp_path / "train.parquet"
    valid = tmp_path / "valid.parquet"
    train.write_bytes(b"train")
    valid.write_bytes(b"valid")
    args = argparse.Namespace(
        dataset_name=str(tmp_path),
        data_files=str(train),
        train_split="train",
        validation_split="valid",
        use_validation_split=True,
        train_order_path=None,
        validation_row_ids_path=None,
    )
    monkeypatch.setattr("modernmolbert.train_selfies_ape_modernbert._MODERNBERT_BASE_CONFIG", train)
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.file_sha256", lambda path: path.name
    )
    hashes = _run_input_hashes(args, train, valid)
    assert hashes["validation_parquet"] == "valid.parquet"


def test_frozen_order_uses_original_parquet_row_ids(tmp_path: Path):
    train = tmp_path / "train.parquet"
    Dataset.from_dict({"selfies": ["row_0", "row_1", "row_2"]}).to_parquet(str(train))
    order = tmp_path / "order.npy"
    np.save(order, np.array([2, 0, 1], dtype=np.int64))

    class Tokenizer:
        def __call__(self, sequence: str, **_kwargs):
            return {"input_ids": [0, int(sequence[-1]) + 5, 2]}

    args = argparse.Namespace(
        data_dir=None,
        data_files=str(train),
        dataset_name=str(tmp_path),
        train_split="train",
        molecule_column="selfies",
        output_dir=str(tmp_path / "run"),
        train_order_path=order,
        global_train_shuffle=True,
        use_validation_split=True,
        max_seq_length=16,
        seed=42,
    )
    rows = list(make_train_iterable_dataset(args, Tokenizer()))  # type: ignore[arg-type]
    assert [row["input_ids"] for row in rows] == [[0, 7, 2], [0, 5, 2], [0, 6, 2]]


def test_pretokenized_rows_use_stable_hash_split(monkeypatch):
    rows = [
        {"input_ids": [0, 5, 2]},
        {"input_ids": [0, 6, 2]},
    ]
    validation_bucket = sequence_bucket("0,5,2", 100)
    args = argparse.Namespace(
        dataset_name="pretok",
        train_split="train",
        validation_split=None,
        use_validation_split=False,
        molecule_column="SELFIES",
        data_dir=None,
        data_files=None,
        seed=13,
        shuffle_buffer_size=100,
        max_seq_length=8,
        eval_size=4,
        max_eval_batches=0,
        per_device_eval_batch_size=2,
        val_split_mod=100,
        val_split_bucket=validation_bucket,
    )

    def _fake_stream(*args, **kwargs):
        return Dataset.from_list(rows).to_iterable_dataset()

    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.get_streaming_dataset",
        _fake_stream,
    )

    train_rows = list(make_train_iterable_dataset(args, tokenizer=None))  # type: ignore[arg-type]
    eval_dataset = make_eval_dataset(args, tokenizer=None)  # type: ignore[arg-type]

    assert [row["input_ids"] for row in train_rows] == [[0, 6, 2]]
    assert eval_dataset["input_ids"] == [[0, 5, 2]]


def test_explicit_train_parquet_keeps_validation_population(tmp_path: Path) -> None:
    Dataset.from_dict({"SELFIES": ["row_1", "row_2"]}).to_parquet(str(tmp_path / "train.parquet"))
    Dataset.from_dict({"SELFIES": ["row_9", "row_10"]}).to_parquet(str(tmp_path / "valid.parquet"))

    class Tokenizer:
        def __call__(self, sequence: str, **_kwargs):
            return {"input_ids": [0, int(sequence.rsplit("_", 1)[1]) + 5, 2]}

    args = argparse.Namespace(
        dataset_name=str(tmp_path),
        train_split="train",
        validation_split="valid",
        use_validation_split=True,
        molecule_column="SELFIES",
        data_dir=None,
        data_files=str(tmp_path / "train.parquet"),
        seed=42,
        shuffle_buffer_size=10,
        max_seq_length=16,
        eval_size=2,
        max_eval_batches=0,
        per_device_eval_batch_size=2,
    )

    validation = make_eval_dataset(args, Tokenizer())  # type: ignore[arg-type]
    assert sorted(ids[1] for ids in validation["input_ids"]) == [14, 15]


def test_training_encoding_rejects_over_context_molecule():
    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [0, 5, 6, 7, 2]}

    with pytest.raises(ValueError, match="exceeding context 4"):
        _encode_without_truncation(Tokenizer(), "CCO", 4)  # type: ignore[arg-type]


def test_resume_requires_identical_run_and_complete_checkpoint(tmp_path: Path, monkeypatch):
    revision = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.get_git_revision", lambda: revision.copy()
    )
    tokenizer = tmp_path / "tokenizer.json"
    metadata = tmp_path / "tokenizer.metadata.json"
    tokenizer.write_text("original")
    metadata.write_text("metadata")
    output = tmp_path / "run"
    args = argparse.Namespace(
        output_dir=str(output),
        resume_from_checkpoint=None,
        data_files=None,
        dataset_name="remote-dataset",
        train_split="train",
        use_validation_split=False,
        seed=42,
        max_steps=30_000,
        require_clean_git=True,
    )
    assert prepare_run_directory(args, tokenizer, metadata) is None
    original = (output / "run_identity.json").read_bytes()
    assert not (output / "run_args.json").exists()

    checkpoint = output / "checkpoint-5000"
    checkpoint.mkdir()
    args.resume_from_checkpoint = checkpoint
    with pytest.raises(ValueError, match="Incomplete resume checkpoint"):
        prepare_run_directory(args, tokenizer, metadata)

    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / filename).write_text("state")
    assert prepare_run_directory(args, tokenizer, metadata) == checkpoint
    assert (output / "run_identity.json").read_bytes() == original

    revision["commit"] = "b" * 40
    with pytest.raises(ValueError, match="code revision differ"):
        prepare_run_directory(args, tokenizer, metadata)
    revision["commit"] = "a" * 40

    tokenizer.write_text("changed")
    with pytest.raises(ValueError, match="inputs, or code revision differ"):
        prepare_run_directory(args, tokenizer, metadata)


def test_campaign_run_identity_uses_shared_manifest(tmp_path: Path, monkeypatch):
    revision = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.get_git_revision", lambda: revision
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer")
    metadata = tmp_path / "tokenizer.metadata.json"
    metadata.write_text("metadata", encoding="utf-8")
    from modernmolbert.utils import file_sha256

    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "code_commit": revision["commit"],
                "frozen_files_sha256": {str(tokenizer): file_sha256(tokenizer)},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        output_dir=str(tmp_path / "run"),
        resume_from_checkpoint=None,
        campaign_manifest=manifest,
        require_clean_git=True,
    )
    assert prepare_run_directory(args, tokenizer, metadata) is None
    identity = json.loads((tmp_path / "run/run_identity.json").read_text(encoding="utf-8"))
    assert identity["inputs"] == {"campaign_manifest_sha256": file_sha256(manifest)}
    assert "input_sha256" not in identity


def test_final_run_result_is_appended_without_changing_resume_identity(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.load_tokenizer_metadata",
        lambda _path: {"tokenizer_sha256": "b" * 64},
    )
    run = tmp_path / "run"
    model = run / "final_model"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"weights")
    (run / "run_identity.json").write_text(
        json.dumps({"schema": 2, "args": {"max_steps": 30_000}, "git": {"commit": "a"}}),
        encoding="utf-8",
    )
    finalize_run_identity(
        args=argparse.Namespace(output_dir=str(run), load_best_model_at_end=False),
        backend="cuda",
        vocab_size=600,
        special_ids={"bos_token": 0},
        n_params=1,
        tokenizer_stats={},
        tokenizer_metadata_path=tmp_path / "metadata.json",
        final_model_dir=model,
        selected_step=30_000,
        final_eval_metrics={"eval_loss": 2.0},
        trainer_state={"global_step": 30_000},
    )
    identity = json.loads((run / "run_identity.json").read_text(encoding="utf-8"))
    assert identity["args"] == {"max_steps": 30_000}
    assert identity["result"]["terminal_step"] == 30_000
    assert identity["result"]["tokenizer_sha256"] == "b" * 64
    assert not (run / "run_metadata.json").exists()


def test_run_argument_reader_supports_new_and_legacy_records(tmp_path: Path):
    (tmp_path / "run_args.json").write_text('{"seed": 1}', encoding="utf-8")
    assert load_run_args(tmp_path) == {"seed": 1}
    (tmp_path / "run_identity.json").write_text(
        json.dumps({"schema": 2, "args": {"seed": 42}}), encoding="utf-8"
    )
    assert load_run_args(tmp_path) == {"seed": 42}


def test_clean_git_requirement_rejects_dirty_checkout(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "modernmolbert.train_selfies_ape_modernbert.get_git_revision",
        lambda: {"commit": "a" * 40, "dirty": True},
    )
    args = argparse.Namespace(output_dir=str(tmp_path / "run"), require_clean_git=True)
    with pytest.raises(ValueError, match="clean Git checkout"):
        prepare_run_directory(args, tmp_path / "vocab.json", tmp_path / "metadata.json")


def test_masked_accuracy_reduces_logits_before_accumulation():
    logits = torch.tensor([[[0.0, 3.0], [4.0, 0.0]]])
    labels = torch.tensor([[1, -100]])
    predictions = preprocess_logits_for_metrics(logits, labels)
    assert predictions.shape == labels.shape
    assert compute_metrics((predictions.numpy(), labels.numpy())) == {"masked_accuracy": 1.0}


def test_hf_login_checks_both_tokens(monkeypatch):
    from modernmolbert.hf_upload import resolve_hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_ORG", raising=False)
    assert resolve_hf_token(hf_login=False) is None

    monkeypatch.setenv("HF_TOKEN_ORG", "org_tok")
    assert resolve_hf_token(hf_login=False) == "org_tok"

    monkeypatch.setenv("HF_TOKEN", "user_tok")
    # HF_TOKEN_ORG takes precedence
    assert resolve_hf_token(hf_login=False) == "org_tok"

    monkeypatch.delenv("HF_TOKEN_ORG", raising=False)
    assert resolve_hf_token(hf_login=False) == "user_tok"


def _default_args(**overrides):
    with _Argv("--output_dir", "run", "--tokenizer_vocab_path", "vocab.json"):
        args = parse_args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_validate_args_accepts_defaults():
    validate_args(_default_args(), backend="cpu")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mlm_probability": 1.5}, "mlm_probability"),
        ({"bf16": True, "fp16": True}, "mutually exclusive"),
        ({"eval_size": 0}, "eval_size"),
        ({"per_device_train_batch_size": 0}, "batch sizes"),
        ({"val_split_bucket": 100}, "val_split_bucket"),
        ({"load_best_model_at_end": True, "save_steps": 100, "eval_steps": 50}, "save_steps"),
        ({"masking_strategy": "unknown"}, "Unknown masking_strategy"),
        ({"masking_strategy": "span", "span_p_geom": 1.0}, "span_p_geom"),
    ],
)
def test_validate_args_rejects_invalid_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_args(_default_args(**overrides), backend="cpu")
