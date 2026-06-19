import json
from pathlib import Path
from argparse import Namespace

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, ModernBertConfig

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.train_selfies_ape_modernbert import write_run_metadata
from modernmolbert.utils import (
    copy_tokenizer_metadata_from_anywhere,
    copy_tokenizer_artifacts,
    file_sha256,
    write_tokenizer_metadata,
)


def _tiny_tokenizer() -> APEPreTrainedTokenizer:
    tok = APEPreTrainedTokenizer()
    tok.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
    }
    tok.special_tokens = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
    }
    tok.update_reverse_vocabulary()
    return tok


def test_end_to_end_save_and_reload_with_tokenizer_artifacts(tmp_path: Path):
    tokenizer = _tiny_tokenizer()

    vocab_path = tmp_path / "selfies_ape_tokenizer.json"
    tokenizer.save_vocabulary_file(vocab_path)
    metadata_path = tmp_path / "selfies_ape_tokenizer.metadata.json"
    write_tokenizer_metadata(
        metadata_path,
        {
            "representation": "SELFIES",
            "tokenizer_sha256": file_sha256(vocab_path),
            "tokenizer_path": str(vocab_path),
        },
    )

    config = ModernBertConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = AutoModelForMaskedLM.from_config(config)

    batch = tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
    batch = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    out = model(**batch, labels=labels)
    out.loss.backward()

    output_dir = tmp_path / "run"
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_model_dir))

    copy_tokenizer_artifacts(vocab_path, metadata_path, output_dir, final_model_dir)

    for expected in [
        "vocab.json",
        "selfies_vocab.json",
        "tokenizer_metadata.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_ape.py",
        "ape_tokenizer/vocab.json",
        "ape_tokenizer/selfies_vocab.json",
        "ape_tokenizer/tokenizer_metadata.json",
        "ape_tokenizer/tokenizer_config.json",
        "ape_tokenizer/special_tokens_map.json",
        "ape_tokenizer/tokenization_ape.py",
    ]:
        assert (final_model_dir / expected).exists()

    root_metadata = [
        final_model_dir / "tokenizer_metadata.json",
        final_model_dir / "ape_tokenizer_metadata.json",
    ]
    nested_metadata = [
        final_model_dir / "ape_tokenizer" / "tokenizer_metadata.json",
        final_model_dir / "ape_tokenizer" / "ape_tokenizer_metadata.json",
    ]
    assert any(path.exists() for path in root_metadata)
    assert any(path.exists() for path in nested_metadata)
    assert not (final_model_dir / "tokenizer.json").exists()

    reloaded_model = AutoModelForMaskedLM.from_pretrained(str(final_model_dir))
    reloaded_tokenizer = APEPreTrainedTokenizer.from_pretrained(str(final_model_dir))
    reloaded_subdir_tokenizer = AutoTokenizer.from_pretrained(
        str(final_model_dir / "ape_tokenizer"),
        trust_remote_code=True,
    )

    eval_batch = reloaded_tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
    eval_batch = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in eval_batch.items()}
    with torch.no_grad():
        logits = reloaded_model(**eval_batch).logits

    assert torch.isfinite(logits).all()
    assert logits.shape[0] == 1
    assert logits.shape[1] == eval_batch["input_ids"].shape[1]

    auto_batch = reloaded_tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
    assert auto_batch["input_ids"].shape == eval_batch["input_ids"].shape
    subdir_batch = reloaded_subdir_tokenizer("[C][C][O]", add_special_tokens=True)
    assert subdir_batch["input_ids"] == auto_batch["input_ids"].squeeze(0).tolist()


def test_write_run_metadata_writes_hub_model_card(tmp_path: Path):
    metadata_path = tmp_path / "selfies_ape_tokenizer.metadata.json"
    write_tokenizer_metadata(
        metadata_path,
        {
            "representation": "SELFIES",
            "tokenizer_sha256": "abc123",
            "tokenizer_path": "tokenizer/selfies_ape_tokenizer.json",
        },
    )
    output_dir = tmp_path / "run"
    args = Namespace(
        output_dir=str(output_dir),
        dataset_name="data/pretrain/chembl36_selfies",
        selfies_column="selfies",
        representation="SELFIES",
        train_split="train",
        validation_split=None,
        use_validation_split=False,
        max_seq_length=256,
        mlm_probability=0.3,
        masking_strategy="span",
        model_size="small",
    )

    write_run_metadata(
        args=args,
        backend="cpu",
        vocab_size=8,
        special_ids={
            "pad_token": 1,
            "bos_token": 0,
            "eos_token": 2,
            "unk_token": 3,
            "mask_token": 4,
        },
        n_params=1234,
        tokenizer_stats={"unk_rate": 0.0},
        tokenizer_vocab_path=tmp_path / "selfies_ape_tokenizer.json",
        tokenizer_metadata_path=metadata_path,
        final_eval_metrics={"eval_loss": 1.5},
        trainer_state={
            "best_model_checkpoint": "checkpoint-10",
            "best_metric": 1.5,
            "best_global_step": 10,
        },
    )

    model_card = output_dir / "final_model" / "README.md"
    assert model_card.exists()
    text = model_card.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "library_name: transformers" in text
    assert "pipeline_tag: fill-mask" in text
    assert "SELFIES strings only" in text
    assert "AutoTokenizer.from_pretrained" in text
    assert 'subfolder="ape_tokenizer"' in text
    assert "trust_remote_code=True" in text


def test_write_run_metadata_smiles_model_card(tmp_path: Path):
    metadata_path = tmp_path / "smiles_ape_tokenizer.metadata.json"
    write_tokenizer_metadata(
        metadata_path,
        {
            "representation": "SMILES",
            "tokenizer_sha256": "def456",
            "tokenizer_path": "tokenizer/smiles_ape_tokenizer.json",
        },
    )
    output_dir = tmp_path / "run"
    args = Namespace(
        output_dir=str(output_dir),
        dataset_name="data/pretrain/chembl36_selfies",
        selfies_column="smiles_canonical_clean",
        representation="SMILES",
        train_split="train",
        validation_split=None,
        use_validation_split=False,
        max_seq_length=128,
        mlm_probability=0.3,
        masking_strategy="span",
        model_size="small",
    )

    write_run_metadata(
        args=args,
        backend="cpu",
        vocab_size=8,
        special_ids={
            "pad_token": 1,
            "bos_token": 0,
            "eos_token": 2,
            "unk_token": 3,
            "mask_token": 4,
        },
        n_params=1234,
        tokenizer_stats={"unk_rate": 0.0},
        tokenizer_vocab_path=tmp_path / "smiles_ape_tokenizer.json",
        tokenizer_metadata_path=metadata_path,
    )

    text = (output_dir / "final_model" / "README.md").read_text(encoding="utf-8")
    assert "SELFIES strings only" not in text
    assert "canonical SMILES strings" in text
    assert "`SMILES`" in text
    assert "smiles_vocab.json" in text

    metadata = json.loads((output_dir / "ape_tokenizer_metadata.json").read_text())
    assert metadata["representation"] == "SMILES"
    assert metadata["molecule_column"] == "smiles_canonical_clean"


def test_copy_tokenizer_metadata_from_anywhere_with_tokenizer_metadata_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target_root = tmp_path / "final_model"
    target_nested = target_root / "ape_tokenizer"
    source.mkdir(parents=True, exist_ok=True)
    (source / "tokenizer_metadata.json").write_text('{"representation": "SELFIES"}\n')

    copy_tokenizer_metadata_from_anywhere([source], target_root)
    copy_tokenizer_metadata_from_anywhere([source], target_nested)

    assert (target_root / "tokenizer_metadata.json").exists()
    assert (target_nested / "tokenizer_metadata.json").exists()


def test_copy_tokenizer_metadata_from_anywhere_with_ape_metadata_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target_root = tmp_path / "final_model"
    target_nested = target_root / "ape_tokenizer"
    source.mkdir(parents=True, exist_ok=True)
    (source / "ape_tokenizer_metadata.json").write_text('{"representation": "SELFIES"}\n')

    copy_tokenizer_metadata_from_anywhere([source], target_root)
    copy_tokenizer_metadata_from_anywhere([source], target_nested)

    assert (target_root / "ape_tokenizer_metadata.json").exists()
    assert (target_nested / "ape_tokenizer_metadata.json").exists()


def test_copy_tokenizer_metadata_from_anywhere_warns_when_missing(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir(parents=True, exist_ok=True)

    copy_tokenizer_metadata_from_anywhere([source], target)

    captured = capsys.readouterr()
    assert "WARNING: no tokenizer metadata file found in any source directory" in captured.out
