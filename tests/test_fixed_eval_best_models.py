import importlib.util
import json
from pathlib import Path

from modernmolbert.collator import MolecularMLMCollator


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "analysis" / "sweep" / "fixed_eval_best_models.py"
    spec = importlib.util.spec_from_file_location("fixed_eval_best_models", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_fixed_masked_dataset_is_reproducible_and_valid() -> None:
    m = _load_module()
    examples = [
        {"input_ids": [0, 5, 6, 7, 2, 1, 1, 1]},
        {"input_ids": [0, 8, 9, 10, 11, 12, 2, 1]},
        {"input_ids": [0, 13, 14, 15, 2, 1, 1, 1]},
    ]
    kwargs = {
        "examples": examples,
        "batch_size": 2,
        "seed": 123,
        "pad_token_id": 1,
        "mask_token_id": 4,
        "vocab_size": 32,
        "special_token_ids": [0, 1, 2, 3, 4],
        "ids_to_tokens": {i: f"[T{i}]" for i in range(32)},
        "mlm_probability": 0.5,
        "masking_strategy": "standard",
    }

    fixed_a = m.build_fixed_masked_dataset(**kwargs)
    fixed_b = m.build_fixed_masked_dataset(**kwargs)

    assert fixed_a["fingerprint_sha256"] == fixed_b["fingerprint_sha256"]
    assert fixed_a["masked_tokens"] > 0
    assert fixed_a["num_examples"] == len(examples)
    assert fixed_a["seq_length"] == 8
    assert 0.0 < fixed_a["actual_mask_fraction"] <= 1.0

    tensors = fixed_a["tensors"]
    labels = tensors["labels"]
    original = tensors["original_input_ids"]
    masked = labels.ne(-100)
    for special_id in [0, 1, 2, 3, 4]:
        assert not (masked & original.eq(special_id)).any()


def test_tensor_fingerprint_changes_when_labels_change() -> None:
    m = _load_module()
    fixed = m.build_fixed_masked_dataset(
        [{"input_ids": [0, 5, 6, 7, 2, 1, 1, 1]}],
        batch_size=1,
        seed=7,
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=32,
        special_token_ids=[0, 1, 2, 3, 4],
        ids_to_tokens={i: f"[T{i}]" for i in range(32)},
        mlm_probability=1.0,
        masking_strategy="standard",
    )
    tensors = fixed["tensors"]
    original_fingerprint = m.tensor_fingerprint(tensors, ["input_ids", "labels"])
    tensors["labels"][0, 1] = 99

    assert m.tensor_fingerprint(tensors, ["input_ids", "labels"]) != original_fingerprint


def test_script_uses_repo_collator_class() -> None:
    m = _load_module()

    assert m.MolecularMLMCollator is MolecularMLMCollator
