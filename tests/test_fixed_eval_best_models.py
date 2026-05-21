import importlib.util
import json
from pathlib import Path

from modernmolbert.collator import MolecularMLMCollator


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "fixed_eval_best_models.py"
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


def test_resolve_selected_runs_reads_best_metadata(tmp_path: Path) -> None:
    m = _load_module()
    sweep = tmp_path / "sweep"
    run_dir = sweep / "mask_standard__mlm_0p15__lr_4e-4"
    ckpt_dir = run_dir / "checkpoint-30000"
    ckpt_dir.mkdir(parents=True)
    _write_json(
        run_dir / "run_args.json",
        {
            "masking_strategy": "standard",
            "mlm_probability": 0.15,
            "learning_rate": 4e-4,
        },
    )
    _write_json(
        sweep / "best_standard_run.json",
        {
            "run_name": "mask_standard__mlm_0p15__lr_4e-4",
            "best_model_checkpoint": str(ckpt_dir),
        },
    )
    _write_json(
        sweep / "best_span_run.json",
        {
            "run_name": "mask_standard__mlm_0p15__lr_4e-4",
            "best_model_checkpoint": str(ckpt_dir),
        },
    )

    selected = m.resolve_selected_runs(sweep)

    assert [run.label for run in selected] == ["standard", "span"]
    assert selected[0].run_name == "mask_standard__mlm_0p15__lr_4e-4"
    assert selected[0].run_dir == run_dir
    assert selected[0].trained_masking_strategy == "standard"


def test_script_uses_repo_collator_class() -> None:
    m = _load_module()

    assert m.MolecularMLMCollator is MolecularMLMCollator
