import json
import math
from pathlib import Path

import pytest

from modernmolbert.select_pretraining_run import (
    best_eval_from_log_history,
    copy_best_model,
    discover_runs,
    flatten_scalar_dict,
    summarize_run,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _make_complete_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "run_args.json",
        {
            "max_steps": 10,
            "model_size": "tiny",
            "mlm_probability": 0.15,
            "masking_strategy": "span",
            "learning_rate": 0.001,
            "tokenizer_vocab_path": "tokenizer.json",
        },
    )
    _write_json(
        run_dir / "trainer_state.json",
        {
            "global_step": 10,
            "log_history": [
                {"step": 2, "eval_loss": 0.9},
                {"step": 4, "eval_loss": None},
                {"step": 6, "eval_loss": "not-a-number"},
                {"step": 8, "eval_loss": 0.4},
            ],
        },
    )
    _write_json(run_dir / "eval_results.json", {"eval_loss": 0.5, "eval_perplexity": 1.7})
    _write_json(run_dir / "train_results.json", {"train_loss": 0.6})
    _write_json(
        run_dir / "ape_tokenizer_metadata.json",
        {"num_parameters": 1234, "trainer_state_summary": {"best_model_checkpoint": "ckpt-8"}},
    )
    final_model = run_dir / "final_model"
    final_model.mkdir()
    (final_model / "config.json").write_text("{}\n", encoding="utf-8")
    (final_model / "pytorch_model.bin").write_bytes(b"weights")


def test_flatten_scalar_dict_keeps_json_scalar_values_only() -> None:
    flattened = flatten_scalar_dict(
        {
            "eval_loss": 0.2,
            "eval_ok": True,
            "eval_name": "run",
            "eval_missing": None,
            "eval_nested": {"ignored": 1},
            "eval_list": [1, 2],
        },
    )

    assert flattened == {
        "eval_loss": 0.2,
        "eval_ok": True,
        "eval_name": "run",
        "eval_missing": None,
    }


def test_best_eval_from_log_history_ignores_unusable_values() -> None:
    trainer_state = {
        "log_history": [
            {"step": 1, "loss": 0.7},
            {"step": 2, "eval_loss": "bad"},
            {"step": 3, "eval_loss": None},
            {"step": 4, "eval_loss": 0.5},
            {"step": 5, "loss": 0.4},
        ]
    }

    assert best_eval_from_log_history(trainer_state, "loss", True) == (0.4, 5)
    assert best_eval_from_log_history(trainer_state, "loss", False) == (0.7, 1)
    assert best_eval_from_log_history(trainer_state, "accuracy", False) == (None, None)


def test_summarize_run_reports_completion_and_selection_metric(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    _make_complete_run(run_dir)

    summary = summarize_run(run_dir, metric="eval_loss", lower_is_better=True)

    assert summary["run_name"] == "run_a"
    assert summary["status"] == "complete"
    assert summary["has_final_model"] is True
    assert summary["completed_max_steps"] is True
    assert summary["best_metric"] == pytest.approx(0.4)
    assert summary["best_step_from_history"] == 8
    assert summary["selection_metric"] == pytest.approx(0.5)
    assert summary["num_parameters"] == 1234
    assert summary["metadata_best_checkpoint"] == "ckpt-8"


def test_summarize_run_marks_incomplete_and_nan_selection_metric(tmp_path) -> None:
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    _write_json(run_dir / "run_args.json", {"max_steps": 10})
    _write_json(run_dir / "trainer_state.json", {"global_step": 3, "log_history": []})

    summary = summarize_run(run_dir, metric="eval_loss", lower_is_better=True)

    assert summary["status"] == "incomplete"
    assert summary["has_final_model"] is False
    assert summary["completed_max_steps"] is False
    assert math.isnan(summary["selection_metric"])


def test_discover_runs_ignores_hidden_dirs_and_non_runs(tmp_path) -> None:
    _make_complete_run(tmp_path / "run_a")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "file.txt").write_text("notes\n", encoding="utf-8")
    run_b = tmp_path / "run_b"
    run_b.mkdir()
    _write_json(run_b / "trainer_state.json", {"global_step": 1})

    assert [path.name for path in discover_runs(tmp_path)] == ["run_a", "run_b"]


def test_copy_best_model_copies_final_model_and_refuses_existing_destination(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    _make_complete_run(run_dir)
    destination = tmp_path / "best_model"

    copy_best_model({"final_model": str(run_dir / "final_model")}, destination)

    assert (destination / "config.json").exists()
    assert (destination / "pytorch_model.bin").read_bytes() == b"weights"

    with pytest.raises(FileExistsError, match="Destination already exists"):
        copy_best_model({"final_model": str(run_dir / "final_model")}, destination)
