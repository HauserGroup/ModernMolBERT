import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "collect_sweep_results.py"
    spec = importlib.util.spec_from_file_location("collect_sweep_results", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_adds_metric_provenance_columns(tmp_path: Path) -> None:
    collect_sweep_results = _load_module()
    run_dir = tmp_path / "mask_standard__mlm_0p15__lr_1e-4"
    run_dir.mkdir()

    _write_json(
        run_dir / "run_args.json",
        {
            "model_size": "small",
            "masking_strategy": "standard",
            "mlm_probability": 0.15,
            "learning_rate": 1e-4,
            "load_best_model_at_end": True,
        },
    )
    _write_json(
        run_dir / "all_results.json",
        {
            "eval_loss": 0.4,
            "eval_masked_accuracy": 0.8,
            "eval_perplexity": 1.49,
            "train_loss": 0.5,
        },
    )
    _write_json(
        run_dir / "trainer_state.json",
        {
            "best_model_checkpoint": "runs/sweep/mask_standard__mlm_0p15__lr_1e-4/checkpoint-20000",
            "log_history": [
                {"step": 10000, "eval_loss": 0.6, "eval_masked_accuracy": 0.7},
                {"step": 20000, "eval_loss": 0.35, "eval_masked_accuracy": 0.82},
                {"step": 30000, "eval_loss": 0.42, "eval_masked_accuracy": 0.79},
            ],
        },
    )

    rows = collect_sweep_results.collect(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["metric_source"] == "best_model_final_eval"
    assert row["load_best_model_at_end"] is True
    assert row["eval_masking_strategy"] == "standard"
    assert row["eval_mlm_probability"] == 0.15
    assert row["best_step"] == 20000
    assert row["best_logged_eval_loss"] == 0.35
    assert row["last_logged_eval_step"] == 30000
    assert row["last_logged_eval_loss"] == 0.42
    assert row["final_eval_loss"] == 0.4
    assert row["eval_loss"] == 0.4


def test_collect_handles_missing_trainer_state(tmp_path: Path) -> None:
    collect_sweep_results = _load_module()
    run_dir = tmp_path / "mask_span__mlm_0p20__lr_2e-4"
    run_dir.mkdir()

    _write_json(run_dir / "run_args.json", {"model_size": "small"})
    _write_json(run_dir / "all_results.json", {"eval_loss": 0.3})

    rows = collect_sweep_results.collect(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["metric_source"] == "final_model_eval"
    assert row["best_checkpoint"] == ""
    assert row["best_step"] == ""
    assert row["best_logged_eval_loss"] == ""
    assert row["last_logged_eval_loss"] == ""
    assert row["final_eval_loss"] == 0.3
