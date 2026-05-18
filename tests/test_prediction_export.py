from pathlib import Path

import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.praski_export import PRASKI_COLUMNS
from modernmolbert.eval.benchmarking_molecular_models.prediction_export import (
    prediction_artifacts_to_praski_frame,
    write_prediction_praski_csv,
)


def _write_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    (config_dir / "datasets.yaml").write_text(
        "\n".join(
            [
                "datasets:",
                "  clf_tiny:",
                "    name: tiny",
                "    metric: roc_auc",
                "    task: classification",
                "    source:",
                "      name: test",
            ]
        )
    )


def test_prediction_artifacts_to_praski_frame_scores_npz_predictions(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir)
    predictions_dir = tmp_path / "predictions"
    artifact_dir = predictions_dir / "tiny" / "fake_embedder"
    artifact_dir.mkdir(parents=True)
    np.savez(
        artifact_dir / "rf.npz",
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.2, 0.8, 0.9]),
    )

    frame = prediction_artifacts_to_praski_frame(
        predictions_dir,
        config_dir=config_dir,
        library_hash="hash",
    )

    assert list(frame.columns) == PRASKI_COLUMNS
    assert frame.loc[0, "dataset"] == "tiny"
    assert frame.loc[0, "task"] == "classification"
    assert frame.loc[0, "embedder"] == "fake_embedder"
    assert frame.loc[0, "model"] == "rf"
    assert frame.loc[0, "library_hash"] == "hash"
    assert frame.loc[0, "cv_metric_name"] == "roc_auc"
    assert frame.loc[0, "test_metric_name"] == "roc_auc"
    assert frame.loc[0, "test_metric"] == 1.0
    assert frame.loc[0, "key"] == "tiny_fake_embedder_rf"


def test_write_prediction_praski_csv_ignores_legacy_npy_only_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir)
    predictions_dir = tmp_path / "predictions"
    artifact_dir = predictions_dir / "tiny" / "fake_embedder"
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "rf.npy", np.array([0.1, 0.9]))

    output_csv = tmp_path / "results.csv"
    frame = write_prediction_praski_csv(
        predictions_dir,
        output_csv,
        config_dir=config_dir,
    )

    assert frame.empty
    assert output_csv.read_text().splitlines()[0] == ",".join(PRASKI_COLUMNS)
