import numpy as np
import pytest

from modernmolbert.eval.benchmarking_molecular_models.common.types import HeadResult
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    log_predictions,
)


def test_prediction_archive_contains_prepared_source_row_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = HeadResult(
        embedder="encoder",
        dataset_name="assay",
        y_test_true=np.array([0, 1]),
        y_test_pred=np.array([[0.9, 0.1], [0.2, 0.8]]),
        model="ridge",
        hyperparams={},
        cv_score=0.8,
        test_source_row_indices=np.array([4, 7]),
        prepared_data_sha256="a" * 64,
    )
    log_predictions(result, "predictions")
    path = tmp_path / "predictions/assay/encoder/ridge.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert archive["test_source_row_indices"].tolist() == [4, 7]
        assert archive["prepared_data_sha256"].item() == "a" * 64
        assert archive["y_true"].tolist() == [0, 1]


def test_prediction_archive_rejects_row_mapping_length_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = HeadResult(
        embedder="encoder",
        dataset_name="assay",
        y_test_true=np.array([0, 1]),
        y_test_pred=np.array([[0.9, 0.1], [0.2, 0.8]]),
        model="ridge",
        hyperparams={},
        cv_score=0.8,
        test_source_row_indices=np.array([4]),
    )
    with pytest.raises(ValueError, match="must match prediction rows"):
        log_predictions(result, "predictions")
    assert not (tmp_path / "predictions/assay/encoder/ridge.npy").exists()
