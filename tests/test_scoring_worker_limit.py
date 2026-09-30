import logging

import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.supervised.train import fit_model


def test_explicit_worker_limit_controls_cross_validation(caplog):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 12)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    with caplog.at_level(logging.INFO):
        result = fit_model(
            X=X,
            y=y,
            task="classification",
            model_head="ridge",
            memory_weight=1,
            n_jobs=2,
            missing_labels="as-negative",
        )
    assert "GridSearchCV n_jobs=2 (outer=2, head=ridge)" in caplog.text
    assert np.isfinite(result["best_score"])
