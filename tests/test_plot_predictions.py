from pathlib import Path

import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.plot_predictions import (
    make_plots,
)


def _write_npz(root: Path, dataset: str, embedder: str, head: str, y_true, y_score) -> Path:
    path = root / dataset / embedder / f"{head}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y_true=np.asarray(y_true), y_score=np.asarray(y_score))
    return path


def test_make_plots_writes_binary_and_multioutput_classification_figures(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    output = tmp_path / "plots"
    _write_npz(predictions, "BBBP", "ours", "knn", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    _write_npz(
        predictions,
        "TOX21",
        "ours",
        "rf",
        [[0, 1], [1, 0], [0, 1]],
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
    )

    saved = make_plots(predictions, output)

    assert saved == [output / "BBBP.png", output / "TOX21.png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in saved)
