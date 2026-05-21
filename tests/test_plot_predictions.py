from pathlib import Path

import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.plot_predictions import (
    _find_npz_files,
    make_plots,
)


def _write_npz(root: Path, dataset: str, embedder: str, head: str, y_true, y_score) -> Path:
    path = root / dataset / embedder / f"{head}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y_true=np.asarray(y_true), y_score=np.asarray(y_score))
    return path


def test_find_npz_files_filters_by_embedder_and_dataset(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    keep = _write_npz(predictions, "BBBP", "ours", "knn", [0, 1], [0.1, 0.9])
    _write_npz(predictions, "BBBP", "baseline", "knn", [0, 1], [0.2, 0.8])
    _write_npz(predictions, "ESOL", "ours", "ridge", [1.0, 2.0], [1.1, 1.9])
    np.savez(predictions / "flat.npz", y_true=np.array([0, 1]), y_score=np.array([0.1, 0.9]))

    found = _find_npz_files(predictions, embedder="ours", datasets=["BBBP"])

    assert found == {"BBBP": {"ours": {"knn": keep}}}


def test_make_plots_returns_empty_for_no_npz_files(tmp_path: Path, capsys) -> None:
    output = tmp_path / "plots"

    assert make_plots(tmp_path / "missing", output) == []
    assert "No .npz files found" in capsys.readouterr().out
    assert not output.exists()


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


def test_make_plots_writes_regression_figure(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    output = tmp_path / "plots"
    _write_npz(predictions, "ESOL", "ours", "ridge", [0.1, 0.5, 1.0], [0.2, 0.4, 1.1])

    saved = make_plots(predictions, output)

    assert saved == [output / "ESOL_regression.png"]
    assert saved[0].exists()
    assert saved[0].stat().st_size > 0
