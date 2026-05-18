import types

import pandas as pd

from modernmolbert.eval.dabest_plots import (
    dabest_embedder_comparison,
    dabest_model_comparison,
)


class _FakeFigure:
    def __init__(self) -> None:
        self.saved_to = None

    def savefig(self, path, **kwargs) -> None:
        self.saved_to = path


class _FakeMeanDiff:
    def __init__(self, figure: _FakeFigure) -> None:
        self._figure = figure
        self.plot_kwargs = None

    def plot(self, **kwargs):
        self.plot_kwargs = kwargs
        return self._figure


class _FakeAnalysis:
    def __init__(self, figure: _FakeFigure) -> None:
        self.mean_diff = _FakeMeanDiff(figure)


def _install_fake_dabest(monkeypatch):
    captured = {}
    figure = _FakeFigure()

    def fake_load(**kwargs):
        captured.update(kwargs)
        return _FakeAnalysis(figure)

    monkeypatch.setitem(
        __import__("sys").modules,
        "dabest",
        types.SimpleNamespace(load=fake_load),
    )
    return captured, figure


def _row(dataset: str, embedder: str, model: str, metric: float) -> dict:
    return {
        "id": 1,
        "dataset": dataset,
        "task": "classification",
        "embedder": embedder,
        "model": model,
        "hyperparams": "{}",
        "library_hash": "test",
        "cv_metric_name": "roc_auc",
        "cv_metric": metric,
        "test_metric_name": "roc_auc",
        "test_metric": metric,
        "key": f"{dataset}_{embedder}_{model}",
    }


def test_dabest_model_comparison_reads_praski_csv_and_pairs_by_dataset_embedder(
    monkeypatch,
    tmp_path,
) -> None:
    captured, _ = _install_fake_dabest(monkeypatch)
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            _row("AMES", "modern", "rf", 0.82),
            _row("AMES", "modern", "ridge", 0.81),
            _row("BBBP", "modern", "rf", 0.72),
            _row("BBBP", "modern", "ridge", 0.74),
            _row("incomplete", "modern", "rf", 0.50),
        ]
    ).to_csv(csv_path, index=False)

    dabest_model_comparison(
        csv_path,
        control_model="rf",
        comparison_models=["ridge"],
        metric_name="roc_auc",
    )

    assert captured["x"] == "model"
    assert captured["y"] == "test_metric"
    assert captured["idx"] == ("rf", "ridge")
    assert captured["paired"] == "baseline"
    assert captured["id_col"] == "__pair_id__"
    assert set(captured["data"]["__pair_id__"]) == {"AMES__modern", "BBBP__modern"}


def test_dabest_embedder_comparison_pairs_by_dataset_and_head(monkeypatch) -> None:
    captured, _ = _install_fake_dabest(monkeypatch)
    frame = pd.DataFrame(
        [
            _row("AMES", "modern", "rf", 0.82),
            _row("AMES", "ecfp", "rf", 0.80),
            _row("AMES", "modern", "ridge", 0.81),
            _row("AMES", "ecfp", "ridge", 0.79),
            _row("BBBP", "modern", "rf", 0.72),
            _row("BBBP", "ecfp", "rf", 0.70),
        ]
    )

    dabest_embedder_comparison(
        frame,
        control_embedder="ecfp",
        comparison_embedders=["modern"],
        metric_name="roc_auc",
        models=["rf"],
    )

    assert captured["x"] == "embedder"
    assert captured["idx"] == ("ecfp", "modern")
    assert set(captured["data"]["model"]) == {"rf"}
    assert set(captured["data"]["__pair_id__"]) == {"AMES__rf", "BBBP__rf"}
