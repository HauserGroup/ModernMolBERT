import types
from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.eval.dabest_plots import (
    dabest_embedder_comparison,
    dabest_model_comparison,
)

PRASKI_CSV = (
    Path(__file__).parent.parent / "data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv"
)
praski_csv_present = pytest.mark.skipif(
    not PRASKI_CSV.exists(), reason=f"Praski CSV not found: {PRASKI_CSV}"
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
        types.SimpleNamespace(load=fake_load, __version__="0.2.5"),
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
    assert captured["paired"] is True
    assert captured["id_col"] == "__pair_id__"
    assert set(captured["data"].columns) == {"model", "test_metric", "__pair_id__"}
    assert captured["data"]["__pair_id__"].nunique() == 2


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
    assert set(captured["data"].columns) == {"embedder", "test_metric", "__pair_id__"}
    assert captured["data"]["__pair_id__"].nunique() == 2


# ── integration tests against the real Praski preprint CSV ───────────────────


@praski_csv_present
def test_praski_csv_loads_with_correct_schema() -> None:
    df = pd.read_csv(PRASKI_CSV)
    required = {"id", "dataset", "task", "embedder", "model", "test_metric", "test_metric_name"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    assert not df.empty
    assert set(df["model"].unique()) == {"rf", "ridge", "knn"}
    assert df["task"].eq("classification").all()


@praski_csv_present
def test_model_comparison_pairs_on_real_data(monkeypatch) -> None:
    captured, _ = _install_fake_dabest(monkeypatch)

    dabest_model_comparison(
        PRASKI_CSV,
        control_model="rf",
        comparison_models=["ridge"],
        metric_name="roc_auc",
        embedders=["CDDD", "ECFP"],
    )

    # 2 embedders × 25 datasets = 50 pairs
    assert captured["data"]["__pair_id__"].nunique() == 50
    assert captured["x"] == "model"
    assert captured["idx"] == ("rf", "ridge")


@praski_csv_present
def test_embedder_comparison_pairs_on_real_data(monkeypatch) -> None:
    captured, _ = _install_fake_dabest(monkeypatch)

    dabest_embedder_comparison(
        PRASKI_CSV,
        control_embedder="CDDD",
        comparison_embedders=["ECFP"],
        metric_name="roc_auc",
        models=["rf"],
    )

    # 25 datasets × 1 model = 25 pairs
    assert captured["data"]["__pair_id__"].nunique() == 25
    assert captured["x"] == "embedder"
    assert captured["idx"] == ("CDDD", "ECFP")


@praski_csv_present
def test_dataset_filter_reduces_pairs(monkeypatch) -> None:
    captured, _ = _install_fake_dabest(monkeypatch)
    datasets = ["AMES", "DILI", "hERG"]

    dabest_model_comparison(
        PRASKI_CSV,
        control_model="rf",
        comparison_models=["ridge"],
        metric_name="roc_auc",
        datasets=datasets,
    )

    df = pd.read_csv(PRASKI_CSV)
    n_embedders = df[df["dataset"].isin(datasets)]["embedder"].nunique()
    assert captured["data"]["__pair_id__"].nunique() == n_embedders * len(datasets)
