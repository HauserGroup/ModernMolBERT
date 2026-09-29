import argparse
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models import score
from modernmolbert.eval.benchmarking_molecular_models.common.types import EmbeddedDataset


def make_dataset_info(name: str, metric: str = "roc_auc"):
    return SimpleNamespace(
        name=name,
        metric=metric,
        task="classification",
    )


def make_embedded_dataset() -> EmbeddedDataset:
    return EmbeddedDataset(
        name="toy",
        task="classification",
        embedder="base_embedder",
        splits={
            "train": [0, 1, 2, 3, 4, 5],
            "valid": [6, 7],
            "test": [8, 9, 10, 11],
        },
        X=np.arange(24).reshape(12, 2),
        y=pd.DataFrame({"label": np.arange(12)}),
    )


def test_resolve_subsample_config_reads_score_yaml_values() -> None:
    args = argparse.Namespace(
        subsample_size=None,
        subsample_scope=None,
        subsample_seed=None,
    )

    subsample = score.resolve_subsample_config(
        {"subsample": 64, "subsample_scope": "all", "subsample_seed": 7},
        args,
    )

    assert subsample == score.SubsampleConfig(max_samples=64, scope="all", seed=7)


def test_make_scoring_model_name_adds_subsample_identity() -> None:
    name = score.make_scoring_model_name(
        "modernmolbert_best_span",
        score.SubsampleConfig(max_samples=512, scope="train", seed=13),
    )

    assert name == "modernmolbert_best_span__subsample_train512_seed13"


def test_subsample_embedded_dataset_train_scope_keeps_full_test_split() -> None:
    embedded = make_embedded_dataset()
    embedded.metadata["source_row_indices"] = list(range(100, 112))

    subset = score.subsample_embedded_dataset(
        embedded,
        subsample=score.SubsampleConfig(max_samples=4, scope="train", seed=13),
        embedder_name="base__subsample_train4_seed13",
    )

    assert subset.embedder == "base__subsample_train4_seed13"
    assert subset.X.shape == (8, 2)
    assert len(subset.splits["train"]) + len(subset.splits["valid"]) == 4
    assert len(subset.splits["test"]) == 4
    assert subset.y.iloc[subset.splits["test"]]["label"].tolist() == [8, 9, 10, 11]
    assert embedded.X.shape == (12, 2)
    assert subset.metadata["source_row_indices"][subset.splits["test"][0]] == 108
    assert subset.metadata["source_row_indices"][subset.splits["test"][-1]] == 111


def test_subsample_embedded_dataset_all_scope_samples_test_too() -> None:
    embedded = make_embedded_dataset()

    subset = score.subsample_embedded_dataset(
        embedded,
        subsample=score.SubsampleConfig(max_samples=6, scope="all", seed=13),
        embedder_name="base__subsample_all6_seed13",
    )

    assert subset.X.shape == (6, 2)
    assert sum(len(indices) for indices in subset.splits.values()) == 6
    assert 0 < len(subset.splits["test"]) < len(embedded.splits["test"])


def test_build_run_plan_skips_requested_datasets() -> None:
    items = [
        score.DatasetItem(
            config_name="clf_AMES",
            name="AMES",
            info=make_dataset_info("AMES"),
        ),
        score.DatasetItem(
            config_name="clf_ogbg-molhiv",
            name="ogbg-molhiv",
            info=make_dataset_info("ogbg-molhiv"),
        ),
        score.DatasetItem(
            config_name="clf_ogbg-molmuv",
            name="ogbg-molmuv",
            info=make_dataset_info("ogbg-molmuv"),
        ),
    ]

    run_items, skipped_items = score.build_run_plan(
        items=items,
        skip_set={"ogbg-molhiv", "ogbg-molmuv"},
    )

    assert [item.name for item in run_items] == ["AMES"]
    assert [(item.name, item.reason) for item in skipped_items] == [
        ("ogbg-molhiv", "requested skip"),
        ("ogbg-molmuv", "requested skip"),
    ]


def test_run_eval_returns_true_on_success(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_eval_procedure(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(score, "eval_procedure", fake_eval_procedure)

    embed_config = cast(
        score.EmbeddingConfig,
        SimpleNamespace(
            embedded_directory=tmp_path / "embedded",
            predictions_directory=tmp_path / "predictions",
        ),
    )

    ok = score.run_eval(
        safe=True,
        embed_config=embed_config,
        full_model_name="runs/modernmolbert_best_span",
        short_model_name="modernmolbert_best_span",
        dataset_info=make_dataset_info("AMES"),
        model_head="rf",
        output_csv=tmp_path / "results.csv",
        override=False,
    )

    assert ok is True
    assert len(calls) == 1
    assert calls[0]["model_name"] == "modernmolbert_best_span"
    assert calls[0]["model_head"] == "rf"


def test_run_eval_returns_false_on_failure_in_safe_mode(monkeypatch, tmp_path) -> None:
    def fake_eval_procedure(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(score, "eval_procedure", fake_eval_procedure)

    embed_config = cast(
        score.EmbeddingConfig,
        SimpleNamespace(
            embedded_directory=tmp_path / "embedded",
            predictions_directory=tmp_path / "predictions",
        ),
    )

    ok = score.run_eval(
        safe=True,
        embed_config=embed_config,
        full_model_name="runs/modernmolbert_best_span",
        short_model_name="modernmolbert_best_span",
        dataset_info=make_dataset_info("AMES"),
        model_head="rf",
        output_csv=tmp_path / "results.csv",
        override=False,
    )

    assert ok is False
