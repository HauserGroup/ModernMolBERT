import argparse
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models import score
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import EmbeddedDataset


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


def test_normalize_name_removes_clf_and_reg_prefixes() -> None:
    assert score.normalize_name("ogbg-molhiv") == "ogbg-molhiv"
    assert score.normalize_name("clf_ogbg-molhiv") == "ogbg-molhiv"
    assert score.normalize_name("reg_esol") == "esol"
    assert score.normalize_name("config/datasets/clf_ogbg-molmuv.yaml") == "ogbg-molmuv"


def test_make_short_model_name_from_local_path() -> None:
    assert score.make_short_model_name("runs/modernmolbert_best_span") == "modernmolbert_best_span"


def test_make_short_model_name_from_plain_name() -> None:
    assert score.make_short_model_name("modernmolbert_best_span") == "modernmolbert_best_span"


def test_load_dataset_items_uses_dataset_info_name_as_canonical(monkeypatch, tmp_path) -> None:
    def fake_expand_dataset_selection(config_dir, selections):
        return ["clf_ogbg-molhiv", "clf_ogbg-molbbbp"]

    def fake_load_dataset_config(config_dir, config_name):
        mapping = {
            "clf_ogbg-molhiv": "ogbg-molhiv",
            "clf_ogbg-molbbbp": "ogbg-molbbbp",
        }
        return make_dataset_info(mapping[config_name])

    monkeypatch.setattr(score, "expand_dataset_selection", fake_expand_dataset_selection)
    monkeypatch.setattr(score, "load_dataset_config", fake_load_dataset_config)

    items = score.load_dataset_items(
        config_dir=tmp_path,
        selections=["all"],
    )

    assert [item.config_name for item in items] == [
        "clf_ogbg-molhiv",
        "clf_ogbg-molbbbp",
    ]
    assert [item.name for item in items] == [
        "ogbg-molhiv",
        "ogbg-molbbbp",
    ]


def test_should_skip_item_matches_canonical_name() -> None:
    item = score.DatasetItem(
        config_name="clf_ogbg-molhiv",
        name="ogbg-molhiv",
        info=make_dataset_info("ogbg-molhiv"),
    )

    assert score.should_skip_item(item, {"ogbg-molhiv"}) is True


def test_should_skip_item_matches_normalized_config_name() -> None:
    item = score.DatasetItem(
        config_name="clf_ogbg-molhiv",
        name="ogbg-molhiv",
        info=make_dataset_info("ogbg-molhiv"),
    )

    skip_set = {score.normalize_name("clf_ogbg-molhiv")}

    assert score.should_skip_item(item, skip_set) is True


def test_resolve_skip_set_normalizes_cli_skip_names() -> None:
    args = argparse.Namespace(
        skip_datasets=["clf_ogbg-molhiv", "ogbg-molmuv.yaml"],
    )

    assert score.resolve_skip_set({}, args) == {
        "ogbg-molhiv",
        "ogbg-molmuv",
    }


def test_resolve_subsample_config_defaults_to_train_scope() -> None:
    args = argparse.Namespace(
        subsample_size=128,
        subsample_scope=None,
        subsample_seed=None,
    )

    subsample = score.resolve_subsample_config({}, args)

    assert subsample == score.SubsampleConfig(max_samples=128, scope="train", seed=13)


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


def test_should_skip_item_does_not_match_unrelated_name() -> None:
    item = score.DatasetItem(
        config_name="clf_ogbg-molbbbp",
        name="ogbg-molbbbp",
        info=make_dataset_info("ogbg-molbbbp"),
    )

    assert score.should_skip_item(item, {"ogbg-molhiv"}) is False


def test_dataset_checkpoint_path_uses_canonical_dataset_and_embedder(tmp_path) -> None:
    path = score.dataset_checkpoint_path(
        checkpoint_dir=tmp_path,
        dataset="ogbg-molhiv",
        embedder="modernmolbert_best_span",
    )

    assert path == tmp_path / "ogbg-molhiv__modernmolbert_best_span.csv"


def test_checkpoint_exists_false_when_checkpoint_dir_is_none() -> None:
    assert (
        score.checkpoint_exists(
            checkpoint_dir=None,
            dataset="AMES",
            embedder="modernmolbert_best_span",
        )
        is False
    )


def test_checkpoint_exists_true_for_nonempty_checkpoint(tmp_path) -> None:
    checkpoint = score.dataset_checkpoint_path(
        checkpoint_dir=tmp_path,
        dataset="AMES",
        embedder="modernmolbert_best_span",
    )
    checkpoint.write_text("dataset,embedder\nAMES,modernmolbert_best_span\n")

    assert (
        score.checkpoint_exists(
            checkpoint_dir=tmp_path,
            dataset="AMES",
            embedder="modernmolbert_best_span",
        )
        is True
    )


def test_checkpoint_exists_false_for_empty_checkpoint(tmp_path) -> None:
    checkpoint = score.dataset_checkpoint_path(
        checkpoint_dir=tmp_path,
        dataset="AMES",
        embedder="modernmolbert_best_span",
    )
    checkpoint.write_text("")

    assert (
        score.checkpoint_exists(
            checkpoint_dir=tmp_path,
            dataset="AMES",
            embedder="modernmolbert_best_span",
        )
        is False
    )


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
        checkpoint_dir=None,
        embedder="modernmolbert_best_span",
        resume=True,
    )

    assert [item.name for item in run_items] == ["AMES"]
    assert [(item.name, item.reason) for item in skipped_items] == [
        ("ogbg-molhiv", "requested skip"),
        ("ogbg-molmuv", "requested skip"),
    ]


def test_build_run_plan_skips_existing_checkpoints(tmp_path) -> None:
    items = [
        score.DatasetItem(
            config_name="clf_AMES",
            name="AMES",
            info=make_dataset_info("AMES"),
        ),
        score.DatasetItem(
            config_name="clf_DILI",
            name="DILI",
            info=make_dataset_info("DILI"),
        ),
    ]

    checkpoint = score.dataset_checkpoint_path(
        checkpoint_dir=tmp_path,
        dataset="AMES",
        embedder="modernmolbert_best_span",
    )
    checkpoint.write_text("dataset,embedder\nAMES,modernmolbert_best_span\n")

    run_items, skipped_items = score.build_run_plan(
        items=items,
        skip_set=set(),
        checkpoint_dir=tmp_path,
        embedder="modernmolbert_best_span",
        resume=True,
    )

    assert [item.name for item in run_items] == ["DILI"]
    assert [(item.name, item.reason) for item in skipped_items] == [
        ("AMES", "checkpoint exists"),
    ]


def test_build_run_plan_does_not_skip_checkpoint_when_resume_false(tmp_path) -> None:
    items = [
        score.DatasetItem(
            config_name="clf_AMES",
            name="AMES",
            info=make_dataset_info("AMES"),
        ),
    ]

    checkpoint = score.dataset_checkpoint_path(
        checkpoint_dir=tmp_path,
        dataset="AMES",
        embedder="modernmolbert_best_span",
    )
    checkpoint.write_text("dataset,embedder\nAMES,modernmolbert_best_span\n")

    run_items, skipped_items = score.build_run_plan(
        items=items,
        skip_set=set(),
        checkpoint_dir=tmp_path,
        embedder="modernmolbert_best_span",
        resume=False,
    )

    assert [item.name for item in run_items] == ["AMES"]
    assert skipped_items == []


def test_build_run_plan_requested_skip_takes_priority_over_checkpoint(tmp_path) -> None:
    items = [
        score.DatasetItem(
            config_name="clf_AMES",
            name="AMES",
            info=make_dataset_info("AMES"),
        ),
    ]

    checkpoint = score.dataset_checkpoint_path(
        checkpoint_dir=tmp_path,
        dataset="AMES",
        embedder="modernmolbert_best_span",
    )
    checkpoint.write_text("dataset,embedder\nAMES,modernmolbert_best_span\n")

    run_items, skipped_items = score.build_run_plan(
        items=items,
        skip_set={"AMES"},
        checkpoint_dir=tmp_path,
        embedder="modernmolbert_best_span",
        resume=True,
    )

    assert run_items == []
    assert [(item.name, item.reason) for item in skipped_items] == [
        ("AMES", "requested skip"),
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


def test_run_eval_raises_on_failure_without_safe_mode(monkeypatch, tmp_path) -> None:
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

    with pytest.raises(RuntimeError, match="boom"):
        score.run_eval(
            safe=False,
            embed_config=embed_config,
            full_model_name="runs/modernmolbert_best_span",
            short_model_name="modernmolbert_best_span",
            dataset_info=make_dataset_info("AMES"),
            model_head="rf",
            output_csv=tmp_path / "results.csv",
            override=False,
        )
