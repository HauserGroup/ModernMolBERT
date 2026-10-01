import argparse
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models import score
from modernmolbert.eval.benchmarking_molecular_models.common.types import EmbeddedDataset
from modernmolbert.eval.benchmarking_molecular_models.praski_export import append_result_row


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
    )

    assert ok is False


def test_get_disabled_reason_disables_knn_for_hiv_and_muv() -> None:
    muv_info = make_dataset_info("clf_ogbg-molmuv")
    hiv_info = make_dataset_info("clf_ogbg-molhiv")
    ames_info = make_dataset_info("clf_AMES")

    assert score.get_disabled_reason(muv_info, "knn") == "KNN disabled for MUV"
    assert score.get_disabled_reason(hiv_info, "knn") == "KNN disabled for HIV"
    assert score.get_disabled_reason(ames_info, "knn") is None
    assert score.get_disabled_reason(hiv_info, "rf") is None


def test_scoring_identity_changes_with_cohort_or_label_policy() -> None:
    def identity(embedding: str = "a" * 64, mode: str = "as-negative", version: str = "v1"):
        return score.scoring_identity(
            dataset="AMES",
            embedder="emb",
            head="rf",
            embedding_sha256=embedding,
            prepared_sha256="b" * 64,
            missing_labels=mode,
            version_hash=version,
            code_revision={"commit": "c" * 40, "dirty": False},
        )

    original = identity()
    assert identity(embedding="d" * 64) != original
    assert identity(mode="observed") != original
    assert identity(version="v2") != original


def test_results_row_is_only_resume_state(tmp_path) -> None:
    output = tmp_path / "results.csv"
    prediction = tmp_path / "rf.npz"
    pd.DataFrame(
        [{"dataset": "AMES", "embedder": "emb", "model": "rf", "scoring_identity": "id-1"}]
    ).to_csv(output, index=False)
    np.savez(prediction, scoring_identity=np.asarray("id-1"))
    assert score.score_row_is_complete(output, "AMES", "emb", "rf", "id-1", prediction)
    prediction.unlink()
    assert not score.score_row_is_complete(output, "AMES", "emb", "rf", "id-1", prediction)
    np.savez(prediction, scoring_identity=np.asarray("other"))
    assert not score.score_row_is_complete(output, "AMES", "emb", "rf", "id-1", prediction)

    import pytest

    with pytest.raises(ValueError, match="different or duplicate"):
        score.score_row_is_complete(output, "AMES", "emb", "rf", "id-2", prediction)
    rows = pd.read_csv(output)
    pd.concat([rows, rows]).to_csv(output, index=False)
    with pytest.raises(ValueError, match="different or duplicate"):
        score.score_row_is_complete(output, "AMES", "emb", "rf", "id-1", prediction)


def test_scoring_main_resumes_only_matching_result_and_prediction(monkeypatch, tmp_path) -> None:
    embedded_dir = tmp_path / "embedded"
    predictions_dir = tmp_path / "predictions"
    source = embedded_dir / "toy" / "emb.joblib"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen embedding")
    output_csv = tmp_path / "results.csv"
    args = argparse.Namespace(
        config_dir="config",
        model_name="emb",
        overrides=[],
        datasets=["toy"],
        skip_datasets=None,
        subsample_size=None,
        subsample_scope=None,
        subsample_seed=42,
        heads=["rf"],
        output_csv=output_csv,
        resume=True,
        safe=False,
        missing_labels="as-negative",
        n_jobs=1,
    )
    monkeypatch.setattr(score, "parse_args", lambda: args)
    monkeypatch.setattr(score, "load_yaml_config", lambda path: {})
    monkeypatch.setattr(
        score,
        "load_embedding_config",
        lambda path: {
            "raw_directory": str(tmp_path / "raw"),
            "embedded_directory": str(embedded_dir),
            "predictions_directory": str(predictions_dir),
            "prepared_directory": str(tmp_path / "prepared"),
            "max_invalid_embeddings": 0,
        },
    )
    monkeypatch.setattr(
        score,
        "load_dataset_items",
        lambda **kwargs: [score.DatasetItem("clf_toy", "toy", make_dataset_info("toy"))],
    )
    embedded = make_embedded_dataset()
    embedded.metadata["prepared_data_sha256"] = "a" * 64
    monkeypatch.setattr(score, "load_embedded_dataset", lambda **kwargs: embedded)
    monkeypatch.setattr(score, "get_model_version_hash", lambda: "grid-v1")
    monkeypatch.setattr(score, "get_git_revision", lambda: {"commit": "b" * 40, "dirty": False})

    calls: list[str] = []

    def fake_run_eval(**kwargs) -> bool:
        identity = kwargs["scoring_identity_value"]
        calls.append(identity)
        append_result_row(
            output_csv,
            {
                "dataset": "toy",
                "embedder": "emb",
                "model": "rf",
                "scoring_identity": identity,
                "cv_metric_name": "roc_auc",
            },
            replace_existing=True,
        )
        prediction = predictions_dir / "toy" / "emb" / "rf.npz"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prediction, scoring_identity=np.asarray(identity))
        return True

    monkeypatch.setattr(score, "run_eval", fake_run_eval)
    assert score.main() == 0
    assert score.main() == 0
    assert len(calls) == 1
    assert len(pd.read_csv(output_csv)) == 1

    args.missing_labels = "observed"
    with pytest.raises(ValueError, match="different or duplicate identity"):
        score.main()
