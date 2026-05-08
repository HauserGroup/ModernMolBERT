from pathlib import Path

import numpy as np
import pandas as pd

from modernmolbert.eval.datasets import EvalDataset
from modernmolbert.eval.downstream import FrozenDownstreamConfig
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.registry import make_featurizer
from modernmolbert.eval.runner import FrozenBenchmarkRunner


def test_feature_batch_shape_check_passes() -> None:
    batch = FeatureBatch(
        X=np.ones((2, 4), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )
    batch.check(3)


def test_feature_batch_shape_check_fails_on_bad_row_count() -> None:
    batch = FeatureBatch(
        X=np.ones((1, 4), dtype=np.float32),
        valid_mask=np.array([True, False, True]),
    )

    try:
        batch.check(3)
    except ValueError as e:
        assert "Number of rows" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_dummy_featurizer_valid_mask() -> None:
    featurizer = DummyFeaturizer(n_features=4)
    out = featurizer.featurize_smiles(["CCO", "", "N"])
    out.check(3)

    assert out.valid_mask.tolist() == [True, False, True]
    assert out.X.shape == (2, 4)


def test_registry_makes_dummy_featurizer() -> None:
    featurizer = make_featurizer("dummy", name="dummy_test", n_features=3)
    out = featurizer.featurize_smiles(["CCO"])
    assert out.X.shape == (1, 3)


def test_eval_dataset_check() -> None:
    train = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]})
    test = pd.DataFrame({"smiles": ["CO", "CN"], "label": [0, 1]})

    ds = EvalDataset(
        name="tiny",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )
    ds.check()


def test_frozen_runner_classification(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CCCl", "CCBr", "CO"],
            "label": [0, 0, 1, 1, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCBr", "CO"],
            "label": [0, 1, 1, 0],
        }
    )

    ds = EvalDataset(
        name="tiny_classification",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(
        downstream_config=FrozenDownstreamConfig(),
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    result = runner.run(
        dataset=ds,
        featurizer=DummyFeaturizer(n_features=8),
        output_dir=tmp_path / "out",
    )

    assert len(result.task_results) == 1
    metrics = result.task_results[0].metrics
    assert "accuracy" in metrics
    assert "roc_auc" in metrics

    assert (tmp_path / "out" / "results.json").exists()
    assert (tmp_path / "out" / "results.csv").exists()


def test_frozen_runner_regression(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC", "CCCl"],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CO", "CN", "CCBr"],
            "y": [0.15, 0.25, 0.55],
        }
    )

    ds = EvalDataset(
        name="tiny_regression",
        task_type="regression",
        task_names=["y"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(
        downstream_config=FrozenDownstreamConfig(),
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    result = runner.run(
        dataset=ds,
        featurizer=DummyFeaturizer(n_features=8),
        output_dir=tmp_path / "out",
    )

    assert len(result.task_results) == 1
    metrics = result.task_results[0].metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_runner_cache_reuse(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC"],
            "label": [0, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "smiles": ["CO", "CCBr"],
            "label": [0, 1],
        }
    )

    ds = EvalDataset(
        name="cache_test",
        task_type="classification",
        task_names=["label"],
        train=train,
        valid=None,
        test=test,
    )

    runner = FrozenBenchmarkRunner(
        cache_dir=tmp_path / "cache",
        use_cache=True,
        batch_size=2,
    )

    featurizer = DummyFeaturizer(n_features=8)

    runner.run(dataset=ds, featurizer=featurizer)
    runner.run(dataset=ds, featurizer=featurizer)

    cache_files = list((tmp_path / "cache").rglob("features.npy"))
    assert cache_files
