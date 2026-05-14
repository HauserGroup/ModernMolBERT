from pathlib import Path

import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.data import (
    load_dataset_catalog,
    select_dataset_configs,
)
from modernmolbert.eval.benchmarking_molecular_models.heads import (
    downstream_configs_for_heads,
    frozen_downstream_config,
)
from modernmolbert.eval.benchmarking_molecular_models.run import (
    _write_summary,
    build_suite_config,
)


def test_active_benchmark_package_contains_no_upstream_model_zoo() -> None:
    active = Path("src/modernmolbert/eval/benchmarking_molecular_models")

    assert active.exists()
    assert not (active / "model_wrappers").exists()
    assert not (active / ".git").exists()
    assert not (active / "embed_wrapper.sh").exists()
    assert not (active / "run_scoring.sh").exists()


def test_dataset_catalog_selects_prepared_moleculenet_configs() -> None:
    catalog = load_dataset_catalog()

    assert "bbbp" in catalog
    assert catalog["bbbp"]["loader"] == "prepared_moleculenet"

    configs = select_dataset_configs(["bbbp"], prepared_root="/tmp/prepared")

    assert configs == [
        {
            "name": "bbbp",
            "loader": "prepared_moleculenet",
            "dataset_dir": "/tmp/prepared/bbbp",
            "eval_split": "test",
            "merge_train_valid": True,
        }
    ]


def test_head_configs_are_limited_to_lightweight_downstream_heads() -> None:
    configs = downstream_configs_for_heads(["logreg", "ridge"], seed=17)

    assert [x["model_type"] for x in configs["classification"]] == ["logistic_regression"]
    assert [x["model_type"] for x in configs["regression"]] == ["ridge"]
    assert frozen_downstream_config("logreg", seed=17).model_type == "logistic_regression"


def test_build_suite_config_uses_only_modernmolbert_featurizer() -> None:
    suite = build_suite_config(
        model_path="runs/model/final_model",
        datasets=["bbbp"],
        prepared_root="/tmp/prepared",
        heads=["logreg"],
        seed=13,
        use_cache=False,
    )

    assert len(suite.featurizers) == 1
    assert suite.featurizers[0].config["type"] == "modernmolbert_selfies"
    assert suite.featurizers[0].config["model_dir"] == "runs/model/final_model"
    assert suite.datasets[0].config["dataset_dir"] == "/tmp/prepared/bbbp"
    assert [x.name for x in suite.downstream_models] == ["logistic_regression"]
    assert suite.use_cache is False


def test_write_summary_selects_best_metric_per_dataset_task(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["bbbp", "bbbp", "esol", "esol"],
            "task": ["label", "label", "y", "y"],
            "downstream_name": ["a", "b", "ridge", "ridge2"],
            "roc_auc": [0.7, 0.8, None, None],
            "rmse": [None, None, 1.5, 1.2],
        }
    )

    out = tmp_path / "summary.csv"
    _write_summary(frame, out)

    summary = pd.read_csv(out)

    assert summary.loc[summary["dataset"] == "bbbp", "downstream_name"].item() == "b"
    assert summary.loc[summary["dataset"] == "esol", "downstream_name"].item() == "ridge2"
