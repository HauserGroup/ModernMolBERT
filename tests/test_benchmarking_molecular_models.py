from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    Dataset,
    EmbeddedDataset,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.eval_metrics import (
    multioutput_auroc_score,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.models import (
    get_knn_distance,
    tanimoto_count_distance,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.train import (
    fit_and_eval_embedding,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.utils import (
    get_model_version_hash,
)


def test_benchmark_package_is_stripped_to_dataset_scoring_and_export() -> None:
    root = Path("src/modernmolbert/eval/benchmarking_molecular_models")

    assert root.exists()
    assert (root / "download.py").exists()
    assert (root / "score.py").exists()
    assert (root / "run_scoring.sh").exists()
    assert (root / "config" / "datasets.yaml").exists()

    removed = [
        "docs",
        "results",
        "config/dataset",
        "config/experiment",
        "config/model",
        "embed.py",
        "embed_wrapper.sh",
        "run_embed.sh",
        "get_embedding_size.py",
        "embedding_size.json",
        "fixup.ipynb",
        "selfies_debugger.ipynb",
        "visualizations.ipynb",
        "requirements.txt",
        "base_requirements.txt",
        "export_results.py",
        "src/common/db.py",
    ]
    for rel_path in removed:
        assert not (root / rel_path).exists()


def test_dataset_registry_is_single_yaml() -> None:
    config_dir = Path("src/modernmolbert/eval/benchmarking_molecular_models/config")

    assert expand_dataset_selection(config_dir, ["clf_ogbg-mol*"]) == [
        "clf_ogbg-molbace",
        "clf_ogbg-molbbbp",
        "clf_ogbg-molclintox",
        "clf_ogbg-molhiv",
        "clf_ogbg-molmuv",
        "clf_ogbg-molsider",
        "clf_ogbg-moltox21",
        "clf_ogbg-moltoxcast",
    ]
    assert load_dataset_config(config_dir, "clf_AMES").name == "AMES"


def test_stripped_benchmark_has_no_hydra_or_omegaconf_runtime_dependency() -> None:
    root = Path("src/modernmolbert/eval/benchmarking_molecular_models")
    checked_files = [
        path
        for path in root.rglob("*")
        if path.suffix in {".py", ".yaml"} and "__pycache__" not in path.parts
    ]

    for path in checked_files:
        text = path.read_text()
        assert "hydra" not in text.lower(), path
        assert "omegaconf" not in text.lower(), path
        assert "get_original_cwd" not in text, path


def test_stripped_benchmark_has_no_sql_runtime() -> None:
    root = Path("src/modernmolbert/eval/benchmarking_molecular_models")
    checked_files = [
        path
        for path in root.rglob("*")
        if path.suffix in {".py", ".yaml", ".md"} and "__pycache__" not in path.parts
    ]

    for path in checked_files:
        text = path.read_text().lower()
        assert "sqlite" not in text, path
        assert "classificationreport" not in text, path
        assert "meta.db" not in text, path


def test_download_prepares_missing_dataset_without_network(monkeypatch, tmp_path) -> None:
    from modernmolbert.eval.benchmarking_molecular_models import download

    config_dir = tmp_path / "config"
    (config_dir / "embedding").mkdir(parents=True)
    (config_dir / "embedding" / "default.yaml").write_text(
        "\n".join(
            [
                "raw_directory: data/raw",
                "embedded_directory: data/embedded",
                "data_directory: data/downloaded",
                "prepared_directory: data/prepared",
                "predictions_directory: data/predictions",
                "clock_directory: data/clock",
                "svd_directory: data/svd",
                "max_invalid_embeddings: 50",
            ]
        )
    )
    (config_dir / "downloader.yaml").write_text("cache: true\ndatasets:\n  - tiny_clf\n")
    (config_dir / "datasets.yaml").write_text(
        "\n".join(
            [
                "datasets:",
                "  tiny_clf:",
                "    name: tiny",
                "    metric: roc_auc",
                "    task: classification",
                "    source:",
                "      name: OGB",
            ]
        )
    )
    fake_dataset = Dataset(
        name="tiny",
        task="classification",
        data=pd.DataFrame({"smiles": ["C"], "label": [1]}),
        splits={"train": [0], "valid": [], "test": []},
    )
    calls = []

    def fake_load(dataset_config, raw_dir):
        calls.append((dataset_config.name, raw_dir))
        return fake_dataset

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download, "load", fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download.py",
            "--config-dir",
            str(config_dir),
        ],
    )

    download.main()

    assert calls == [("tiny", "data/raw")]
    assert (tmp_path / "data/prepared/tiny.joblib").exists()
    assert (tmp_path / "data/prepared/tiny.json").exists()


def test_local_tanimoto_count_distance_matches_count_vector_formula() -> None:
    x = np.array([1, 2, 0, 4])
    y = np.array([1, 1, 3, 0])

    assert tanimoto_count_distance(x, y) == pytest.approx(1.0 - 2 / 10)
    assert get_knn_distance(np.dtype("int64")) is tanimoto_count_distance
    assert get_knn_distance(np.dtype("float32")) == "cosine"


def test_multioutput_auroc_masks_nan_labels_per_output() -> None:
    y_true = np.array(
        [
            [0, 1],
            [1, np.nan],
            [0, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y_score = np.array(
        [
            [0.1, 0.8],
            [0.9, 0.6],
            [0.2, 0.3],
            [0.8, 0.9],
        ]
    )

    assert multioutput_auroc_score(y_true, y_score) == 1.0


def test_model_version_hash_is_stable_digest() -> None:
    digest = get_model_version_hash()

    assert digest == get_model_version_hash()
    assert isinstance(digest, str)
    assert len(digest) == 16


def test_fit_and_eval_embedding_binary_classification_knn() -> None:
    X = np.array([[i, i % 3] for i in range(20)], dtype=float)
    y = pd.DataFrame({"label": [0] * 10 + [1] * 10})
    dataset = EmbeddedDataset(
        name="tiny",
        task="classification",
        embedder="toy_embedder",
        splits={
            "train": list(range(0, 10)),
            "valid": list(range(10, 15)),
            "test": list(range(15, 20)),
        },
        X=X,
        y=y,
    )

    result = fit_and_eval_embedding(
        dataset=dataset,
        metric_name="roc_auc",
        model_head="knn",
        memory_weight=32,
    )

    assert result.model == "knn"
    assert result.y_test_pred.shape == (5, 2)
    assert "clf__n_neighbors" in result.hyperparams


def test_regression_path_still_exposes_predict_proba_limitation() -> None:
    X = np.array([[i, i + 1] for i in range(20)], dtype=float)
    y = pd.DataFrame({"label": np.linspace(0.0, 1.0, 20)})
    dataset = EmbeddedDataset(
        name="tiny_reg",
        task="regression",
        embedder="toy_embedder",
        splits={
            "train": list(range(0, 10)),
            "valid": list(range(10, 15)),
            "test": list(range(15, 20)),
        },
        X=X,
        y=y,
    )

    with pytest.raises(AttributeError, match="predict_proba"):
        fit_and_eval_embedding(
            dataset=dataset,
            metric_name="mae",
            model_head="ridge",
            memory_weight=32,
        )


def test_run_scoring_requires_embedder_name() -> None:
    script = Path("src/modernmolbert/eval/benchmarking_molecular_models/run_scoring.sh")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 1
    assert "Usage:" in result.stdout
