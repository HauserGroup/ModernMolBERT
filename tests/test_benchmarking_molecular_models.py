from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    PRASKI_COLUMNS,
    append_result_row,
    read_results_csv,
)
from modernmolbert.eval.benchmarking_molecular_models.embed_modernmolbert import (
    embed_dataset,
)
from modernmolbert.eval.featurizers.base import FeatureBatch
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
    assert (root / "embed_modernmolbert.py").exists()
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


class FakeFeaturizer:
    name = "fake"

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def featurize_smiles(self, smiles, *, batch_size: int):
        valid_mask = np.array([smi != "bad" for smi in smiles], dtype=bool)
        X = np.array(
            [[float(i), self.scale] for i, is_valid in enumerate(valid_mask) if is_valid],
            dtype=np.float32,
        )
        return FeatureBatch(X=X, valid_mask=valid_mask, metadata={"hidden_size": 2})


def test_embed_dataset_aligns_invalid_features_and_remaps_splits() -> None:
    dataset = Dataset(
        name="tiny",
        task="classification",
        data=pd.DataFrame({"smiles": ["CCO", "bad", "CCC", "CCN"], "label": [0, 1, 1, 0]}),
        splits={"train": [0, 1], "valid": [2], "test": [3]},
    )

    embedded = embed_dataset(
        dataset,
        featurizer=FakeFeaturizer(),
        embedder_name="fake_embedder",
        batch_size=2,
    )

    assert embedded.embedder == "fake_embedder"
    assert embedded.X.shape == (3, 2)
    assert embedded.y["label"].tolist() == [0, 1, 0]
    assert embedded.splits == {"train": [0], "valid": [1], "test": [2]}


def write_embedding_test_config(config_dir: Path) -> None:
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


def test_embed_modernmolbert_cli_skips_existing_and_overwrites(monkeypatch, tmp_path) -> None:
    from modernmolbert.eval.benchmarking_molecular_models import embed_modernmolbert

    config_dir = tmp_path / "config"
    write_embedding_test_config(config_dir)
    prepared_dir = tmp_path / "data/prepared"
    prepared_dir.mkdir(parents=True)
    dataset = Dataset(
        name="tiny",
        task="classification",
        data=pd.DataFrame({"smiles": ["CCO", "CCC"], "label": [0, 1]}),
        splits={"train": [0], "valid": [], "test": [1]},
    )
    joblib.dump(dataset, prepared_dir / "tiny.joblib")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(embed_modernmolbert, "make_featurizer", lambda args: FakeFeaturizer(1.0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embed_modernmolbert.py",
            "--config-dir",
            str(config_dir),
            "--datasets",
            "tiny_clf",
            "--embedder",
            "fake_embedder",
        ],
    )
    embed_modernmolbert.main()
    output_path = tmp_path / "data/embedded/tiny/fake_embedder.joblib"
    first = joblib.load(output_path)
    assert first.X[:, 1].tolist() == [1.0, 1.0]

    monkeypatch.setattr(embed_modernmolbert, "make_featurizer", lambda args: FakeFeaturizer(2.0))
    embed_modernmolbert.main()
    skipped = joblib.load(output_path)
    assert skipped.X[:, 1].tolist() == [1.0, 1.0]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embed_modernmolbert.py",
            "--config-dir",
            str(config_dir),
            "--datasets",
            "tiny_clf",
            "--embedder",
            "fake_embedder",
            "--overwrite",
        ],
    )
    embed_modernmolbert.main()
    overwritten = joblib.load(output_path)
    assert overwritten.X[:, 1].tolist() == [2.0, 2.0]


def test_embed_modernmolbert_help_and_unknown_dataset() -> None:
    script = Path("src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py")
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0
    assert "--model-dir" in help_result.stdout

    bad_result = subprocess.run(
        [sys.executable, str(script), "--datasets", "does_not_exist"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_result.returncode != 0
    assert "Unknown dataset" in bad_result.stderr


def test_score_writes_dataset_checkpoint(monkeypatch, tmp_path) -> None:
    from modernmolbert.eval.benchmarking_molecular_models import score

    config_dir = tmp_path / "config"
    write_embedding_test_config(config_dir)
    (config_dir / "score.yaml").write_text("cache: true\ndatasets:\n  - tiny_clf\n")
    output_csv = tmp_path / "results.csv"
    checkpoint_dir = tmp_path / "checkpoints"

    def fake_eval(
        cfg,
        embed_config,
        model_name,
        dataset_info,
        short_model_name,
        model_head,
        output_csv_arg,
        override,
    ):
        append_result_row(
            output_csv_arg,
            {
                "dataset": dataset_info.name,
                "task": dataset_info.task,
                "embedder": short_model_name,
                "model": model_head,
                "hyperparams": "{}",
                "library_hash": "test",
                "cv_metric_name": dataset_info.metric,
                "cv_metric": 0.5,
                "test_metric_name": dataset_info.metric,
                "test_metric": 0.6,
            },
        )

    monkeypatch.setattr(score, "eval", fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score.py",
            "--config-dir",
            str(config_dir),
            "--datasets",
            "tiny_clf",
            "--heads",
            "rf",
            "--embedder",
            "fake_embedder",
            "--output-csv",
            str(output_csv),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    score.main()

    checkpoint = read_results_csv(checkpoint_dir / "tiny.csv")
    assert list(checkpoint.columns) == PRASKI_COLUMNS
    assert checkpoint.loc[0, "dataset"] == "tiny"
    assert checkpoint.loc[0, "embedder"] == "fake_embedder"


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
