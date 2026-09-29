from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.embed_modernmolbert import (
    embed_dataset,
)
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.benchmarking_molecular_models.common.types import (
    Dataset,
    EmbeddedDataset,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    multioutput_auroc_score,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.train import (
    fit_and_eval_embedding,
)


def test_tdc_loader_keeps_scaffold_split_when_one_smiles_fails(capsys) -> None:
    from modernmolbert.eval.benchmarking_molecular_models.common.data_v2 import (
        load_tdc_module_dataset,
    )

    instances = []

    class FakeTdcDataset:
        def __init__(self) -> None:
            self.entity1_name = "Drug"

        def get_data(self, format: str = "df"):
            assert format == "df"
            return pd.DataFrame(
                {
                    "Drug_ID": [1, 2, 3, 4],
                    "Drug": ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "not-a-smiles"],
                    "Y": [1, 0, 1, 0],
                }
            )

        def get_split(self, method: str = "random"):
            raise AssertionError(f"unexpected TDC split call: {method}")

    def fake_module(name: str, path: str, **kwargs):
        assert name == "Bioavailability_Ma"
        assert path == "data/raw"
        assert kwargs == {"label_name": "Y"}
        dataset = FakeTdcDataset()
        instances.append(dataset)
        return dataset

    data, splits = load_tdc_module_dataset(
        fake_module,
        "Bioavailability_Ma",
        "data/raw",
        label="Y",
    )

    assert instances[0].entity1_name == "Drug"
    output = capsys.readouterr().out
    assert "omitted 1 SMILES" in output
    assert "Falling back to random split" not in output
    assert set(data["smiles"]) == {"c1ccccc1", "C1CCCCC1", "c1ccncc1"}
    assert "Drug_ID" not in data.columns
    assert sorted(splits["train"] + splits["valid"] + splits["test"]) == list(range(len(data)))


def test_build_dataset_drops_uncanonicalizable_smiles_and_remaps_splits(capsys) -> None:
    from modernmolbert.eval.benchmarking_molecular_models.common.data_v2 import (
        build_dataset,
    )

    dataset = build_dataset(
        name="ogbg-molhiv",
        task="classification",
        raw_data=pd.DataFrame(
            {
                "smiles": ["CC", "[AlH6]", "CO", "CN"],
                "label": [0, 1, 0, 1],
            }
        ),
        splits={"train": [0, 1], "valid": [2], "test": [3]},
    )

    output = capsys.readouterr().out
    assert "Dropped 1 molecules" in output
    assert dataset.data["smiles"].tolist() == ["CC", "CO", "CN"]
    assert dataset.data["label"].tolist() == [0, 0, 1]
    assert dataset.splits == {"train": [0], "valid": [1], "test": [2]}


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
    assert embedded.metadata["source_row_indices"] == [0, 2, 3]
    assert embedded.metadata["failed_source_row_indices"] == [1]
    assert embedded.metadata["source_split_counts"] == {"train": 2, "valid": 1, "test": 1}
    assert embedded.metadata["retained_split_counts"] == {"train": 1, "valid": 1, "test": 1}


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
            "--model-dir",
            str(tmp_path / "model"),
        ],
    )
    embed_modernmolbert.main()
    output_path = tmp_path / "data/embedded/tiny/fake_embedder.joblib"
    first = joblib.load(output_path)
    assert first.X[:, 1].tolist() == [1.0, 1.0]
    assert first.metadata["prepared_data_sha256"]
    assert first.metadata["source_row_indices"] == [0, 1]

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
            "--model-dir",
            str(tmp_path / "model"),
            "--overwrite",
        ],
    )
    embed_modernmolbert.main()
    overwritten = joblib.load(output_path)
    assert overwritten.X[:, 1].tolist() == [2.0, 2.0]


def test_embed_dataset_preserves_pooling_metadata() -> None:
    dataset = Dataset(
        name="tiny",
        task="classification",
        data=pd.DataFrame({"smiles": ["C", "CC"], "label": [0, 1]}),
        splits={"train": [0], "valid": [], "test": [1]},
    )

    class FakeFeaturizer:
        def featurize_smiles(self, smiles, batch_size):
            return FeatureBatch(
                X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                valid_mask=np.array([True, True]),
                metadata={
                    "pooling": "mean",
                    "pooling_special_tokens_excluded": True,
                    "model_dir": "runs/model/final_model",
                    "tokenizer_path": "runs/model/final_model",
                    "max_seq_length": 256,
                },
            )

    embedded = embed_dataset(
        dataset,
        featurizer=FakeFeaturizer(),
        embedder_name="modernmolbert",
        batch_size=2,
    )

    assert embedded.metadata["pooling"] == "mean"
    assert embedded.metadata["pooling_special_tokens_excluded"] is True
    assert embedded.metadata["max_seq_length"] == 256


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


def test_multioutput_auroc_handles_inhomogeneous_predict_proba_list() -> None:
    y_true = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=float,
    )
    y_score = [
        np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]),
        np.ones((4, 1)),
        np.array([[0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2]]),
    ]

    assert multioutput_auroc_score(y_true, y_score) == 1.0


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
        metadata={
            "source_row_indices": list(range(100, 120)),
            "prepared_data_sha256": "a" * 64,
        },
    )

    result = fit_and_eval_embedding(
        dataset=dataset,
        model_head="knn",
        memory_weight=32,
    )

    assert result.model == "knn"
    assert result.y_test_pred.shape == (5, 2)
    assert "clf__n_neighbors" in result.hyperparams
    assert result.test_source_row_indices is not None
    assert result.test_source_row_indices.tolist() == [115, 116, 117, 118, 119]
    assert result.prepared_data_sha256 == "a" * 64


def test_regression_path_returns_1d_predictions() -> None:
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

    result = fit_and_eval_embedding(
        dataset=dataset,
        model_head="ridge",
        memory_weight=32,
    )

    assert result.model == "ridge"
    assert result.y_test_pred.ndim == 1
    assert result.y_test_pred.shape == (5,)
