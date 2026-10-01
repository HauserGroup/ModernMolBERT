import json

import joblib
import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.common.types import (
    Dataset,
    EmbeddedDataset,
)
from modernmolbert.hf_upload import file_sha256
import materialize_revision_common_embeddings as common


def test_common_embeddings_share_source_rows_labels_and_splits(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    embedded_dir = tmp_path / "embedded"
    prepared_dir.mkdir()
    embedded_dir.mkdir()
    source = Dataset(
        name="tiny",
        task="classification",
        data=pd.DataFrame({"smiles": ["C", "CC", "CCC", "CCCC"], "Y": [0, 1, 0, 1]}),
        splits={"train": [0, 1], "valid": [2], "test": [3]},
    )
    prepared_path = prepared_dir / "tiny.json"
    source.serialize_legacy(prepared_path)
    prepared_sha = file_sha256(prepared_path)
    (embedded_dir / "tiny").mkdir()
    for index, run_id in enumerate(common.RUN_IDS):
        retained = [0, 2, 3] if index == 0 else [0, 1, 2, 3]
        embedding = EmbeddedDataset(
            name="tiny",
            task="classification",
            embedder=f"PREFLIGHT_{run_id}",
            splits={
                "train": [position for position, row in enumerate(retained) if row in {0, 1}],
                "valid": [position for position, row in enumerate(retained) if row == 2],
                "test": [position for position, row in enumerate(retained) if row == 3],
            },
            X=np.array([[row, index] for row in retained], dtype=np.float32),
            y=source.labels.iloc[retained].copy(),
            metadata={
                "source_row_indices": retained,
                "failed_source_row_indices": [1] if index == 0 else [],
                "prepared_data_sha256": prepared_sha,
                "pooling": "mean",
                "max_seq_length": 384,
            },
        )
        joblib.dump(embedding, embedded_dir / "tiny" / f"PREFLIGHT_{run_id}.joblib")

    monkeypatch.setattr(common, "PREPARED", prepared_dir)
    monkeypatch.setattr(common, "EMBEDDED", embedded_dir)
    manifest = common.materialize_task("tiny", "PREFLIGHT_", "PREFLIGHT_COMMON_", overwrite=False)
    assert manifest["common_supervised_rows"] == 3
    assert manifest["splits"] == {"train": 1, "valid": 1, "test": 1}
    assert len(manifest["labels_sha256"]) == 64
    assert set(manifest["split_source_row_indices_sha256"]) == {"train", "valid", "test"}
    assert manifest["endpoint_viability"]["Y"]["observed_test_rows"] == 1
    for index, run_id in enumerate(common.RUN_IDS):
        embedded = joblib.load(embedded_dir / "tiny" / f"PREFLIGHT_COMMON_{run_id}.joblib")
        assert embedded.metadata["source_row_indices"] == [0, 2, 3]
        assert "failed_source_row_indices" not in embedded.metadata
        assert embedded.splits == {"train": [0], "valid": [1], "test": [2]}
        assert embedded.y["Y"].tolist() == [0, 0, 1]
        np.testing.assert_array_equal(
            embedded.X, np.array([[0, index], [2, index], [3, index]], dtype=np.float32)
        )

    path = embedded_dir / "tiny" / f"PREFLIGHT_{common.RUN_IDS[0]}.joblib"
    source_embedding = joblib.load(path)
    source_embedding.y.iloc[0, 0] = 99
    joblib.dump(source_embedding, path)
    with pytest.raises(ValueError, match="labels differ"):
        common.materialize_task("tiny", "PREFLIGHT_", "OTHER_", overwrite=False)

    source_embedding.y.iloc[0, 0] = 0
    source_embedding.splits["test"] = [0]
    joblib.dump(source_embedding, path)
    with pytest.raises(ValueError, match="split differs"):
        common.materialize_task("tiny", "PREFLIGHT_", "OTHER_", overwrite=False)


def test_final_model_identity_rejects_embedding_from_other_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    model = tmp_path / "runs/revision_factorial_v1/small_ape_selfies/seed42/final_model"
    model.mkdir(parents=True)
    weights = model / "model.safetensors"
    weights.write_bytes(b"accepted weights")
    expected = file_sha256(weights)
    identity = model.parent / "run_identity.json"
    identity.write_text(
        json.dumps(
            {
                "result": {
                    "terminal_step": 30_000,
                    "final_model_file": weights.name,
                    "final_model_sha256": expected,
                }
            }
        )
    )
    source = EmbeddedDataset(
        name="tiny",
        task="classification",
        embedder="REVISION_small_ape_selfies",
        splits={"train": [0], "valid": [], "test": [1]},
        X=np.zeros((2, 2), dtype=np.float32),
        y=pd.DataFrame({"Y": [0, 1]}),
        metadata={"model_dir": str(model), "model_weights_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="Embedding weights differ"):
        common.final_model_identity(source, "small_ape_selfies")
    source.metadata["model_weights_sha256"] = expected
    assert (
        common.final_model_identity(source, "small_ape_selfies")["final_model_sha256"] == expected
    )
