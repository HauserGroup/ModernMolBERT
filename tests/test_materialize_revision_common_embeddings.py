import joblib
import numpy as np
import pandas as pd

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
            splits={"train": [], "valid": [], "test": []},
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
    for index, run_id in enumerate(common.RUN_IDS):
        embedded = joblib.load(embedded_dir / "tiny" / f"PREFLIGHT_COMMON_{run_id}.joblib")
        assert embedded.metadata["source_row_indices"] == [0, 2, 3]
        assert embedded.metadata["failed_source_row_indices"] == [1]
        assert embedded.splits == {"train": [0], "valid": [1], "test": [2]}
        assert embedded.y["Y"].tolist() == [0, 0, 1]
        np.testing.assert_array_equal(
            embedded.X, np.array([[0, index], [2, index], [3, index]], dtype=np.float32)
        )
