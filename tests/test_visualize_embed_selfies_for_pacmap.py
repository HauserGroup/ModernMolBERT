import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from modernmolbert.visualize.embed_selfies_for_pacmap import (
    choose_device,
    load_input_frame,
    mean_pool,
    save_outputs,
)


def test_choose_device_cpu() -> None:
    device = choose_device("cpu")
    assert device == torch.device("cpu")


def test_choose_device_explicit_string() -> None:
    device = choose_device("cpu")
    assert str(device) == "cpu"


def test_choose_device_auto_returns_device() -> None:
    device = choose_device("auto")
    assert isinstance(device, torch.device)


def test_mean_pool_uniform_mask() -> None:
    # 2 sequences, length 3, hidden 4 — all tokens valid
    hidden = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.long)
    out = mean_pool(hidden, mask)
    assert out.shape == (2, 4)
    # expected: mean along dim 1
    expected = hidden.float().mean(dim=1)
    torch.testing.assert_close(out, expected)


def test_mean_pool_partial_mask() -> None:
    # 1 sequence, length 3, hidden 2; only first token valid
    hidden = torch.tensor([[[1.0, 2.0], [99.0, 99.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 0, 0]])
    out = mean_pool(hidden, mask)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]]))


def test_mean_pool_output_shape() -> None:
    hidden = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, dtype=torch.long)
    out = mean_pool(hidden, mask)
    assert out.shape == (4, 64)


def test_load_input_frame_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]", "[O]"], "alogp": [1.0, 2.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    loaded = load_input_frame(path)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == 2


def test_load_input_frame_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]"], "alogp": [1.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    loaded = load_input_frame(path)
    assert "selfies" in loaded.columns


def test_load_input_frame_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_input_frame(tmp_path / "nonexistent.parquet")


def test_load_input_frame_bad_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported input suffix"):
        load_input_frame(path)


def test_save_outputs_writes_all_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]", "[O]"], "alogp": [1.0, 2.0]})
    embeddings = np.random.default_rng(0).random((2, 8)).astype(np.float32)
    metadata = {"model_path": "test", "n_rows": 2}

    save_outputs(df=df, embeddings=embeddings, output_dir=tmp_path, metadata=metadata)

    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "metadata.parquet").exists()
    assert (tmp_path / "embedding_metadata.json").exists()


def test_save_outputs_embeddings_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]", "[O]"]})
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    save_outputs(df=df, embeddings=embeddings, output_dir=tmp_path, metadata={})

    loaded = np.load(tmp_path / "embeddings.npy")
    np.testing.assert_array_equal(loaded, embeddings)


def test_save_outputs_metadata_has_embedding_row(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]", "[O]", "[N]"]})
    embeddings = np.zeros((3, 4), dtype=np.float32)

    save_outputs(df=df, embeddings=embeddings, output_dir=tmp_path, metadata={})

    meta = pd.read_parquet(tmp_path / "metadata.parquet")
    assert "embedding_row" in meta.columns
    assert list(meta["embedding_row"]) == [0, 1, 2]


def test_save_outputs_config_json_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]"]})
    embeddings = np.zeros((1, 4), dtype=np.float32)
    metadata = {"model_path": "some/model", "pooling": "mean", "n_rows": 1}

    save_outputs(df=df, embeddings=embeddings, output_dir=tmp_path, metadata=metadata)

    saved = json.loads((tmp_path / "embedding_metadata.json").read_text())
    assert saved == metadata
