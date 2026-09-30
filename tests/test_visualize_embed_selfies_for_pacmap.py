from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from modernmolbert.visualize.embed_selfies_for_pacmap import (
    load_input_frame,
    mean_pool,
    save_outputs,
)


def test_mean_pool_partial_mask() -> None:
    # 1 sequence, length 3, hidden 2; only first token valid
    hidden = torch.tensor([[[1.0, 2.0], [99.0, 99.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 0, 0]])
    out = mean_pool(hidden, mask)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]]))


def test_load_input_frame_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_input_frame(tmp_path / "nonexistent.parquet")


def test_save_outputs_writes_all_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"selfies": ["[C]", "[O]"], "alogp": [1.0, 2.0]})
    embeddings = np.random.default_rng(0).random((2, 8)).astype(np.float32)
    metadata = {"model_path": "test", "n_rows": 2}

    save_outputs(df=df, embeddings=embeddings, output_dir=tmp_path, metadata=metadata)

    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "metadata.parquet").exists()
    assert (tmp_path / "embedding_metadata.json").exists()
