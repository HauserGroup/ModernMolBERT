import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "paper_frames", Path(__file__).parents[1] / "scripts/paper/build_benchmark_results_frames.py"
)
assert spec is not None and spec.loader is not None
frames = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frames)


def candidates():
    return pd.DataFrame(
        {
            "dataset": ["A", "A"],
            "embedder": ["encoder", "encoder"],
            "model": ["ridge", "rf"],
            "cv_metric_name": ["roc_auc", "roc_auc"],
            "test_metric_name": ["roc_auc", "roc_auc"],
            "cv_metric": [0.8, 0.7],
            "test_metric": [0.6, 0.95],
            "split": ["scaffold13", "scaffold13"],
        }
    )


def test_cv_winner_does_not_depend_on_test_score():
    data = candidates()
    assert frames.collapse_best_head(data).iloc[0]["model"] == "ridge"
    data["test_metric"] = [np.nan, 1.0]
    selected = frames.collapse_best_head(data)
    assert selected.iloc[0]["model"] == "ridge"
    assert pd.isna(selected.iloc[0]["test_metric"])
    assert selected.iloc[0]["selection_metric"] == "cv_metric"


def test_cv_tie_is_test_independent():
    data = candidates()
    data["cv_metric"] = 0.8
    first = frames.collapse_best_head(data).iloc[0]["model"]
    data["test_metric"] = data["test_metric"].to_numpy()[::-1]
    assert frames.collapse_best_head(data).iloc[0]["model"] == first


@pytest.mark.parametrize(
    "column,value", [("cv_metric", np.nan), ("cv_metric", np.inf), ("cv_metric_name", "accuracy")]
)
def test_missing_or_incompatible_validation_fails(column, value):
    data = candidates()
    data.loc[0, column] = value
    with pytest.raises(ValueError):
        frames.collapse_best_head(data)


def test_repeated_runs_are_not_selected_by_score():
    data = candidates()
    with pytest.raises(ValueError, match="Repeated head"):
        frames.collapse_best_head(pd.concat([data, data]))
    data.loc[0, "split"] = "another_split"
    with pytest.raises(ValueError, match="provenance"):
        frames.collapse_best_head(data)


def test_subsampling_identity_and_metadata_survive_normalisation(tmp_path):
    path = tmp_path / "praski_best_standard" / "results.csv"
    path.parent.mkdir()
    data = candidates()
    data["embedder"] = "modernmolbert_best_standard__subsample_train8000_seed42"
    data["embedding_max_seq_length"] = 128
    data.to_csv(path, index=False)
    selected = frames.collapse_best_head(frames.normalize_own_result(path))
    assert selected.iloc[0]["embedder"].endswith("__subsample_train8000_seed42")
    assert selected.iloc[0]["embedding_max_seq_length"] == 128
    assert selected.iloc[0]["result_source"] == str(path.resolve())


def test_directory_cannot_relabel_small_checkpoint_as_base(tmp_path):
    path = tmp_path / "praski_best_base_standard" / "results.csv"
    path.parent.mkdir()
    data = candidates()
    data["embedder"] = "modernmolbert_best_standard"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="identity conflicts"):
        frames.normalize_own_result(path)
