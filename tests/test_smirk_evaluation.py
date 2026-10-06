"""Provenance checks for the isolated SMIRK matched benchmark cohort."""

import numpy as np
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset, EmbeddedDataset
from scripts.experiments.evaluate_smirk_seed import subset, verify_source_rows


def test_smirk_cohort_keeps_source_order_labels_and_splits() -> None:
    prepared = Dataset(
        name="example",
        task="classification",
        data=pd.DataFrame({"smiles": ["C", "N", "O", "S"], "target": [0, 1, 0, 1]}),
        splits={"train": [0, 1], "valid": [2], "test": [3]},
    )
    source = EmbeddedDataset(
        name="example",
        task="classification",
        embedder="source",
        splits={"train": [0, 2], "valid": [1], "test": [3]},
        X=np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32),
        y=prepared.labels.iloc[[1, 2, 0, 3]].reset_index(drop=True),
        metadata={"source_row_indices": [1, 2, 0, 3]},
    )

    matched = subset(source, [0, 2, 3], "smirk_matched", prepared)

    assert matched.metadata["source_row_indices"] == [0, 2, 3]
    assert matched.splits == {"train": [0], "valid": [1], "test": [2]}
    np.testing.assert_array_equal(matched.X, source.X[[2, 1, 3]])
    assert matched.y.reset_index(drop=True).equals(
        prepared.labels.iloc[[0, 2, 3]].reset_index(drop=True)
    )

    source.splits["test"] = [0]
    with pytest.raises(ValueError, match="Invalid embedded split"):
        verify_source_rows(source, prepared)
