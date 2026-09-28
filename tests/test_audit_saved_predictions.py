import numpy as np
import pandas as pd

from audit_saved_predictions import compare_prediction


def test_saved_prediction_matches_only_the_supported_result_row(tmp_path):
    path = tmp_path / "dataset" / "embedder" / "ridge.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, y_true=np.array([0, 0, 1, 1]), y_score=np.array([0.1, 0.2, 0.8, 0.9]))
    result_rows = pd.DataFrame(
        {"test_metric": [0.6, 1.0], "result_source": ["stale.csv", "matching.csv"]}
    )
    audit = compare_prediction(path, result_rows, prepared_test_labels=np.array([0, 0, 1, 1]))
    assert audit["prediction_test_roc_auc"] == 1.0
    assert audit["n_csv_rows"] == 2
    assert audit["n_matching_csv_rows"] == 1
    assert audit["matching_csv_sources"] == '["matching.csv"]'
    assert audit["n_prediction_rows"] == audit["n_prepared_test_rows"]
    assert audit["labels_match_prepared_test"] is True
