import numpy as np
import pandas as pd

from make_dataset_summary import main, summarise, summarise_test_endpoints
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset

CONFIG = """
datasets:
  clf_toy:
    name: toy
    n_samples: 6
    source:
      name: OGB
  clf_absent:
    name: absent
    source:
      name: TDC
      benchmark: admet
"""


def test_counts_labels_by_split_and_flags_unassigned_rows(tmp_path):
    config = tmp_path / "datasets.yaml"
    config.write_text(CONFIG)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = pd.DataFrame(
        {
            "smiles": ["C"] * 6,
            "a": [0, 1, 1, np.nan, 1, 0],
            "b": [1, 1, 0, 0, np.nan, np.nan],
        }
    )
    Dataset(
        name="toy",
        task="classification",
        data=data,
        splits={"train": [0, 1], "valid": [2], "test": [3, 4]},
    ).serialize_legacy(prepared / "toy.json")

    out = tmp_path / "summary.csv"
    endpoint_out = tmp_path / "endpoints.csv"
    main(
        [
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--output",
            str(out),
            "--endpoint-output",
            str(endpoint_out),
        ]
    )
    summary = pd.read_csv(out).set_index("dataset")

    toy = summary.loc["toy"]
    assert toy["split_rule"] == "OGB scaffold split"
    assert toy["n_endpoints"] == 2
    assert (toy["n_train"], toy["n_valid"], toy["n_test"]) == (2, 1, 2)
    assert toy["n_unassigned"] == 1
    # Test rows 3 and 4: labels (nan, 0) and (1, nan).
    assert toy["test_labelled_cells"] == 2
    assert toy["test_positive_cells"] == 1
    assert toy["test_missing_cells"] == 2
    assert toy["test_prevalence"] == 0.5
    assert toy["prepared_test_evaluable_endpoints"] == 0
    endpoints = pd.read_csv(endpoint_out)
    assert len(endpoints) == 2
    assert endpoints["prepared_test_roc_auc_evaluable"].tolist() == [False, False]
    assert np.isnan(summary.loc["absent", "n_prepared"])


def test_prepared_test_endpoint_eligibility_uses_both_classes():
    data = pd.DataFrame(
        {
            "smiles": ["C"] * 5,
            "balanced": [0, 1, 0, 1, np.nan],
            "positive_only": [1, 1, np.nan, 1, 1],
            "unlabelled": [np.nan] * 5,
        }
    )
    dataset = Dataset(
        name="toy",
        task="classification",
        data=data,
        splits={"train": [0], "valid": [], "test": [1, 2, 3, 4]},
    )
    summary = summarise(dataset)
    assert summary["prepared_test_evaluable_endpoints"] == 1
    assert summary["prepared_test_min_positive_per_evaluable_endpoint"] == 2
    assert summary["prepared_test_min_negative_per_evaluable_endpoint"] == 1
    endpoint_rows = {row["endpoint"]: row for row in summarise_test_endpoints(dataset)}
    assert endpoint_rows["balanced"]["prepared_test_roc_auc_evaluable"] is True
    assert endpoint_rows["positive_only"]["prepared_test_roc_auc_evaluable"] is False
    assert endpoint_rows["unlabelled"]["prepared_test_roc_auc_evaluable"] is False
