import pytest

from build_revision_dataset_tables import (
    render_dataset_table,
    render_endpoint_table,
    validate,
)


def sample_rows():
    datasets = []
    endpoints = []
    for index in range(25):
        dataset = f"toy_{index}"
        n_endpoints = 4 if index < 4 else 3
        datasets.append(
            {
                "dataset": dataset,
                "split_rule": "OGB scaffold split",
                "raw_train": "4",
                "raw_valid": "1",
                "raw_test": "2",
                "common_train": "3",
                "common_valid": "1",
                "common_test": "2",
                "excluded_train": "1",
                "excluded_valid": "0",
                "excluded_test": "0",
                "common_test_observed_cells": str(2 * n_endpoints),
                "n_test_endpoints": str(n_endpoints),
                "n_scored_test_endpoints": str(n_endpoints),
            }
        )
        for endpoint in range(n_endpoints):
            endpoints.append(
                {
                    "dataset": dataset,
                    "endpoint": f"Y_{endpoint}",
                    "common_test_rows": "2",
                    "common_test_observed": "2",
                    "common_test_missing": "0",
                    "common_test_positive": "1",
                    "roc_auc_defined": "True",
                }
            )
    return datasets, endpoints


def test_verified_counts_render_into_self_contained_tables():
    datasets, endpoints = sample_rows()
    validate(datasets, endpoints)
    summary = render_dataset_table(datasets)
    detail = render_endpoint_table(endpoints)
    assert "toy\\_\\allowbreak{}0 & OGB scaffold & 4/1/2 & 3/1/2 & 1/0/0 & 4/4" in summary
    assert "toy\\_\\allowbreak{}0 & Y\\_\\allowbreak{}0 & 2 & 2 & 0 & 1 & yes" in detail
    assert "missing labels as negative" in detail


def test_rejects_split_or_endpoint_accounting_errors():
    datasets, endpoints = sample_rows()
    datasets[0]["excluded_train"] = "0"
    with pytest.raises(ValueError, match="Split accounting mismatch"):
        validate(datasets, endpoints)
    datasets[0]["excluded_train"] = "1"
    endpoints[0]["common_test_missing"] = "1"
    with pytest.raises(ValueError, match="Missing-label count mismatch"):
        validate(datasets, endpoints)
