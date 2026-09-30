import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.praski_export import append_result_row
from modernmolbert.eval.benchmarking_molecular_models.supervised.procedure import (
    check_if_already_evaluated,
)


def _row(head: str = "rf") -> dict:
    return {
        "dataset": "AMES",
        "task": "classification",
        "embedder": "emb",
        "model": head,
        "cv_metric_name": "roc_auc",
        "cv_metric": 0.7,
        "test_metric_name": "roc_auc",
        "test_metric": 0.8,
    }


def _evaluated(csv, head: str = "rf") -> bool:
    return check_if_already_evaluated(csv, "AMES", "emb", "roc_auc", head)


def test_missing_results_file_is_not_evaluated(tmp_path):
    assert not _evaluated(tmp_path / "results.csv")


def test_matching_row_counts_as_evaluated_only_for_its_head(tmp_path):
    csv = tmp_path / "results.csv"
    append_result_row(csv, _row("rf"))

    assert _evaluated(csv, "rf")
    assert not _evaluated(csv, "ridge")


def test_duplicate_rows_are_deleted_and_reevaluated(tmp_path):
    csv = tmp_path / "results.csv"
    append_result_row(csv, _row("rf"))
    append_result_row(csv, _row("rf"))
    append_result_row(csv, _row("ridge"))

    assert not _evaluated(csv, "rf")
    remaining = pd.read_csv(csv)
    assert list(remaining["model"]) == ["ridge"]


def test_unreadable_results_file_is_left_untouched(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text('a,b\n1,"unterminated\n', encoding="utf-8")
    before = csv.read_bytes()

    assert not _evaluated(csv)
    assert csv.read_bytes() == before
