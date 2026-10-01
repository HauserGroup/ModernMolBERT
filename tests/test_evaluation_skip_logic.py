import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models.praski_export import append_result_row


def _row(head: str = "rf", identity: str = "first") -> dict:
    return {
        "dataset": "AMES",
        "task": "classification",
        "embedder": "emb",
        "model": head,
        "scoring_identity": identity,
        "cv_metric_name": "roc_auc",
        "cv_metric": 0.7,
        "test_metric_name": "roc_auc",
        "test_metric": 0.8,
    }


def test_completed_scoring_replaces_only_its_result_row(tmp_path) -> None:
    csv = tmp_path / "results.csv"
    append_result_row(csv, _row("rf"))
    append_result_row(csv, _row("ridge"))

    append_result_row(csv, _row("rf", "second"), replace_existing=True)

    rows = pd.read_csv(csv)
    assert len(rows) == 2
    assert rows.loc[rows["model"] == "rf", "scoring_identity"].item() == "second"
    assert rows.loc[rows["model"] == "ridge", "scoring_identity"].item() == "first"


def test_corrupt_results_file_is_not_replaced(tmp_path) -> None:
    csv = tmp_path / "results.csv"
    csv.write_text('a,b\n1,"unterminated\n', encoding="utf-8")
    before = csv.read_bytes()

    with pytest.raises(pd.errors.ParserError):
        append_result_row(csv, _row(), replace_existing=True)
    assert csv.read_bytes() == before
