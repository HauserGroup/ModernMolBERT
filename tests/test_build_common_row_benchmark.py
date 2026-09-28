import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from build_common_row_benchmark import main, multioutput_average_precision
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.utils import file_sha256

LABELS = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
TEST_ROWS = [4, 5, 6, 7]
SCORES = {
    ("A", "rf"): {4: 0.2, 5: 0.9, 6: 0.6, 7: 0.7},
    ("B", "ridge"): {5: 0.8, 6: 0.3, 7: 0.4},
}


def _write_inputs(tmp_path, *, overrides=None, drop_row_ids=()):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = pd.DataFrame({"smiles": ["C"] * len(LABELS), "y": LABELS})
    Dataset(
        name="toy",
        task="classification",
        data=data,
        splits={"train": [0, 1, 2, 3], "valid": [], "test": TEST_ROWS},
    ).serialize_legacy(prepared / "toy.json")

    predictions = tmp_path / "predictions"
    rows = []
    for (embedder, head), by_row in SCORES.items():
        source_rows = np.array(sorted(by_row))
        y_true = LABELS[source_rows]
        y_score = np.array([by_row[r] for r in source_rows])
        path = predictions / "toy" / embedder / f"{head}.npz"
        path.parent.mkdir(parents=True)
        if embedder in drop_row_ids:
            np.savez(path, y_true=y_true, y_score=y_score)
        else:
            np.savez(path, y_true=y_true, y_score=y_score, test_source_row_indices=source_rows)
        rows.append((embedder, head, 0.9, float(roc_auc_score(y_true, y_score))))
    # Unselected heads: lower CV but higher test score, and no archive.
    rows += [("A", "ridge", 0.8, 0.99), ("B", "rf", 0.7, 0.99)]
    results = pd.DataFrame(rows, columns=["embedder", "model", "cv_metric", "test_metric"]).assign(
        dataset="toy", cv_metric_name="roc_auc", test_metric_name="roc_auc"
    )
    for key, value in (overrides or {}).items():
        results.loc[results["embedder"].eq(key[0]) & results["model"].eq(key[1]), "test_metric"] = (
            value
        )
    results_path = tmp_path / "results.csv"
    results.to_csv(results_path, index=False)
    return results_path, predictions, prepared


def _run(tmp_path, **kwargs):
    results_path, predictions, prepared = _write_inputs(tmp_path, **kwargs)
    out = tmp_path / "out"
    main(
        [
            "--results",
            str(results_path),
            "--predictions-dir",
            str(predictions),
            "--prepared-dir",
            str(prepared),
            "--output-dir",
            str(out),
        ]
    )
    selected = pd.read_csv(out / "selected_heads.csv").set_index("embedder")
    common = pd.read_csv(out / "common_row_scores.csv")
    return selected, common, out


def test_heads_selected_by_cv_and_scored_on_common_rows(tmp_path):
    selected, common, out = _run(tmp_path)
    assert selected.loc["A", "model"] == "rf"
    assert selected.loc["B", "model"] == "ridge"
    assert set(selected["archive_status"]) == {"ok"}
    assert selected.loc["A", "test_coverage"] == 1.0
    assert selected.loc["B", "test_coverage"] == 0.75

    common = common.set_index("embedder")
    assert set(common["n_common_test_rows"]) == {3}
    shared = [5, 6, 7]
    a_scores = [SCORES[("A", "rf")][r] for r in shared]
    assert common.loc["A", "roc_auc_common"] == pytest.approx(
        roc_auc_score(LABELS[shared], a_scores)
    )
    assert common.loc["A", "average_precision_common"] == pytest.approx(
        average_precision_score(LABELS[shared], a_scores)
    )
    assert common.loc["A", "n_positive_common"] == 2

    candidates = pd.read_csv(out / "head_candidates.csv")
    assert candidates["selected"].sum() == 2


def test_score_mismatch_excludes_model_from_common_rows(tmp_path):
    selected, common, _ = _run(tmp_path, overrides={("B", "ridge"): 0.5})
    assert selected.loc["B", "archive_status"] == "score_mismatch"
    assert list(common["embedder"]) == ["A"]
    assert common["n_common_test_rows"].item() == 4


def test_archive_without_row_ids_is_reported_not_compared(tmp_path):
    selected, common, _ = _run(tmp_path, drop_row_ids=("B",))
    assert selected.loc["B", "archive_status"] == "no_row_ids"
    assert list(common["embedder"]) == ["A"]


def test_inputs_are_not_modified(tmp_path):
    results_path, predictions, prepared = _write_inputs(tmp_path)
    inputs = [results_path, *sorted(predictions.rglob("*.npz")), prepared / "toy.json"]
    before = [file_sha256(p) for p in inputs]
    main(
        [
            "--results",
            str(results_path),
            "--predictions-dir",
            str(predictions),
            "--prepared-dir",
            str(prepared),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert [file_sha256(p) for p in inputs] == before


def test_average_precision_skips_unusable_endpoints():
    y_true = np.array([[0, 1, np.nan], [1, 1, 0], [0, 1, np.nan], [1, 1, 1]])
    y_score = np.array([[0.1, 0.5, 0.2], [0.9, 0.4, 0.3], [0.2, 0.6, 0.1], [0.8, 0.7, 0.9]])
    # Endpoint 2 has one class only; endpoint 3 keeps two finite rows.
    expected = (1.0 + float(average_precision_score([0, 1], [0.3, 0.9]))) / 2
    assert multioutput_average_precision(y_true, y_score) == pytest.approx(expected)
