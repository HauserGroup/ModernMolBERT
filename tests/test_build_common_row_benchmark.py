import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from build_common_row_benchmark import (
    common_task_matrix,
    main,
    paired_task_differences,
    rank_roc_auc,
    score_rows,
)
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.utils import file_sha256

LABELS = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
TEST_ROWS = [4, 5, 6, 7]
SCORES = {
    # The high-scoring negative is absent from B: A's native and common AUCs differ.
    ("A", "rf"): {4: 0.95, 5: 0.9, 6: 0.6, 7: 0.7},
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
    prepared_hash = file_sha256(prepared / "toy.json")

    predictions = tmp_path / "predictions"
    rows = []
    for (embedder, head), by_row in SCORES.items():
        source_rows = np.array(sorted(by_row))
        y_true = LABELS[source_rows]
        y_score = np.array([by_row[r] for r in source_rows])
        path = predictions / "toy" / embedder / f"{head}.npz"
        path.parent.mkdir(parents=True)
        if embedder in drop_row_ids:
            np.savez(path, y_true=y_true, y_score=y_score, prepared_data_sha256=prepared_hash)
        else:
            np.savez(
                path,
                y_true=y_true,
                y_score=y_score,
                test_source_row_indices=source_rows,
                prepared_data_sha256=prepared_hash,
            )
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
    assert common.loc["A", "n_scored_endpoints_common"] == 1
    assert common.loc["A", "min_positive_per_scored_endpoint_common"] == 2

    candidates = pd.read_csv(out / "head_candidates.csv")
    assert candidates["selected"].sum() == 2
    native = pd.read_csv(out / "task_matrix.csv", index_col=0)
    matrix = pd.read_csv(out / "common_task_matrix.csv", index_col=0)
    assert native.loc["toy", "A"] == 0.5
    assert matrix.loc["toy", "A"] == 1.0
    assert matrix.loc["toy", "B"] == common.loc["B", "roc_auc_common"]
    status = pd.read_csv(out / "common_task_matrix_status.csv")
    assert status["status"].item() == "ok"
    manifest = json.loads((out / "manifest.json").read_text())
    details = manifest["task_matrices"]["common_task_matrix.csv"]
    assert details["embedders"] == ["A", "B"]
    assert details["training_rows_and_cv_folds_verified"] is False


def test_score_mismatch_excludes_model_from_common_rows(tmp_path):
    selected, common, out = _run(tmp_path, overrides={("B", "ridge"): 0.5})
    assert selected.loc["B", "archive_status"] == "score_mismatch"
    assert list(common["embedder"]) == ["A"]
    assert common["n_common_test_rows"].item() == 4
    matrix = pd.read_csv(out / "common_task_matrix.csv", index_col=0)
    assert matrix.loc["toy"].isna().all()  # Do not report A on a different population.
    status = pd.read_csv(out / "common_task_matrix_status.csv")
    assert status["status"].item() == "incomplete_cohort"
    assert status["missing_models"].item() == "B"


def test_archive_from_different_prepared_file_is_excluded(tmp_path):
    results_path, predictions, prepared = _write_inputs(tmp_path)
    path = predictions / "toy/B/ridge.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["prepared_data_sha256"] = np.asarray("0" * 64)
    np.savez(path, **arrays)
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
    assert selected.loc["B", "archive_status"] == "prepared_hash_mismatch"
    assert list(pd.read_csv(out / "common_row_scores.csv")["embedder"]) == ["A"]


def test_common_row_score_handles_a_single_class_after_exclusion():
    scores = score_rows(np.array([1, 1]), np.array([0.4, 0.8]))
    assert np.isnan(scores["roc_auc_common"])
    assert scores["n_scored_endpoints_common"] == 0
    assert np.isnan(scores["min_positive_per_scored_endpoint_common"])


def test_split_overlap_sensitivity_excludes_same_rows_for_every_model(tmp_path):
    results_path, predictions, prepared = _write_inputs(tmp_path)
    prepared_hash = file_sha256(prepared / "toy.json")
    audit = tmp_path / "split_overlap.csv"
    pd.DataFrame(
        {
            "dataset": ["toy"] * 4,
            "prepared_sha256": [prepared_hash] * 4,
            "test_source_row_index": TEST_ROWS,
            "same_inchikey": [False, True, False, False],
            "same_nonisomeric_smiles": [False] * 4,
        }
    ).to_csv(audit, index=False)
    out = tmp_path / "out"
    main(
        [
            "--results",
            str(results_path),
            "--predictions-dir",
            str(predictions),
            "--prepared-dir",
            str(prepared),
            "--split-overlap-rows",
            str(audit),
            "--output-dir",
            str(out),
        ]
    )
    primary = pd.read_csv(out / "common_row_scores.csv")
    sensitivity = pd.read_csv(out / "common_row_scores_no_split_overlap.csv")
    assert set(primary["n_common_test_rows"]) == {3}
    assert set(sensitivity["n_common_test_rows"]) == {2}
    assert set(sensitivity["n_scored_endpoints_common"]) == {1}
    matrix = pd.read_csv(out / "common_task_matrix_no_split_overlap.csv", index_col=0)
    for row in sensitivity.to_dict("records"):
        assert matrix.loc[row["dataset"], row["embedder"]] == row["roc_auc_common"]
    status = pd.read_csv(out / "common_task_matrix_no_split_overlap_status.csv")
    assert status["status"].item() == "ok"
    assert status["n_common_test_rows"].item() == 2

    bad_audit = pd.read_csv(audit)
    bad_audit.loc[0, "prepared_sha256"] = "0" * 64
    bad_audit.to_csv(tmp_path / "bad_overlap.csv", index=False)
    with pytest.raises(ValueError, match="prepared-file hash mismatch"):
        main(
            [
                "--results",
                str(results_path),
                "--predictions-dir",
                str(predictions),
                "--prepared-dir",
                str(prepared),
                "--split-overlap-rows",
                str(tmp_path / "bad_overlap.csv"),
                "--output-dir",
                str(tmp_path / "bad_out"),
            ]
        )


def test_rank_roc_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200).astype(float)
    scores = np.round(rng.random(200), 1)
    assert rank_roc_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def _paired_archives(tmp_path, n=200):
    rng = np.random.default_rng(1)
    labels = np.tile([0.0, 1.0], n // 2)
    rows = np.arange(n)
    scores = {
        "good": labels * 0.6 + rng.random(n) * 0.5,
        "noisy": rng.random(n),
        "twin": labels * 0.6 + rng.random(n) * 0.5,
    }
    records = []
    for embedder, y_score in scores.items():
        path = tmp_path / f"{embedder}.npz"
        np.savez(path, y_true=labels, y_score=y_score, test_source_row_indices=rows)
        records.append(
            {
                "dataset": "toy",
                "embedder": embedder,
                "prediction_path": str(path),
                "archive_status": "ok",
            }
        )
    return pd.DataFrame(records)


def test_paired_bootstrap_names_a_winner_only_when_clear(tmp_path):
    selected = _paired_archives(tmp_path)
    paired = paired_task_differences(selected, n_boot=300, seed=0).set_index(
        ["embedder_a", "embedder_b"]
    )
    assert len(paired) == 3
    good_vs_noisy = paired.loc[("good", "noisy")]
    assert good_vs_noisy["ci_low"] > 0
    assert good_vs_noisy["clear_winner"] == "good"
    assert paired.loc[("good", "twin"), "clear_winner"] == "neither"
    assert (paired["n_valid_boot"] == 300).all()
    assert (paired["ci_low"] <= paired["roc_auc_difference"]).all()


def test_main_writes_paired_task_differences(tmp_path):
    _, _, out = _run(tmp_path)
    paired = pd.read_csv(out / "paired_task_differences.csv")
    assert list(paired[["embedder_a", "embedder_b"]].iloc[0]) == ["A", "B"]
    assert paired["n_common_test_rows"].iloc[0] == 3


def _write_table(tmp_path, rows):
    table = pd.DataFrame(rows, columns=["embedder", "model", "cv_metric", "test_metric"]).assign(
        dataset="toy", cv_metric_name="roc_auc", test_metric_name="roc_auc"
    )
    # As in the Praski table, heads of one model can come from different scoring batches.
    table["library_hash"] = range(len(table))
    path = tmp_path / "table.csv"
    table.to_csv(path, index=False)
    return path


def test_table_baselines_are_cv_selected_but_never_row_compared(tmp_path):
    results_path, predictions, prepared = _write_inputs(tmp_path)
    # A knn head with the best CV for A: excluded because the table model lacks knn.
    results = pd.read_csv(results_path)
    extra = results.iloc[[0]].assign(model="knn", cv_metric=0.99, test_metric=0.5)
    pd.concat([results, extra]).to_csv(results_path, index=False)
    table_path = _write_table(
        tmp_path,
        [
            ("T", "rf", 0.95, 0.70),
            ("T", "ridge", 0.60, 0.99),
            ("Unused", "rf", 0.99, 0.99),
        ],
    )
    out = tmp_path / "out"
    main(
        [
            "--results",
            str(results_path),
            "--table-results",
            str(table_path),
            "--table-embedders",
            "T",
            "--matrix-labels",
            "T=Table model",
            "A=Reference",
            "--predictions-dir",
            str(predictions),
            "--prepared-dir",
            str(prepared),
            "--output-dir",
            str(out),
            "--n-boot",
            "20",
        ]
    )
    selected = pd.read_csv(out / "selected_heads.csv").set_index("embedder")
    assert selected.loc["T", "model"] == "rf"
    assert selected.loc["T", "archive_status"] == "table_only"
    assert "Unused" not in selected.index
    assert selected.loc["A", "model"] == "rf"  # knn not offered for every model

    common = pd.read_csv(out / "common_row_scores.csv")
    assert set(common["embedder"]) == {"A", "B"}
    paired = pd.read_csv(out / "paired_task_differences.csv")
    assert "T" not in set(paired["embedder_a"]) | set(paired["embedder_b"])

    matrix = pd.read_csv(out / "task_matrix.csv", index_col=0)
    assert matrix.loc["toy", "Table model"] == 0.70
    assert matrix.loc["toy", "Reference"] == selected.loc["A", "test_metric"]
    common_matrix = pd.read_csv(out / "common_task_matrix.csv", index_col=0)
    assert set(common_matrix.columns) == {"Reference", "B"}
    assert common_matrix.loc["toy", "Reference"] == 1.0


def test_no_verified_archives_still_writes_missing_common_matrix(tmp_path):
    _, _, out = _run(tmp_path, drop_row_ids=("A", "B"))
    matrix = pd.read_csv(out / "common_task_matrix.csv", index_col=0)
    assert list(matrix.columns) == ["A", "B"]
    assert matrix.loc["toy"].isna().all()
    status = pd.read_csv(out / "common_task_matrix_status.csv")
    assert status["missing_models"].item() == "A;B"


@pytest.mark.parametrize(
    ("n_rows", "auc", "expected_status"),
    [(0, None, "no_common_rows"), (3, np.nan, "undefined_roc_auc")],
)
def test_common_matrix_keeps_unscorable_dataset_missing(n_rows, auc, expected_status):
    scores = pd.DataFrame(
        {
            "dataset": ["toy"] * 2,
            "embedder": ["A", "B"],
            "n_common_test_rows": [n_rows] * 2,
            "n_models_compared": [2] * 2,
        }
    )
    if auc is not None:
        scores["roc_auc_common"] = auc
    matrix, status = common_task_matrix(scores, ["toy"], ["A", "B"], {})
    assert matrix.loc["toy"].isna().all()
    assert status["status"].item() == expected_status


def test_common_matrix_does_not_change_cohort_for_missing_dataset():
    scores = pd.DataFrame(
        {
            "dataset": ["toy"],
            "embedder": ["A"],
            "n_common_test_rows": [4],
            "n_models_compared": [1],
            "roc_auc_common": [0.9],
        }
    )
    matrix, status = common_task_matrix(scores, ["toy", "absent"], ["A", "B"], {})
    assert matrix.isna().all().all()
    assert status.set_index("dataset").loc["toy", "missing_models"] == "B"
    assert status.set_index("dataset").loc["absent", "missing_models"] == "A;B"


def test_common_matrix_rejects_ambiguous_labels():
    with pytest.raises(ValueError, match="column labels must be unique"):
        common_task_matrix(pd.DataFrame(), ["toy"], ["A", "B"], {"A": "B"})


def test_requested_missing_model_cannot_silently_shrink_cohort(tmp_path):
    results, _, _ = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="Requested embedders not found"):
        main(
            [
                "--results",
                str(results),
                "--embedders",
                "A",
                "Missing",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
