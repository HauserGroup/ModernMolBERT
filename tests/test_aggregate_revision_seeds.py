"""Seed aggregation uses five matched cohorts and keeps seed variation explicit."""

import json

import pandas as pd
import pytest

from aggregate_revision_seeds import (
    MODELS,
    SEEDS,
    aggregate,
    aggregate_sensitivity,
    load_inputs,
    write_outputs,
)
from modernmolbert.utils import file_sha256


def test_aggregate_five_seed_matrices_requires_same_cohort(tmp_path):
    tasks = [f"task_{i:02d}" for i in range(25)]
    cohort = {
        "prepared_sha256": "a" * 64,
        "common_supervised_rows": 20,
        "common_source_row_indices_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
        "split_source_row_indices_sha256": {"train": "d" * 64, "valid": "e" * 64, "test": "f" * 64},
        "splits": {"train": 15, "valid": 0, "test": 5},
        "endpoint_viability": {"Y": {"roc_auc_defined": True}},
    }
    matrix_paths = {}
    evaluation_paths = {}
    for seed in SEEDS:
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir()
        evaluation = {
            "schema": 2,
            "seed": seed,
            "run_ids": [f"run{i}" for i in range(5)],
            "cv": {"folds": 5, "shuffle": False, "seed": None},
            "tasks": {task: cohort for task in tasks},
        }
        evaluation_path = seed_dir / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        values = [0.60, 0.55, 0.50, 0.45, 0.65]
        matrix = pd.DataFrame(
            {
                model: value + (seed - 42) * 0.01
                for model, value in zip(MODELS, values, strict=True)
            },
            index=tasks,
        )
        matrix_path = seed_dir / "common_task_matrix.csv"
        matrix.to_csv(matrix_path)
        sensitivity_path = seed_dir / "common_task_matrix_no_split_overlap.csv"
        (matrix - 0.01).to_csv(sensitivity_path)
        status_path = seed_dir / "common_task_matrix_no_split_overlap_status.csv"
        pd.DataFrame(
            {"dataset": tasks, "status": ["ok"] * 25, "n_common_test_rows": [5] * 25}
        ).to_csv(status_path, index=False)
        selection = {
            "task_matrices": {
                "common_task_matrix.csv": {"training_rows_and_cv_folds_verified": True},
                "common_task_matrix_no_split_overlap.csv": {
                    "training_rows_and_cv_folds_verified": True
                },
            },
            "evaluation_evidence": {"evaluation_manifest_sha256": file_sha256(evaluation_path)},
            "output_sha256": {
                "common_task_matrix.csv": file_sha256(matrix_path),
                "common_task_matrix_no_split_overlap.csv": file_sha256(sensitivity_path),
                "common_task_matrix_no_split_overlap_status.csv": file_sha256(status_path),
            },
            "split_overlap_sensitivity": {"audit_sha256": "1" * 64},
        }
        (seed_dir / "manifest.json").write_text(json.dumps(selection), encoding="utf-8")
        matrix_paths[seed] = matrix_path
        evaluation_paths[seed] = evaluation_path

    matrices, sensitivity_matrices, provenance = load_inputs(matrix_paths, evaluation_paths)
    assert isinstance(provenance["matrices"], dict)
    assert len(provenance["matrices"]) == 5
    assert isinstance(provenance["sensitivity_matrices"], dict)
    assert len(provenance["sensitivity_matrices"]) == 5
    outputs = aggregate(matrices)
    sensitivity_outputs = aggregate_sensitivity(matrices, sensitivity_matrices)
    sensitivity_mean = sensitivity_outputs["mean_sensitivity_task_matrix.csv"]
    assert sensitivity_mean.loc["task_00", MODELS[0]] == pytest.approx(0.61)
    by_seed = sensitivity_outputs["sensitivity_seed_overall_models.csv"]
    assert by_seed["difference"].to_numpy() == pytest.approx([-0.01] * 25)
    mean = outputs["mean_common_task_matrix.csv"]
    assert mean.loc["task_00", MODELS[0]] == pytest.approx(0.62)
    overall = outputs["overall_contrast_summary.csv"].set_index("contrast")
    assert overall.loc["APE_minus_BPE_SELFIES", "mean"] == pytest.approx(0.10)
    assert overall.loc["base_minus_small_APE_SELFIES", "mean"] == pytest.approx(0.05)
    assert overall.loc["representation_by_tokenizer_interaction", "mean"] == pytest.approx(0)
    assert set(overall["n_seeds"].tolist()) == {5}

    output_dir = tmp_path / "aggregate"
    write_outputs(output_dir, outputs | sensitivity_outputs)
    for name in ("mean_common_task_matrix.csv", "mean_sensitivity_task_matrix.csv"):
        reloaded = pd.read_csv(output_dir / name, index_col=0)
        assert reloaded.index.tolist() == tasks
        assert reloaded.columns.tolist() == list(MODELS)
    seed_summary = pd.read_csv(output_dir / "overall_contrast_summary.csv")
    assert "Unnamed: 0" not in seed_summary

    changed = json.loads(evaluation_paths[46].read_text(encoding="utf-8"))
    changed["tasks"]["task_00"]["labels_sha256"] = "0" * 64
    evaluation_paths[46].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not share the frozen supervised cohort"):
        load_inputs(matrix_paths, evaluation_paths)

    changed["tasks"]["task_00"]["labels_sha256"] = "c" * 64
    evaluation_paths[46].write_text(json.dumps(changed), encoding="utf-8")
    selection_path = matrix_paths[46].parent / "manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["evaluation_evidence"]["evaluation_manifest_sha256"] = file_sha256(
        evaluation_paths[46]
    )
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    matrix_paths[46].write_text(matrix_paths[46].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matrix hash differs"):
        load_inputs(matrix_paths, evaluation_paths)

    matrix_paths[46].write_text(matrix_paths[46].read_text().rstrip() + "\n", encoding="utf-8")
    selection["output_sha256"]["common_task_matrix.csv"] = file_sha256(matrix_paths[46])
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    sensitivity_path = matrix_paths[46].parent / "common_task_matrix_no_split_overlap.csv"
    sensitivity_path.write_text(sensitivity_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Overlap sensitivity hash differs"):
        load_inputs(matrix_paths, evaluation_paths)
