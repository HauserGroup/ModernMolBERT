"""Seed aggregation uses five matched cohorts and keeps seed variation explicit."""

import json

import pandas as pd
import pytest

from aggregate_revision_seeds import MODELS, SEEDS, aggregate, load_inputs
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
            "cv": {"folds": 5, "shuffle": True, "seed": 0},
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
        selection = {
            "task_matrices": {
                "common_task_matrix.csv": {"training_rows_and_cv_folds_verified": True}
            },
            "evaluation_evidence": {"evaluation_manifest_sha256": file_sha256(evaluation_path)},
        }
        (seed_dir / "manifest.json").write_text(json.dumps(selection), encoding="utf-8")
        matrix_paths[seed] = matrix_path
        evaluation_paths[seed] = evaluation_path

    matrices, provenance = load_inputs(matrix_paths, evaluation_paths)
    assert isinstance(provenance["matrices"], dict)
    assert len(provenance["matrices"]) == 5
    outputs = aggregate(matrices)
    mean = outputs["mean_common_task_matrix.csv"]
    assert mean.loc["task_00", MODELS[0]] == pytest.approx(0.62)
    overall = outputs["overall_contrast_summary.csv"].set_index("contrast")
    assert overall.loc["APE_minus_BPE_SELFIES", "mean"] == pytest.approx(0.10)
    assert overall.loc["base_minus_small_APE_SELFIES", "mean"] == pytest.approx(0.05)
    assert overall.loc["representation_by_tokenizer_interaction", "mean"] == pytest.approx(0)
    assert set(overall["n_seeds"].tolist()) == {5}

    changed = json.loads(evaluation_paths[46].read_text(encoding="utf-8"))
    changed["tasks"]["task_00"]["labels_sha256"] = "0" * 64
    evaluation_paths[46].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not share the frozen supervised cohort"):
        load_inputs(matrix_paths, evaluation_paths)
