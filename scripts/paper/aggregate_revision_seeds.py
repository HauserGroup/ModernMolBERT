#!/usr/bin/env python3
"""Combine five matched, CV-selected seed matrices without pooling seed replicates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modernmolbert.utils import file_sha256, get_git_revision

SEEDS = (42, 43, 44, 45, 46)
MODELS = (
    "MMB-small-APE-SELFIES",
    "MMB-small-APE-SMILES",
    "MMB-small-BPE-SELFIES",
    "MMB-small-BPE-SMILES",
    "MMB-base-APE-SELFIES",
)
CONTRASTS = {
    "APE_minus_BPE_SELFIES": ((MODELS[0], 1), (MODELS[2], -1)),
    "APE_minus_BPE_SMILES": ((MODELS[1], 1), (MODELS[3], -1)),
    "SELFIES_minus_SMILES_APE": ((MODELS[0], 1), (MODELS[1], -1)),
    "SELFIES_minus_SMILES_BPE": ((MODELS[2], 1), (MODELS[3], -1)),
    "base_minus_small_APE_SELFIES": ((MODELS[4], 1), (MODELS[0], -1)),
    "representation_by_tokenizer_interaction": (
        (MODELS[0], 1),
        (MODELS[2], -1),
        (MODELS[1], -1),
        (MODELS[3], 1),
    ),
}
COHORT_KEYS = (
    "prepared_sha256",
    "common_supervised_rows",
    "common_source_row_indices_sha256",
    "labels_sha256",
    "split_source_row_indices_sha256",
    "splits",
    "endpoint_viability",
)


def seed_paths(values: list[str]) -> dict[int, Path]:
    paths = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text.isdecimal() or not path_text:
            raise ValueError(f"Expected SEED=PATH, got {value!r}")
        seed = int(seed_text)
        if seed in paths:
            raise ValueError(f"Duplicate seed {seed}")
        paths[seed] = Path(path_text)
    if set(paths) != set(SEEDS):
        raise ValueError(f"Expected exactly seeds {SEEDS}")
    return paths


def load_inputs(
    matrix_paths: dict[int, Path], evaluation_paths: dict[int, Path]
) -> tuple[dict[int, pd.DataFrame], dict[str, object]]:
    matrices = {}
    references = None
    tasks = None
    provenance = {"matrices": {}, "evaluation_manifests": {}, "selection_manifests": {}}
    for seed in SEEDS:
        evaluation_path = evaluation_paths[seed]
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("schema") != 2 or evaluation.get("seed", 42) != seed:
            raise ValueError(f"Wrong seed or schema in {evaluation_path}")
        if len(evaluation.get("run_ids", [])) != 5 or len(evaluation.get("tasks", {})) != 25:
            raise ValueError(f"Incomplete evaluation manifest for seed {seed}")
        if evaluation.get("cv") != {"folds": 5, "shuffle": True, "seed": 0}:
            raise ValueError(f"CV policy differs for seed {seed}")
        cohort = {
            task: {key: record[key] for key in COHORT_KEYS}
            for task, record in evaluation["tasks"].items()
        }
        if references is None:
            references = cohort
            tasks = sorted(cohort)
        elif cohort != references:
            raise ValueError(f"Seed {seed} does not share the frozen supervised cohort")

        matrix_path = matrix_paths[seed]
        if matrix_path.name != "common_task_matrix.csv":
            raise ValueError(f"Expected common_task_matrix.csv for seed {seed}")
        selection_path = matrix_path.parent / "manifest.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        matrix_info = selection["task_matrices"]["common_task_matrix.csv"]
        evidence = selection.get("evaluation_evidence") or {}
        if matrix_info.get("training_rows_and_cv_folds_verified") is not True or evidence.get(
            "evaluation_manifest_sha256"
        ) != file_sha256(evaluation_path):
            raise ValueError(f"Unverified selected-head matrix for seed {seed}")
        matrix = pd.read_csv(matrix_path, index_col=0)
        if (
            len(matrix) != 25
            or set(matrix.index) != set(tasks or [])
            or set(matrix.columns) != set(MODELS)
            or matrix.index.has_duplicates
            or matrix.columns.has_duplicates
        ):
            raise ValueError(f"Wrong task/model matrix for seed {seed}")
        matrix = matrix.loc[tasks, list(MODELS)].astype(float)
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Undefined or invalid ROC-AUC for seed {seed}")
        matrices[seed] = matrix
        provenance["matrices"][seed] = {
            "path": str(matrix_path),
            "sha256": file_sha256(matrix_path),
        }
        provenance["evaluation_manifests"][seed] = {
            "path": str(evaluation_path),
            "sha256": file_sha256(evaluation_path),
        }
        provenance["selection_manifests"][seed] = {
            "path": str(selection_path),
            "sha256": file_sha256(selection_path),
        }
    return matrices, provenance


def aggregate(matrices: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    long = pd.concat(
        [
            matrix.stack().rename("roc_auc").reset_index().assign(seed=seed)
            for seed, matrix in matrices.items()
        ],
        ignore_index=True,
    ).rename(columns={"level_0": "dataset", "level_1": "model"})
    summary = (
        long.groupby(["dataset", "model"], sort=True)["roc_auc"]
        .agg(["count", "mean", "std", "min", "max"])
        .rename(columns={"count": "n_seeds", "std": "sd_across_seeds"})
        .reset_index()
    )
    if not summary["n_seeds"].eq(len(SEEDS)).all():
        raise ValueError("Every task/model needs five seed scores")
    mean_matrix = summary.pivot(index="dataset", columns="model", values="mean")
    mean_matrix = mean_matrix.loc[next(iter(matrices.values())).index, list(MODELS)]

    contrasts = []
    for seed, matrix in matrices.items():
        for name, terms in CONTRASTS.items():
            values = sum(weight * matrix[model] for model, weight in terms)
            for dataset, value in values.items():
                contrasts.append(
                    {"seed": seed, "dataset": dataset, "contrast": name, "difference": value}
                )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_summary = (
        contrast_frame.groupby(["dataset", "contrast"], sort=True)["difference"]
        .agg(["count", "mean", "std", "min", "max"])
        .rename(columns={"count": "n_seeds", "std": "sd_across_seeds"})
        .reset_index()
    )
    overall_by_seed = (
        contrast_frame.groupby(["seed", "contrast"], sort=True)["difference"]
        .mean()
        .rename("mean_across_tasks")
        .reset_index()
    )
    overall_summary = (
        overall_by_seed.groupby("contrast", sort=True)["mean_across_tasks"]
        .agg(["count", "mean", "std", "min", "max"])
        .rename(columns={"count": "n_seeds", "std": "sd_across_seeds"})
        .reset_index()
    )
    model_overall = (
        long.groupby(["seed", "model"], sort=True)["roc_auc"]
        .mean()
        .rename("mean_across_tasks")
        .reset_index()
    )
    return {
        "seed_scores.csv": long,
        "task_seed_summary.csv": summary,
        "mean_common_task_matrix.csv": mean_matrix,
        "seed_task_contrasts.csv": contrast_frame,
        "task_contrast_summary.csv": contrast_summary,
        "seed_overall_contrasts.csv": overall_by_seed,
        "overall_contrast_summary.csv": overall_summary,
        "seed_overall_models.csv": model_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-matrix", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument(
        "--evaluation-manifest", action="append", required=True, metavar="SEED=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix_paths = seed_paths(args.seed_matrix)
    evaluation_paths = seed_paths(args.evaluation_manifest)
    matrices, provenance = load_inputs(matrix_paths, evaluation_paths)
    outputs = aggregate(matrices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=filename != "mean_common_task_matrix.csv")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "code": get_git_revision(),
                "seeds": list(SEEDS),
                "models": list(MODELS),
                "task_count": 25,
                "provenance": provenance,
                "summary_policy": (
                    "Arithmetic task means within each seed; then mean, sample SD, min and max "
                    "across five seeds. No molecule/task pseudoreplication for seed variation."
                ),
                "output_sha256": {name: file_sha256(args.output_dir / name) for name in outputs},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote five-seed results to {args.output_dir}")


if __name__ == "__main__":
    main()
