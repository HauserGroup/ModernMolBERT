"""Verify five SMIRK evaluation seeds and summarize matched selected-head ROC-AUC."""

import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.common.types import EmbeddedDataset
from modernmolbert.eval.benchmarking_molecular_models.score import score_row_is_complete
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    get_skfp_roc_auc,
)
from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "outputs/experimental_smirk_v1"
SEEDS = (42, 43, 44, 45, 46)
RUN_IDS = (
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
)


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def accepted_reference_path(seed: int) -> Path:
    if seed == 42:
        return ROOT / "outputs/revision_factorial_v1/evaluation_manifest.json"
    return ROOT / f"outputs/revision_factorial_multiseed_v1/evaluation_seed{seed}.json"


def accepted_score_path(seed: int, task: str, model: str) -> Path:
    if seed == 42:
        base = ROOT / "outputs/eval/revision_factorial_v1/common_rows"
    else:
        base = ROOT / f"outputs/eval/revision_factorial_multiseed_v1/seed{seed}/common_rows"
    return base / model / f"{task}.csv"


def pilot_score_path(seed: int, task: str, model: str) -> Path:
    return PILOT / "scoring" / f"seed{seed}" / model / f"{task}.csv"


def selected_head(
    *,
    seed: int,
    task: str,
    model: str,
    embedder: str,
    embedding_sha: str,
    prepared_sha: str,
    expected_test_rows: list[int],
    score_path: Path,
) -> dict:
    embedding_path = ROOT / "data/embedded" / task / f"{embedder}.joblib"
    if file_sha256(embedding_path) != embedding_sha:
        raise ValueError(f"Embedding changed since cohort materialization: {embedding_path}")
    embedding: EmbeddedDataset = joblib.load(embedding_path)
    source_rows = list(map(int, embedding.metadata["source_row_indices"]))
    test_rows = [source_rows[int(i)] for i in embedding.splits["test"]]
    if (
        test_rows != expected_test_rows
        or embedding.metadata.get("prepared_data_sha256") != prepared_sha
        or embedding.embedder != embedder
    ):
        raise ValueError(f"Score input does not use the matched test cohort: {embedding_path}")
    candidates = pd.read_csv(score_path)
    expected_heads = (
        {"rf", "ridge"}
        if "hiv" in task.lower() or "muv" in task.lower()
        else {
            "rf",
            "ridge",
            "knn",
        }
    )
    if (
        len(candidates) != len(expected_heads)
        or set(candidates["model"]) != expected_heads
        or not np.all(candidates["dataset"].to_numpy() == task)
        or not np.all(candidates["embedder"].to_numpy() == embedder)
        or not np.all(candidates["prepared_data_sha256"].to_numpy() == prepared_sha)
        or not np.all(candidates["missing_labels"].to_numpy() == "as-negative")
        or not np.all(candidates["cv_metric_name"].to_numpy() == "roc_auc")
        or not np.all(candidates["test_metric_name"].to_numpy() == "roc_auc")
    ):
        raise ValueError(f"Incomplete or mismatched head candidates: {score_path}")
    for row in candidates.to_dict("records"):
        head = str(row["model"])
        prediction = ROOT / "data/predictions" / task / embedder / f"{head}.npz"
        identity = str(row["scoring_identity"])
        if not score_row_is_complete(score_path, task, embedder, head, identity, prediction):
            raise ValueError(f"Missing score prediction archive: {prediction}")
        with np.load(prediction, allow_pickle=False) as archive:
            if (
                "test_source_row_indices" not in archive
                or archive["test_source_row_indices"].tolist() != expected_test_rows
                or str(archive["prepared_data_sha256"]) != prepared_sha
            ):
                raise ValueError(f"Prediction row provenance differs: {prediction}")
            recomputed = get_skfp_roc_auc(archive["y_score"], archive["y_true"])
        if (
            not np.isfinite(float(row["cv_metric"]))
            or not 0 <= float(row["cv_metric"]) <= 1
            or not np.isfinite(float(row["test_metric"]))
            or not np.isclose(recomputed, float(row["test_metric"]), atol=1e-8, rtol=0)
        ):
            raise ValueError(f"Nonfinite or nonreproducible ROC-AUC: {score_path}/{head}")
    selected = candidates.sort_values(
        ["cv_metric", "model"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return {
        "seed": seed,
        "dataset": task,
        "model": model,
        "embedder": embedder,
        "selected_head": str(selected["model"]),
        "cv_roc_auc": float(selected["cv_metric"]),
        "test_roc_auc": float(selected["test_metric"]),
        "scoring_identity": str(selected["scoring_identity"]),
        "score_csv_sha256": file_sha256(score_path),
        "embedding_sha256": embedding_sha,
        "matched_test_rows": len(expected_test_rows),
    }


def main() -> None:
    records = []
    cohorts = None
    for seed in SEEDS:
        evaluation_path = PILOT / f"evaluation_seed{seed}.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        reference_path = accepted_reference_path(seed)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if (
            evaluation.get("schema") != 1
            or evaluation.get("seed") != seed
            or evaluation.get("accepted_reference_evaluation_sha256") != file_sha256(reference_path)
            or set(evaluation.get("tasks", {})) != set(reference.get("tasks", {}))
            or len(evaluation["tasks"]) != 25
        ):
            raise ValueError(f"Incomplete or changed SMIRK seed {seed} evaluation manifest")
        seed_cohorts = {}
        for task, item in sorted(evaluation["tasks"].items()):
            prepared_sha = item["prepared_sha256"]
            reference_item = reference["tasks"][task]
            if prepared_sha != reference_item["prepared_sha256"]:
                raise ValueError(f"Prepared dataset differs from accepted cohort: {task}")
            smirk_name = f"SMIRK_COMMON_s{seed}_small_smirk_smiles"
            smirk_path = ROOT / "data/embedded" / task / f"{smirk_name}.joblib"
            if file_sha256(smirk_path) != item["common_embedding_sha256"]:
                raise ValueError(f"SMIRK common embedding changed: {smirk_path}")
            smirk: EmbeddedDataset = joblib.load(smirk_path)
            matched_rows = list(map(int, smirk.metadata["source_row_indices"]))
            if (
                len(matched_rows) != item["matched_rows"]
                or index_hash(matched_rows) != item["matched_source_row_indices_sha256"]
            ):
                raise ValueError(f"SMIRK matched row count changed: {task}")
            expected_test = [matched_rows[int(i)] for i in smirk.splits["test"]]
            seed_cohorts[task] = (len(matched_rows), item["matched_source_row_indices_sha256"])
            records.append(
                selected_head(
                    seed=seed,
                    task=task,
                    model="smirk",
                    embedder=smirk_name,
                    embedding_sha=item["common_embedding_sha256"],
                    prepared_sha=prepared_sha,
                    expected_test_rows=expected_test,
                    score_path=pilot_score_path(seed, task, "smirk"),
                )
            )
            for run_id in RUN_IDS:
                if run_id in item["comparators_rescored"]:
                    embedder = f"SMIRK_MATCHED_s{seed}_{run_id}"
                    embedding_sha = item["comparators_rescored"][run_id]["matched_embedding_sha256"]
                    score_path = pilot_score_path(seed, task, run_id)
                else:
                    embedder = (
                        f"REVISION_COMMON_{run_id}"
                        if seed == 42
                        else f"REVISION_COMMON_s{seed}_{run_id}"
                    )
                    embedding_sha = reference_item["models"][run_id]["common_embedding_sha256"]
                    score_path = accepted_score_path(seed, task, run_id)
                records.append(
                    selected_head(
                        seed=seed,
                        task=task,
                        model=run_id,
                        embedder=embedder,
                        embedding_sha=embedding_sha,
                        prepared_sha=prepared_sha,
                        expected_test_rows=expected_test,
                        score_path=score_path,
                    )
                )
        if cohorts is None:
            cohorts = seed_cohorts
        elif cohorts != seed_cohorts:
            raise ValueError(f"SMIRK matched cohorts differ across seeds: {seed}")
    selected = pd.DataFrame.from_records(records)
    if len(selected) != 5 * 25 * 6:
        raise ValueError("Expected exactly five seeds, 25 tasks and six models")
    cell_scores = {
        (str(record["dataset"]), str(record["model"]), int(record["seed"])): float(
            record["test_roc_auc"]
        )
        for record in records
    }
    if len(cell_scores) != len(records):
        raise ValueError("Duplicate seed/task/model score cell")
    selected_path = PILOT / "selected_heads.csv"
    selected.to_csv(selected_path, index=False)
    tasks = sorted({str(record["dataset"]) for record in records})
    task_mean = pd.DataFrame(
        {
            model: [
                float(np.mean([cell_scores[task, model, seed] for seed in SEEDS])) for task in tasks
            ]
            for model in ("smirk", *RUN_IDS)
        },
        index=tasks,
    )
    task_mean_path = PILOT / "mean_task_matrix.csv"
    task_mean.to_csv(task_mean_path)
    overall = {
        model: float(np.mean(task_mean[model].to_numpy(dtype=float)))
        for model in ("smirk", *RUN_IDS)
    }
    contrast = {
        run_id: float(
            np.mean(
                task_mean["smirk"].to_numpy(dtype=float) - task_mean[run_id].to_numpy(dtype=float)
            )
        )
        for run_id in RUN_IDS
    }
    if cohorts is None:
        raise ValueError("No matched cohorts were loaded")
    summary = {
        "schema": 1,
        "seeds": list(SEEDS),
        "tasks": 25,
        "models": ["smirk", *RUN_IDS],
        "matched_cohorts": {task: rows for task, (rows, _) in sorted(cohorts.items())},
        "macro_mean_roc_auc": overall,
        "smirk_minus_comparator_macro_mean_roc_auc": contrast,
        "selected_heads_sha256": file_sha256(selected_path),
        "mean_task_matrix_sha256": file_sha256(task_mean_path),
        "evaluation_manifest_sha256": {
            str(seed): file_sha256(PILOT / f"evaluation_seed{seed}.json") for seed in SEEDS
        },
    }
    output = PILOT / "comparison_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Verified 750 selected-head cells across five matched seed cohorts: {output}")


if __name__ == "__main__":
    main()
