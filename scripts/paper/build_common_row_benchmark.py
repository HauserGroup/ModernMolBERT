#!/usr/bin/env python3
"""Select downstream heads by CV and compare models on common test rows.

Reads head-level score CSVs written by ``score.py`` and the saved test
prediction archives (``<predictions-dir>/<dataset>/<embedder>/<head>.npz``).
Neither input is modified: archived ``test_metric`` values are reported as
stored, and the prediction archives are only used to verify them and to
rescore models on shared rows.

1. For each dataset, restrict candidate heads to those evaluated for every
   included embedder. Then ``collapse_best_head`` picks the best training-side
   CV ROC-AUC per dataset x embedder, never the test score.
2. Each selected head's archive is checked against the prepared dataset:
   source-row indices must be unique prepared test rows, labels must match the
   prepared labels at those rows, the prepared-file SHA-256 must match, and
   ROC-AUC recomputed from the archive must equal the archived ``test_metric``.
   Coverage is the share of prepared test rows that received a prediction.
3. For each dataset, models whose archive passed every check are rescored on
   the test rows that all of them predicted, so paired differences are taken
   over the same molecules. Average precision and positive counts are computed
   from the same fixed predictions; they are never used for selection.
4. For each dataset and pair of verified models, the ROC-AUC difference on
   those common rows gets a paired bootstrap interval: the same resampled test
   molecules score both models. A per-task win counts only when the interval
   excludes zero; a small test set can otherwise turn noise into a win.

Archives without ``test_source_row_indices`` (all pre-``112efc5`` runs) or
without a matching ``prepared_data_sha256`` are reported but excluded from the
common-row comparison.

Outputs in ``--output-dir``:
    head_candidates.csv    every head row, with ``eligible_head`` and ``selected`` flags
    selected_heads.csv     CV-selected heads with archive checks and coverage
    common_row_scores.csv  per dataset x embedder scores on common test rows
    common_row_scores_no_split_overlap.csv  optional paired sensitivity output
    paired_task_differences.csv  per dataset x model pair ROC-AUC difference and interval
    manifest.json          input hashes, code revision and arguments

Usage:
    uv run python scripts/paper/build_common_row_benchmark.py \\
        --results outputs/eval/revision_clean_small_v1/results.csv \\
        --output-dir outputs/eval/revision_clean_small_v1/common_rows
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from build_benchmark_results_frames import collapse_best_head
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    _normalize_auc_scores,
    get_skfp_roc_auc,
)
from modernmolbert.utils import file_sha256

SCORE_ATOL = 1e-9
HEAD_KEYS = ["dataset", "embedder", "test_metric_name", "model"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select downstream heads by CV and compare models on common test rows."
    )
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=None,
        help="Restrict to these embedders (default: every embedder in --results).",
    )
    parser.add_argument("--exclude-datasets", nargs="*", default=[])
    parser.add_argument(
        "--split-overlap-rows",
        type=Path,
        help="Optional row audit from audit_split_overlap.py for a paired exclusion sensitivity.",
    )
    parser.add_argument(
        "--paired-reference",
        help="Compare this embedder with each other one (default: every pair of embedders).",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Paired bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_head_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["result_source"] = str(path.resolve())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def common_candidate_heads(heads: pd.DataFrame) -> dict[str, set[str]]:
    """Allow only heads offered for every included embedder in each dataset."""
    shared = {}
    for dataset, group in heads.groupby("dataset", sort=True):
        offered = [set(rows["model"].astype(str)) for _, rows in group.groupby("embedder")]
        eligible = set.intersection(*offered)
        if not eligible:
            raise ValueError(f"No common candidate heads for dataset {dataset}")
        shared[str(dataset)] = eligible
    return shared


def load_prepared_test(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (prepared test row indices, labels for every prepared row)."""
    dataset = Dataset.deserialize_legacy(path)
    test_rows = np.asarray(dataset.splits.get("test", []), dtype=np.int64)
    labels = dataset.labels.to_numpy(dtype=float)
    return test_rows, labels


def _as_label_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def check_archive(
    path: Path,
    archived_test_metric: float,
    prepared_test_rows: np.ndarray,
    prepared_labels: np.ndarray,
    prepared_sha256: str,
) -> dict[str, object]:
    """Verify one prediction archive against the prepared data and archived score."""
    record: dict[str, object] = {
        "prediction_path": str(path),
        "archive_sha256": None,
        "archive_test_metric": np.nan,
        "n_predicted_test": np.nan,
        "archive_prepared_sha256": None,
    }
    if not path.exists():
        return record | {"archive_status": "missing_archive"}
    record["archive_sha256"] = file_sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        y_true = archive["y_true"]
        y_score = archive["y_score"]
        rows = (
            archive["test_source_row_indices"]
            if "test_source_row_indices" in archive.files
            else None
        )
        prepared_hash = (
            str(archive["prepared_data_sha256"].item())
            if "prepared_data_sha256" in archive.files
            else None
        )
    recomputed = float(get_skfp_roc_auc(y_score, y_true))
    record["n_predicted_test"] = len(y_true)
    record["archive_test_metric"] = recomputed
    record["archive_prepared_sha256"] = prepared_hash
    if not np.isclose(recomputed, archived_test_metric, rtol=0, atol=SCORE_ATOL):
        return record | {"archive_status": "score_mismatch"}
    if rows is None:
        return record | {"archive_status": "no_row_ids"}
    if rows.ndim != 1 or len(rows) != len(y_true) or len(np.unique(rows)) != len(rows):
        return record | {"archive_status": "invalid_row_ids"}
    if not np.isin(rows, prepared_test_rows).all():
        return record | {"archive_status": "rows_outside_test"}
    if not np.array_equal(
        _as_label_matrix(prepared_labels[rows]), _as_label_matrix(y_true), equal_nan=True
    ):
        return record | {"archive_status": "label_mismatch"}
    if prepared_hash is None:
        return record | {"archive_status": "no_prepared_hash"}
    if prepared_hash != prepared_sha256:
        return record | {"archive_status": "prepared_hash_mismatch"}
    return record | {"archive_status": "ok"}


def multioutput_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean average precision over endpoints with finite labels and both classes."""
    y_true = _as_label_matrix(y_true)
    y_score = _as_label_matrix(y_score)
    scores = []
    for col in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, col])
        labels = y_true[mask, col].astype(int)
        if len(np.unique(labels)) < 2:
            continue
        scores.append(float(average_precision_score(labels, y_score[mask, col])))
    return float(np.mean(scores)) if scores else float("nan")


def score_rows(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    labels = _as_label_matrix(y_true)
    finite = np.isfinite(labels)
    n_positive = int((labels[finite] == 1).sum())
    n_labelled = int(finite.sum())
    scored_positive_counts = []
    for column in range(labels.shape[1]):
        observed = labels[finite[:, column], column]
        positive = int((observed == 1).sum())
        negative = int((observed == 0).sum())
        if positive and negative:
            scored_positive_counts.append(positive)
    return {
        "roc_auc_common": (
            float(get_skfp_roc_auc(y_score, y_true)) if scored_positive_counts else float("nan")
        ),
        "average_precision_common": multioutput_average_precision(y_true, y_score),
        "n_scored_endpoints_common": len(scored_positive_counts),
        "min_positive_per_scored_endpoint_common": (
            min(scored_positive_counts) if scored_positive_counts else float("nan")
        ),
        "n_labelled_common": n_labelled,
        "n_positive_common": n_positive,
        "prevalence_common": n_positive / n_labelled if n_labelled else float("nan"),
    }


def aligned_common_rows(
    group: pd.DataFrame, excluded_rows: set[int] | None = None
) -> tuple[list[int], dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    """Return the common test rows and each model's labels and scores on them, in order."""
    archives = {}
    for embedder, path in zip(group["embedder"], group["prediction_path"], strict=True):
        with np.load(str(path), allow_pickle=False) as archive:
            archives[str(embedder)] = (
                archive["test_source_row_indices"],
                archive["y_true"],
                archive["y_score"],
            )
    common_set = set.intersection(*(set(rows.tolist()) for rows, _, _ in archives.values()))
    common = sorted(common_set - (excluded_rows or set()))
    aligned = {}
    for embedder, (rows, y_true, y_score) in archives.items():
        position = {int(source): i for i, source in enumerate(rows)}
        take = [position[source] for source in common]
        aligned[embedder] = (y_true[take], y_score[take], len(rows))
    return common, aligned


def common_row_scores(
    selected: pd.DataFrame, excluded_rows_by_dataset: dict[str, set[int]] | None = None
) -> pd.DataFrame:
    """Rescore every verified model of a dataset on the rows all of them predicted."""
    records = []
    verified = selected.loc[selected["archive_status"].eq("ok")]
    for dataset, group in verified.groupby("dataset", sort=True):
        excluded = (excluded_rows_by_dataset or {}).get(str(dataset))
        common, aligned = aligned_common_rows(group, excluded)
        for row in group.itertuples(index=False):
            y_true, y_score, n_predicted = aligned[str(row.embedder)]
            record: dict[str, object] = {
                "dataset": dataset,
                "embedder": row.embedder,
                "model": row.model,
                "n_models_compared": len(aligned),
                "models_compared": ";".join(sorted(aligned)),
                "n_common_test_rows": len(common),
                "n_predicted_test": n_predicted,
                "test_metric_archived": row.test_metric,
            }
            if common:
                record |= score_rows(y_true, y_score)
            records.append(record)
    return pd.DataFrame(records)


def rank_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC from average ranks; equal to sklearn's value, including ties."""
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    ranks = rankdata(scores)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def endpoint_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels and positive-class scores as (rows, endpoints) float matrices."""
    labels = _as_label_matrix(y_true)
    scores = _normalize_auc_scores(y_score, n_outputs=labels.shape[1], n_samples=len(labels))
    return labels, _as_label_matrix(scores)


def paired_auc_difference(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, rows: np.ndarray
) -> float:
    """Mean ROC-AUC of a minus b over endpoints with both classes among ``rows``."""
    differences = []
    for column in range(labels.shape[1]):
        observed = rows[np.isfinite(labels[rows, column])]
        y = labels[observed, column]
        if 0 < y.sum() < len(y):
            differences.append(
                rank_roc_auc(y, scores_a[observed, column])
                - rank_roc_auc(y, scores_b[observed, column])
            )
    return float(np.mean(differences)) if differences else float("nan")


def paired_task_differences(
    selected: pd.DataFrame,
    *,
    reference: str | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Bootstrap each dataset's paired ROC-AUC difference over its common test rows."""
    rng = np.random.default_rng(seed)
    records = []
    verified = selected.loc[selected["archive_status"].eq("ok")]
    for dataset, group in verified.groupby("dataset", sort=True):
        common, aligned = aligned_common_rows(group)
        embedders = sorted(aligned)
        if reference is None:
            pairs = [(a, b) for i, a in enumerate(embedders) for b in embedders[i + 1 :]]
        elif reference in aligned:
            pairs = [(reference, other) for other in embedders if other != reference]
        else:
            pairs = []
        for model_a, model_b in pairs:
            labels, scores_a = endpoint_scores(*aligned[model_a][:2])
            _, scores_b = endpoint_scores(*aligned[model_b][:2])
            n = len(common)
            observed = paired_auc_difference(labels, scores_a, scores_b, np.arange(n))
            boot = np.array(
                [
                    paired_auc_difference(labels, scores_a, scores_b, rng.integers(0, n, size=n))
                    for _ in range(n_boot if n else 0)
                ]
            )
            boot = boot[np.isfinite(boot)]
            low, high = (
                np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                if len(boot)
                else (np.nan, np.nan)
            )
            records.append(
                {
                    "dataset": dataset,
                    "embedder_a": model_a,
                    "embedder_b": model_b,
                    "n_common_test_rows": n,
                    "roc_auc_difference": observed,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_boot": n_boot,
                    "n_valid_boot": len(boot),
                    "clear_winner": (model_a if low > 0 else model_b if high < 0 else "neither"),
                }
            )
    return pd.DataFrame(records)


def load_split_overlap_exclusions(
    path: Path, prepared_cache: dict[str, tuple[np.ndarray, np.ndarray, str]]
) -> dict[str, set[int]]:
    """Verify audit provenance and mark test rows sharing a fitting-side identity."""
    audit = pd.read_csv(path)
    required = {
        "dataset",
        "prepared_sha256",
        "test_source_row_index",
        "same_inchikey",
        "same_nonisomeric_smiles",
    }
    if missing := required - set(audit.columns):
        raise ValueError(f"Split-overlap audit missing columns: {sorted(missing)}")
    exclusions: dict[str, set[int]] = {}
    for dataset, (test_rows, _, prepared_sha) in prepared_cache.items():
        group = audit.loc[audit["dataset"].eq(dataset)]
        if len(group) != len(test_rows):
            raise ValueError(f"Split-overlap audit row count mismatch for {dataset}")
        if set(group["prepared_sha256"]) != {prepared_sha}:
            raise ValueError(f"Split-overlap audit prepared-file hash mismatch for {dataset}")
        audited_rows = group["test_source_row_index"].astype(int).tolist()
        if len(set(audited_rows)) != len(audited_rows) or set(audited_rows) != set(test_rows):
            raise ValueError(f"Split-overlap audit test-row mapping mismatch for {dataset}")
        flag_columns = []
        for column in ("same_inchikey", "same_nonisomeric_smiles"):
            values = group[column].astype(str).str.lower()
            if not values.isin(["true", "false"]).all():
                raise ValueError(f"Invalid {column} flag in split-overlap audit for {dataset}")
            flag_columns.append(values.eq("true"))
        flagged = flag_columns[0] | flag_columns[1]
        exclusions[dataset] = set(group.loc[flagged, "test_source_row_index"].astype(int))
    return exclusions


def git_revision() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    heads = load_head_results(args.results)
    if args.embedders:
        heads = heads.loc[heads["embedder"].isin(args.embedders)]
    heads = heads.loc[~heads["dataset"].isin(args.exclude_datasets)].reset_index(drop=True)
    if heads.empty:
        raise ValueError("No head results left after filtering")

    common_heads = common_candidate_heads(heads)
    eligible = heads.apply(
        lambda row: str(row["model"]) in common_heads[str(row["dataset"])], axis=1
    )
    selected = collapse_best_head(heads.loc[eligible].copy())
    selected_keys = set(selected[HEAD_KEYS].itertuples(index=False, name=None))
    candidates = heads.assign(
        eligible_head=eligible,
        selected=[
            key in selected_keys for key in heads[HEAD_KEYS].itertuples(index=False, name=None)
        ],
    )

    prepared_cache: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    checks = []
    for row in selected.to_dict("records"):
        dataset = str(row["dataset"])
        if dataset not in prepared_cache:
            path = args.prepared_dir / f"{dataset}.json"
            test_rows, labels = load_prepared_test(path)
            prepared_cache[dataset] = (test_rows, labels, file_sha256(path))
        test_rows, labels, prepared_sha = prepared_cache[dataset]
        archive = args.predictions_dir / dataset / str(row["embedder"]) / f"{row['model']}.npz"
        check = check_archive(archive, float(row["test_metric"]), test_rows, labels, prepared_sha)
        n_test = len(test_rows)
        n_predicted = float(str(check["n_predicted_test"]))
        checks.append(
            check
            | {
                "prepared_sha256": prepared_sha,
                "n_prepared_test": n_test,
                "test_coverage": n_predicted / n_test if n_test else np.nan,
            }
        )
    selected = pd.concat([selected, pd.DataFrame(checks, index=selected.index)], axis=1)
    common = common_row_scores(selected)
    excluded = (
        load_split_overlap_exclusions(args.split_overlap_rows, prepared_cache)
        if args.split_overlap_rows
        else None
    )
    sensitivity = common_row_scores(selected, excluded) if excluded is not None else None
    paired = paired_task_differences(
        selected, reference=args.paired_reference, n_boot=args.n_boot, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "head_candidates.csv", index=False)
    selected.to_csv(args.output_dir / "selected_heads.csv", index=False)
    common.to_csv(args.output_dir / "common_row_scores.csv", index=False)
    if sensitivity is not None:
        sensitivity.to_csv(args.output_dir / "common_row_scores_no_split_overlap.csv", index=False)
    paired.to_csv(args.output_dir / "paired_task_differences.csv", index=False)
    manifest = {
        "script": "scripts/paper/build_common_row_benchmark.py",
        "code": git_revision(),
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "results_sha256": {str(path): file_sha256(path) for path in args.results},
        "selection_rule": (
            "common candidate heads per dataset, then max training-side "
            "CV ROC-AUC per dataset x embedder"
        ),
        "candidate_heads_by_dataset": {
            dataset: sorted(models) for dataset, models in common_heads.items()
        },
        "archive_status_counts": selected["archive_status"].value_counts().to_dict(),
        "paired_task_bootstrap": {
            "resamples": args.n_boot,
            "seed": args.seed,
            "interval": "95% percentile; test rows resampled jointly for both models",
            "clear_winner_rule": "interval excludes zero",
        },
    }
    if excluded is not None:
        manifest["split_overlap_sensitivity"] = {
            "audit_path": str(args.split_overlap_rows),
            "audit_sha256": file_sha256(args.split_overlap_rows),
            "exclusion_rule": "same standard InChIKey or canonical SMILES without stereochemistry",
            "n_flagged_test_rows_by_dataset": {
                dataset: len(rows) for dataset, rows in excluded.items()
            },
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Selected {len(selected)} heads from {len(candidates)} candidates")
    print(f"Archive checks: {manifest['archive_status_counts']}")
    print(f"Common-row scores: {len(common)} rows; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
