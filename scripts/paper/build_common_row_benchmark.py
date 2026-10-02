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

With ``--evaluation-manifest``, additionally verify all five common embeddings
against their frozen hashes, prepared task hashes, shared supervised row counts,
and the deterministic five-fold CV policy before marking cohort provenance valid.

Archives without ``test_source_row_indices`` (all pre-``112efc5`` runs) or
without a matching ``prepared_data_sha256`` are reported but excluded from the
common-row comparison.

Baselines can come from a head-level table scored elsewhere, such as the
imported Praski et al. results (``--table-results`` with ``--table-embedders``).
Their heads are chosen by the same CV rule from the same shared candidate
set, but they have no prediction archives: their status is ``table_only``,
their test scores are used as published, and they take no part in
common-row or paired per-dataset comparisons. Whether their test molecules
match ours cannot be checked.

Outputs in ``--output-dir``:
    head_candidates.csv    every head row, with ``eligible_head`` and ``selected`` flags
    selected_heads.csv     CV-selected heads with archive checks and coverage
    common_row_scores.csv  per dataset x embedder scores on common test rows
    common_row_scores_no_split_overlap.csv  optional paired sensitivity output
    paired_task_differences.csv  per dataset x model pair ROC-AUC difference and interval
    task_matrix.csv        dataset x embedder test ROC-AUC of the CV-selected head, for
                           verified archives and table-only baselines on their own rows
    common_task_matrix.csv dataset x archive-backed embedder ROC-AUC on common test rows;
                           incomplete model cohorts are left missing for the whole dataset
    common_task_matrix_status.csv  dataset eligibility and excluded/missing models
    common_task_matrix_no_split_overlap.csv  optional common-row sensitivity matrix
    common_task_matrix_no_split_overlap_status.csv  optional sensitivity eligibility
    manifest.json          input hashes, code revision and arguments

Usage:
    uv run python scripts/paper/build_common_row_benchmark.py \\
        --results outputs/eval/revision_clean_small_v1/results.csv \\
        --output-dir outputs/eval/revision_clean_small_v1/common_rows
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from build_benchmark_results_frames import collapse_best_head
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.eval.benchmarking_molecular_models.supervised.const import (
    PRODUCTION_CV_POLICY,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    _normalize_auc_scores,
    get_skfp_roc_auc,
)
from modernmolbert.utils import file_sha256, get_git_revision as git_revision

SCORE_ATOL = 1e-9
HEAD_KEYS = ["dataset", "embedder", "test_metric_name", "model"]
COMMON_SCORE_COLUMNS = [
    "dataset",
    "embedder",
    "model",
    "n_models_compared",
    "models_compared",
    "n_common_test_rows",
    "n_predicted_test",
    "test_metric_archived",
    "roc_auc_common",
    "average_precision_common",
    "n_scored_endpoints_common",
    "min_positive_per_scored_endpoint_common",
    "n_labelled_common",
    "n_positive_common",
    "prevalence_common",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select downstream heads by CV and compare models on common test rows."
    )
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--table-results",
        type=Path,
        nargs="+",
        default=[],
        help="Head-level CSVs scored elsewhere, without prediction archives (e.g. the Praski table).",
    )
    parser.add_argument(
        "--table-embedders",
        nargs="+",
        default=[],
        help="Embedders to take from --table-results; required with it.",
    )
    parser.add_argument(
        "--matrix-labels",
        nargs="*",
        default=[],
        metavar="EMBEDDER=LABEL",
        help="Column labels for native and common-row task matrices.",
    )
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--embedded-dir", type=Path, default=Path("data/embedded"))
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        help="Verify the frozen common training/test rows and embedding hashes for this cohort.",
    )
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
    args = parser.parse_args(argv)
    if bool(args.table_results) != bool(args.table_embedders):
        parser.error("--table-results and --table-embedders must be given together")
    return args


def load_head_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["result_source"] = str(path.resolve())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_table_heads(paths: list[Path], embedders: list[str]) -> pd.DataFrame:
    """Head rows for the requested embedders from tables without prediction archives.

    In the imported Praski table ``library_hash`` is a per-batch Python hash,
    not a grid identity: the hERG-Karim fingerprint kNN rows were scored in a
    later batch than their other heads. It is kept as ``table_batch_id`` so the
    run-provenance check applied to our own results does not reject it.
    """
    table = load_head_results(paths)
    table = table.loc[table["embedder"].isin(embedders)].copy()
    if "library_hash" in table.columns:
        table = table.rename(columns={"library_hash": "table_batch_id"})
    if missing := sorted(set(embedders) - set(table["embedder"])):
        raise ValueError(f"Table embedders not found in --table-results: {missing}")
    return table.assign(score_source="table")


def parse_labels(values: list[str]) -> dict[str, str]:
    labels = {}
    for value in values:
        embedder, sep, label = value.partition("=")
        if not sep or not embedder or not label:
            raise ValueError(f"Expected EMBEDDER=LABEL, got {value!r}")
        labels[embedder] = label
    return labels


def task_matrix(selected: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Test ROC-AUC of each CV-selected head that is verified or taken from a table."""
    usable = selected.loc[selected["archive_status"].isin(["ok", "table_only"])]
    matrix = usable.pivot(index="dataset", columns="embedder", values="test_metric")
    return matrix.rename(columns=labels).rename_axis(index=None, columns=None)


def common_task_matrix(
    scores: pd.DataFrame,
    datasets: list[str],
    embedders: list[str],
    labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export common-row scores for one fixed cohort, never a smaller surviving subset.

    ``common_row_scores`` also reports diagnostic scores when only some archives
    verify. Those scores must not enter the primary matrix: a missing/invalid
    model changes the intersection for every model. Keep the dataset as an all-NaN
    row with an explicit status instead of silently changing the comparison.
    This checks test-row comparability only, not training rows or CV folds.
    """
    names = [labels.get(embedder, embedder) for embedder in embedders]
    if len(set(names)) != len(names):
        raise ValueError("Common task matrix column labels must be unique")
    matrix = pd.DataFrame(np.nan, index=datasets, columns=embedders)
    expected = set(embedders)
    records = []
    for dataset in datasets:
        group = scores.loc[scores["dataset"].eq(dataset)] if not scores.empty else scores
        found = set(group["embedder"]) if not group.empty else set()
        record = {
            "dataset": dataset,
            "n_models_expected": len(expected),
            "n_models_verified": len(found),
            "missing_models": ";".join(sorted(expected - found)),
            "n_common_test_rows": np.nan,
            "status": "incomplete_cohort",
        }
        if found == expected and expected:
            if group["embedder"].duplicated().any():
                raise ValueError(f"Repeated common-row scores for {dataset}")
            counts = group["n_common_test_rows"].unique()
            if len(counts) != 1 or not np.all(
                group["n_models_compared"].to_numpy(dtype=int) == len(expected)
            ):
                raise ValueError(f"Inconsistent common-row cohort for {dataset}")
            record["n_common_test_rows"] = int(counts[0])
            values = group.set_index("embedder").reindex(embedders)
            if counts[0] == 0:
                record["status"] = "no_common_rows"
            elif (
                "roc_auc_common" not in values
                or not np.isfinite(values["roc_auc_common"].to_numpy(dtype=float)).all()
            ):
                record["status"] = "undefined_roc_auc"
            else:
                matrix.loc[dataset, embedders] = values["roc_auc_common"].to_numpy(dtype=float)
                record["status"] = "ok"
        records.append(record)
    return matrix.rename(columns=labels).rename_axis(index=None, columns=None), pd.DataFrame(
        records
    )


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


def verify_evaluation_manifest(
    path: Path,
    *,
    prepared_dir: Path,
    embedded_dir: Path,
    cohort: list[str],
    datasets: list[str],
) -> dict[str, object]:
    """Tie every scored embedding to the frozen five-model supervised cohort."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_ids = manifest.get("run_ids", [])
    prefix = manifest.get("common_prefix")
    if manifest.get("schema") != 2 or len(run_ids) != 5 or not isinstance(prefix, str):
        raise ValueError("Expected a schema-2 five-model evaluation manifest")
    if set(cohort) != {f"{prefix}{run_id}" for run_id in run_ids}:
        raise ValueError("Scored embedder cohort differs from the evaluation manifest")
    if set(datasets) != set(manifest.get("tasks", {})):
        raise ValueError("Scored task set differs from the evaluation manifest")
    if manifest.get("cv") != PRODUCTION_CV_POLICY:
        raise ValueError("Evaluation manifest does not declare the fixed five-fold CV policy")
    for dataset in datasets:
        task = manifest["tasks"][dataset]
        if file_sha256(prepared_dir / f"{dataset}.json") != task["prepared_sha256"]:
            raise ValueError(f"Prepared task differs from the evaluation manifest: {dataset}")
        if set(task["models"]) != set(run_ids):
            raise ValueError(f"Incomplete five-model cohort in evaluation manifest: {dataset}")
        if sum(task["splits"].values()) != task["common_supervised_rows"]:
            raise ValueError(f"Common split counts differ from retained rows: {dataset}")
        for run_id in run_ids:
            embedder = f"{prefix}{run_id}"
            embedding = embedded_dir / dataset / f"{embedder}.joblib"
            if file_sha256(embedding) != task["models"][run_id]["common_embedding_sha256"]:
                raise ValueError(f"Common embedding differs from evaluation manifest: {embedding}")
    return {
        "evaluation_manifest_sha256": file_sha256(path),
        "cv": manifest["cv"],
        "supervised_rows_and_folds": "verified by schema-2 common cohort and fixed scorer CV policy",
    }


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
    try:
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
    except Exception:
        return record | {"archive_status": "corrupt_archive"}
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
    return pd.DataFrame(records, columns=COMMON_SCORE_COLUMNS)


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
            digest = hashlib.sha256("\0".join((dataset, model_a, model_b)).encode("utf-8")).digest()
            rng = np.random.default_rng([seed, *np.frombuffer(digest[:16], dtype="<u4").tolist()])
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    labels = parse_labels(args.matrix_labels)
    heads = load_head_results(args.results).assign(score_source="archive")
    if args.embedders:
        if missing := sorted(set(args.embedders) - set(heads["embedder"])):
            raise ValueError(f"Requested embedders not found in --results: {missing}")
        heads = heads.loc[heads["embedder"].isin(args.embedders)]
    if args.table_results:
        table = load_table_heads(args.table_results, args.table_embedders)
        if clash := sorted(set(table["embedder"]) & set(heads["embedder"])):
            raise ValueError(f"Embedders appear in both --results and --table-results: {clash}")
        heads = pd.concat([heads, table], ignore_index=True)
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
            test_rows, row_labels = load_prepared_test(path)
            prepared_cache[dataset] = (test_rows, row_labels, file_sha256(path))
        test_rows, prepared_labels, prepared_sha = prepared_cache[dataset]
        if row["score_source"] == "table":
            check: dict[str, object] = {
                "prediction_path": None,
                "archive_sha256": None,
                "archive_test_metric": np.nan,
                "n_predicted_test": np.nan,
                "archive_prepared_sha256": None,
                "archive_status": "table_only",
            }
        else:
            archive = args.predictions_dir / dataset / str(row["embedder"]) / f"{row['model']}.npz"
            check = check_archive(
                archive, float(row["test_metric"]), test_rows, prepared_labels, prepared_sha
            )
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
    local = heads.loc[heads["score_source"].eq("archive")]
    if local.empty:
        raise ValueError("No archive-backed head results left after filtering")
    cohort = sorted(set(args.embedders or local["embedder"].unique().tolist()))
    datasets = sorted(local["dataset"].unique().tolist())
    evaluation_evidence = None
    if args.evaluation_manifest is not None:
        expected_pairs = {(dataset, embedder) for dataset in datasets for embedder in cohort}
        scored_pairs = set(local[["dataset", "embedder"]].itertuples(index=False, name=None))
        if scored_pairs != expected_pairs:
            raise ValueError("Head results omit a task/model pair from the fixed evaluation cohort")
        evaluation_evidence = verify_evaluation_manifest(
            args.evaluation_manifest,
            prepared_dir=args.prepared_dir,
            embedded_dir=args.embedded_dir,
            cohort=cohort,
            datasets=datasets,
        )
    common_matrix, common_status = common_task_matrix(common, datasets, cohort, labels)
    sensitivity_matrix = (
        common_task_matrix(sensitivity, datasets, cohort, labels)
        if sensitivity is not None
        else None
    )
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
    task_matrix(selected, labels).to_csv(args.output_dir / "task_matrix.csv")
    common_matrix.to_csv(args.output_dir / "common_task_matrix.csv")
    common_status.to_csv(args.output_dir / "common_task_matrix_status.csv", index=False)
    if sensitivity_matrix is not None:
        sensitivity_matrix[0].to_csv(args.output_dir / "common_task_matrix_no_split_overlap.csv")
        sensitivity_matrix[1].to_csv(
            args.output_dir / "common_task_matrix_no_split_overlap_status.csv", index=False
        )
    manifest = {
        "script": "scripts/paper/build_common_row_benchmark.py",
        "code": git_revision(),
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "results_sha256": {str(path): file_sha256(path) for path in args.results},
        "table_results_sha256": {str(path): file_sha256(path) for path in args.table_results},
        "table_embedders": args.table_embedders,
        "matrix_labels": labels,
        "task_matrices": {
            "task_matrix.csv": {
                "population": "each model's own test rows; includes table-only baselines",
                "metric": "test_metric",
            },
            "common_task_matrix.csv": {
                "population": "intersection of verified test rows for the fixed archive cohort",
                "metric": "roc_auc_common",
                "embedders": cohort,
                "incomplete_cohort_policy": "all scores missing for that dataset",
                "status_file": "common_task_matrix_status.csv",
                "status_counts": common_status["status"].value_counts().to_dict(),
                "training_rows_and_cv_folds_verified": evaluation_evidence is not None,
            },
        },
        "selection_rule": (
            "common candidate heads per dataset, then max training-side "
            "CV ROC-AUC per dataset x embedder"
        ),
        "candidate_heads_by_dataset": {
            dataset: sorted(models) for dataset, models in common_heads.items()
        },
        "archive_status_counts": selected["archive_status"].value_counts().to_dict(),
        "evaluation_evidence": evaluation_evidence,
        "output_sha256": {
            name: file_sha256(args.output_dir / name)
            for name in (
                "head_candidates.csv",
                "selected_heads.csv",
                "common_row_scores.csv",
                "paired_task_differences.csv",
                "task_matrix.csv",
                "common_task_matrix.csv",
                "common_task_matrix_status.csv",
            )
        },
        "paired_task_bootstrap": {
            "resamples": args.n_boot,
            "seed": args.seed,
            "interval": "95% percentile; test rows resampled jointly for both models",
            "clear_winner_rule": "interval excludes zero",
        },
    }
    if excluded is not None:
        assert sensitivity_matrix is not None
        manifest["output_sha256"].update(
            {
                name: file_sha256(args.output_dir / name)
                for name in (
                    "common_row_scores_no_split_overlap.csv",
                    "common_task_matrix_no_split_overlap.csv",
                    "common_task_matrix_no_split_overlap_status.csv",
                )
            }
        )
        manifest["task_matrices"]["common_task_matrix_no_split_overlap.csv"] = {
            "population": "common test rows after the split-overlap exclusion",
            "metric": "roc_auc_common",
            "embedders": cohort,
            "incomplete_cohort_policy": "all scores missing for that dataset",
            "status_file": "common_task_matrix_no_split_overlap_status.csv",
            "status_counts": sensitivity_matrix[1]["status"].value_counts().to_dict(),
            "training_rows_and_cv_folds_verified": evaluation_evidence is not None,
        }
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
