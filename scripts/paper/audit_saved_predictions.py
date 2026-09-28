"""Compare saved test predictions with historical benchmark result rows.

This is a provenance audit, not a corrected benchmark: matching a scalar score
does not prove molecule IDs, splits, checkpoint revision, or CV provenance.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    get_skfp_roc_auc,
)


def compare_prediction(
    prediction_path: Path,
    result_rows: pd.DataFrame,
    prepared_test_labels: np.ndarray | None,
    prepared_test_indices: np.ndarray | None = None,
) -> dict[str, object]:
    with np.load(prediction_path, allow_pickle=False) as predictions:
        required = {"y_true", "y_score"}
        optional = {"test_source_row_indices"}
        if not required.issubset(predictions.files) or set(predictions.files) - required - optional:
            raise ValueError(
                f"Unexpected prediction columns in {prediction_path}: {predictions.files}"
            )
        y_true = predictions["y_true"]
        y_score = predictions["y_score"]
        source_rows = (
            predictions["test_source_row_indices"]
            if "test_source_row_indices" in predictions.files
            else None
        )
    if y_true.shape != y_score.shape:
        raise ValueError(f"Prediction/label shape mismatch in {prediction_path}")
    score = float(get_skfp_roc_auc(y_score, y_true))
    labels_match = None
    rows_belong_to_test = None
    if prepared_test_labels is not None:
        expected = np.asarray(prepared_test_labels, dtype=float)
        observed = np.asarray(y_true, dtype=float)
        if source_rows is not None:
            if source_rows.ndim != 1 or len(source_rows) != len(y_true):
                raise ValueError(f"Invalid test source row mapping in {prediction_path}")
            if prepared_test_indices is None:
                raise ValueError("Prepared test indices are required for mapped predictions")
            test_positions = {
                int(source_row): position
                for position, source_row in enumerate(prepared_test_indices)
            }
            rows_belong_to_test = all(int(row) in test_positions for row in source_rows)
            if rows_belong_to_test:
                expected = expected[[test_positions[int(row)] for row in source_rows]]
        if expected.ndim == 2 and expected.shape[1] == 1 and observed.ndim == 1:
            expected = expected[:, 0]
        labels_match = bool(
            rows_belong_to_test is not False
            and expected.shape == observed.shape
            and np.array_equal(expected, observed, equal_nan=True)
        )
    scores = pd.to_numeric(result_rows["test_metric"], errors="raise").to_numpy(dtype=float)
    matches = np.flatnonzero(np.isclose(scores, score, rtol=0, atol=1e-10))
    return {
        "dataset": prediction_path.parent.parent.name,
        "embedder": prediction_path.parent.name,
        "head": prediction_path.stem,
        "n_prediction_rows": int(y_true.shape[0]),
        "has_test_source_row_indices": source_rows is not None,
        "source_rows_belong_to_prepared_test": rows_belong_to_test,
        "n_unique_source_rows": len(np.unique(source_rows)) if source_rows is not None else None,
        "n_prepared_test_rows": (
            len(prepared_test_labels) if prepared_test_labels is not None else None
        ),
        "labels_match_prepared_test": labels_match,
        "n_label_columns": 1 if y_true.ndim == 1 else int(y_true.shape[1]),
        "prediction_test_roc_auc": score,
        "n_csv_rows": len(result_rows),
        "n_matching_csv_rows": len(matches),
        "csv_test_roc_auc_values": json.dumps(scores.tolist()),
        "matching_csv_sources": json.dumps(
            [str(result_rows.iloc[i]["result_source"]) for i in matches]
        ),
        "min_abs_score_difference": float(np.min(np.abs(scores - score))) if len(scores) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--results-root", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for path in sorted(args.results_root.glob("praski_best_*/results.csv")):
        frame = pd.read_csv(path)
        frame["result_source"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No historical results under {args.results_root}")
    results = pd.concat(frames, ignore_index=True)
    for col in ("dataset", "embedder", "model", "test_metric"):
        if col not in results:
            raise ValueError(f"Missing {col} from results CSVs")
    prepared_test_labels = {}
    prepared_test_indices = {}
    for path in args.prepared_dir.glob("*.json"):
        dataset = Dataset.deserialize_legacy(path)
        indices = list(dataset.splits.get("test", []))
        prepared_test_labels[path.stem] = dataset.labels.iloc[indices].to_numpy(dtype=float)
        prepared_test_indices[path.stem] = np.asarray(indices, dtype=int)

    rows = []
    for path in sorted(args.predictions_dir.glob("*/*/*.npz")):
        matching = results.loc[
            results["dataset"].eq(path.parent.parent.name)
            & results["embedder"].eq(path.parent.name)
            & results["model"].eq(path.stem)
        ]
        rows.append(
            compare_prediction(
                path,
                matching,
                prepared_test_labels.get(path.parent.parent.name),
                prepared_test_indices.get(path.parent.parent.name),
            )
        )
    if not rows:
        raise FileNotFoundError(f"No prediction archives under {args.predictions_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} prediction archives; wrote {args.output}")
    print(f"Exact scalar score matches: {sum(bool(r['n_matching_csv_rows']) for r in rows)}")
    print(f"No matching CSV row: {sum(not r['n_csv_rows'] for r in rows)}")
    print(
        f"Prepared test size mismatches: {sum(r['n_prediction_rows'] != r['n_prepared_test_rows'] for r in rows)}"
    )
    print(
        f"Prepared test label matches: {sum(r['labels_match_prepared_test'] is True for r in rows)}"
    )


if __name__ == "__main__":
    main()
