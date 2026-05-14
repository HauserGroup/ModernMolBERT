from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PRASKI_COLUMNS = [
    "id",
    "dataset",
    "task",
    "embedder",
    "model",
    "hyperparams",
    "library_hash",
    "cv_metric_name",
    "cv_metric",
    "test_metric_name",
    "test_metric",
    "key",
]


def add_result_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["key"] = pd.Series(dtype=str)
        return out

    out["key"] = (
        out["dataset"].astype(str)
        + "_"
        + out["embedder"].astype(str)
        + "_"
        + out["model"].astype(str)
    )
    return out


def to_praski_schema(
    frame: pd.DataFrame,
    *,
    embedder_name: str | None = None,
    library_hash: str | int | None = None,
) -> pd.DataFrame:
    out = frame.copy()

    rename_map = {
        "display_name": "dataset",
        "task_type": "task",
        "downstream_name": "model",
        "downstream_best_params": "hyperparams",
        "metric_name": "test_metric_name",
    }
    for source, target in rename_map.items():
        if source in out.columns and (target not in out.columns or source == "display_name"):
            out[target] = out[source]

    if embedder_name is not None:
        out["embedder"] = embedder_name
    if library_hash is not None:
        out["library_hash"] = str(library_hash)

    if "id" not in out.columns:
        out["id"] = range(1, len(out) + 1)
    if "cv_metric_name" not in out.columns and "test_metric_name" in out.columns:
        out["cv_metric_name"] = out["test_metric_name"]
    if "hyperparams" not in out.columns:
        out["hyperparams"] = "{}"

    out = add_result_key(out)

    for column in PRASKI_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    return out[PRASKI_COLUMNS]


def export_classification_reports(
    *,
    database_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    database_path = Path(database_path)
    output_csv = Path(output_csv)

    if not database_path.exists():
        raise FileNotFoundError(f"Benchmark database not found: {database_path}")

    with sqlite3.connect(database_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                id,
                dataset,
                task,
                embedder,
                model,
                hyperparams,
                library_hash,
                cv_metric_name,
                cv_metric,
                test_metric_name,
                test_metric
            FROM classificationreport
            ORDER BY id
            """,
            conn,
        )

    out = to_praski_schema(frame)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
