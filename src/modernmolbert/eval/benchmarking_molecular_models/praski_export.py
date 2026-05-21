from __future__ import annotations

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

    return pd.DataFrame(out.loc[:, PRASKI_COLUMNS])


def next_result_id(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 1

    ids: list[int] = []
    for value in frame["id"].tolist():
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue

    if not ids:
        return 1

    return max(ids) + 1


def empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRASKI_COLUMNS)


def read_results_csv(output_csv: str | Path) -> pd.DataFrame:
    output_csv = Path(output_csv)
    if not output_csv.exists():
        return empty_results_frame()
    try:
        return to_praski_schema(pd.read_csv(output_csv))
    except Exception:
        return empty_results_frame()


def result_mask(
    frame: pd.DataFrame,
    *,
    dataset: str,
    embedder: str,
    cv_metric_name: str,
    model: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return (
        (frame["dataset"] == dataset)
        & (frame["embedder"] == embedder)
        & (frame["cv_metric_name"] == cv_metric_name)
        & (frame["model"] == model)
    )


def count_result_rows(
    output_csv: str | Path,
    *,
    dataset: str,
    embedder: str,
    cv_metric_name: str,
    model: str,
) -> int:
    frame = read_results_csv(output_csv)
    return int(
        result_mask(
            frame,
            dataset=dataset,
            embedder=embedder,
            cv_metric_name=cv_metric_name,
            model=model,
        ).sum()
    )


def delete_result_rows(
    output_csv: str | Path,
    *,
    dataset: str,
    embedder: str,
    cv_metric_name: str,
    model: str,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    frame = read_results_csv(output_csv)
    mask = result_mask(
        frame,
        dataset=dataset,
        embedder=embedder,
        cv_metric_name=cv_metric_name,
        model=model,
    )
    out = frame.loc[~mask].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def append_result_row(output_csv: str | Path, row: dict) -> pd.DataFrame:
    output_csv = Path(output_csv)
    existing = read_results_csv(output_csv)
    next_id = next_result_id(existing)

    row_frame = to_praski_schema(pd.DataFrame([{**row, "id": next_id}]))
    out = pd.concat([existing, row_frame], ignore_index=True)
    out = to_praski_schema(out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def write_dataset_checkpoint(
    *,
    results_csv: str | Path,
    checkpoint_dir: str | Path,
    dataset: str,
    embedder: str,
) -> pd.DataFrame:
    frame = read_results_csv(results_csv)
    if frame.empty:
        out = empty_results_frame()
    else:
        out = frame.loc[(frame["dataset"] == dataset) & (frame["embedder"] == embedder)].copy()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / f"{dataset}.csv"
    out.to_csv(output_path, index=False)
    return out
