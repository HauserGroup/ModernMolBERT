import os
from pathlib import Path

import pandas as pd

PRASKI_COLUMNS = [
    "id",
    "dataset",
    "task",
    "embedder",
    "pooling",
    "pooling_special_tokens_excluded",
    "embedding_model_dir",
    "embedding_tokenizer_path",
    "embedding_max_seq_length",
    "model",
    "hyperparams",
    "library_hash",
    "missing_labels",
    "scoring_identity",
    "prepared_data_sha256",
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
    except pd.errors.EmptyDataError:
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


def append_result_row(
    output_csv: str | Path, row: dict, *, replace_existing: bool = False
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    existing = read_results_csv(output_csv)
    if replace_existing:
        mask = result_mask(
            existing,
            dataset=row["dataset"],
            embedder=row["embedder"],
            cv_metric_name=row["cv_metric_name"],
            model=row["model"],
        )
        existing = existing.loc[~mask].copy()
    next_id = next_result_id(existing)

    row_frame = to_praski_schema(pd.DataFrame([{**row, "id": next_id}]))
    out = pd.concat([existing, row_frame], ignore_index=True)
    out = to_praski_schema(out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    out.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, output_csv)
    return out
