"""Align the sweep result CSV files under results/ into one table.

The current results/sweep_results.csv contains both an older 13-column schema
and newer 28-column rows appended later. This script reads rows by width,
normalizes both schemas to the richer schema, and de-duplicates runs while
keeping the row with the most populated fields.

Usage:
    python scripts/align_sweep_result_csvs.py
    python scripts/align_sweep_result_csvs.py --out results/sweep_results_aligned.csv
"""

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = (
    Path("results/sweep_results.csv"),
    Path("results/sweep_results_base.csv"),
)
DEFAULT_OUT = Path("results/sweep_results_aligned.csv")

SIMPLE_COLUMNS = [
    "run",
    "size",
    "strategy",
    "mlm_prob",
    "learning_rate",
    "eval_loss",
    "eval_masked_accuracy",
    "eval_perplexity",
    "train_loss",
    "epoch",
    "num_parameters",
    "train_runtime",
    "eval_runtime",
]

FULL_COLUMNS = [
    "run",
    "size",
    "strategy",
    "mlm_prob",
    "learning_rate",
    "eval_masking_strategy",
    "eval_mlm_probability",
    "load_best_model_at_end",
    "metric_source",
    "metric_note",
    "best_checkpoint",
    "best_step",
    "best_logged_eval_loss",
    "best_logged_eval_masked_accuracy",
    "last_logged_eval_step",
    "last_logged_eval_loss",
    "last_logged_eval_masked_accuracy",
    "final_eval_loss",
    "final_eval_masked_accuracy",
    "final_eval_perplexity",
    "eval_loss",
    "eval_masked_accuracy",
    "eval_perplexity",
    "train_loss",
    "epoch",
    "num_parameters",
    "train_runtime",
    "eval_runtime",
]

SORT_COLUMNS = ["size", "strategy", "mlm_prob", "learning_rate", "run"]
KEY_COLUMNS = ["run", "size", "strategy", "mlm_prob", "learning_rate"]


def _row_to_record(row: list[str], *, path: Path, line_number: int) -> dict[str, str] | None:
    if not row:
        return None

    if row in (SIMPLE_COLUMNS, FULL_COLUMNS):
        return None

    if len(row) == len(FULL_COLUMNS):
        return dict(zip(FULL_COLUMNS, row, strict=True))

    if len(row) == len(SIMPLE_COLUMNS):
        record = {column: "" for column in FULL_COLUMNS}
        record.update(dict(zip(SIMPLE_COLUMNS, row, strict=True)))
        return record

    raise ValueError(
        f"Unexpected {len(row)} columns in {path}:{line_number}; "
        f"expected {len(SIMPLE_COLUMNS)} or {len(FULL_COLUMNS)}."
    )


def read_records(paths: list[Path]) -> list[dict[str, str]]:
    records = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                record = _row_to_record(row, path=path, line_number=line_number)
                if record is not None:
                    records.append(record)
    return records


def _identity(record: dict[str, str]) -> tuple[str, ...]:
    return tuple(record[column] for column in KEY_COLUMNS)


def _populated_count(record: dict[str, str]) -> int:
    return sum(value != "" for value in record.values())


def dedupe_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for record in records:
        key = _identity(record)
        current = best_by_key.get(key)
        if current is None or _populated_count(record) > _populated_count(current):
            best_by_key[key] = record
    return list(best_by_key.values())


def _sort_value(record: dict[str, str], column: str) -> tuple[int, float | str]:
    value = record[column]
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def sort_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        records,
        key=lambda record: tuple(_sort_value(record, column) for column in SORT_COLUMNS),
    )


def write_records(records: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="CSV files to align. Defaults to the two sweep result CSVs in results/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path. Defaults to {DEFAULT_OUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.inputs)
    aligned = sort_records(dedupe_records(records))
    write_records(aligned, args.out)
    print(f"Wrote {len(aligned)} aligned rows to {args.out} from {len(records)} input rows.")


if __name__ == "__main__":
    main()
