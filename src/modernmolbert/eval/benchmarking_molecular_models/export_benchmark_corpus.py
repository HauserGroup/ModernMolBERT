#!/usr/bin/env python3
"""Export SELFIES coverage data from prepared benchmark datasets.

Reads .joblib files in the prepared directory and converts benchmark SMILES to
SELFIES.

Two output modes are supported:

1. symbols:
   Writes one primitive SELFIES symbol per line, e.g. [C@@H1].
   Use this with:
       --extra_vocab_symbols_path

2. selfies:
   Writes one full SELFIES string per line.
   Use this with:
       --extra_vocab_selfies_path

Examples:
    uv run python -m modernmolbert.eval.benchmarking_molecular_models.export_benchmark_corpus \
        --prepared_dir data/prepared \
        --output tokenizer/extra_symbols/benchmark_missing_selfies_symbols.txt \
        --mode symbols \
        --split all

    uv run python -m modernmolbert.eval.benchmarking_molecular_models.export_benchmark_corpus \
        --prepared_dir data/prepared \
        --output data/prepared/benchmark_selfies_symbols_source.txt \
        --mode selfies \
        --split all
"""

import argparse
import re
import sys
import time
from pathlib import Path
from collections.abc import Iterable

import joblib
import pandas as pd
import selfies as sf


SELFIES_SYMBOL_RE = re.compile(r"\[[^\]]+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared_dir",
        type=Path,
        default=Path("data/prepared"),
        help="Directory containing prepared .joblib dataset files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer/extra_symbols/benchmark_missing_selfies_symbols.txt"),
        help="Output file path.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "val", "test", "all"],
        default="all",
        help="Which split(s) to include. 'all' includes every row regardless of split.",
    )
    parser.add_argument(
        "--mode",
        choices=["symbols", "selfies"],
        default="symbols",
        help=(
            "'symbols' writes primitive SELFIES tokens, one per line. "
            "'selfies' writes full SELFIES strings, one per line."
        ),
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Report progress every N input rows.",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=0,
        help="Debug option: process at most this many .joblib files. 0 means no limit.",
    )
    parser.add_argument(
        "--limit_rows_per_file",
        type=int,
        default=0,
        help="Debug option: process at most this many rows per file. 0 means no limit.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def get_dataset_frame(dataset: object, path: Path) -> pd.DataFrame:
    """Extract a pandas DataFrame from a prepared benchmark joblib object."""

    if hasattr(dataset, "data"):
        data = dataset.data
    elif isinstance(dataset, dict) and "data" in dataset:
        data = dataset["data"]
    else:
        raise TypeError(
            f"{path.name}: expected object with .data or dict['data'], got {type(dataset).__name__}"
        )

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{path.name}: expected dataset.data to be a pandas DataFrame, "
            f"got {type(data).__name__}"
        )

    return data


def normalize_split_name(split: str) -> str:
    if split == "val":
        return "valid"
    return split


def iter_smiles_for_split(frame: pd.DataFrame, split: str) -> Iterable[str]:
    """Yield SMILES strings for the requested split without materializing a list."""

    if "smiles" not in frame.columns:
        raise ValueError("missing required 'smiles' column")

    split = normalize_split_name(split)

    if split == "all" or "split" not in frame.columns:
        series = frame["smiles"]
    else:
        split_series = frame["split"].astype(str).str.lower()
        if split == "valid":
            mask = split_series.isin(["valid", "val", "validation"])
        else:
            mask = split_series == split
        series = frame.loc[mask, "smiles"]

    for value in series.dropna():
        smi = str(value).strip()
        if smi:
            yield smi


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        selfies = sf.encoder(smiles)
    except Exception:
        return None
    if not selfies:
        return None
    return selfies


def main() -> None:
    args = parse_args()

    start_total = time.perf_counter()

    joblib_files = sorted(args.prepared_dir.glob("*.joblib"))
    if args.limit_files > 0:
        joblib_files = joblib_files[: args.limit_files]

    if not joblib_files:
        raise FileNotFoundError(f"No .joblib files found in {args.prepared_dir}")

    log(f"Prepared dir: {args.prepared_dir}")
    log(f"Files found:  {len(joblib_files)}")
    log(f"Split:        {args.split}")
    log(f"Mode:         {args.mode}")
    log(f"Output:       {args.output}")
    log("")

    seen_smiles: set[str] = set()
    unique_selfies: set[str] = set()
    unique_symbols: set[str] = set()

    total_input_rows = 0
    total_selected_rows = 0
    total_new_smiles = 0
    sf_failures = 0

    for file_idx, path in enumerate(joblib_files, start=1):
        file_start = time.perf_counter()
        size_mb = path.stat().st_size / 1e6

        log(f"[{file_idx}/{len(joblib_files)}] Loading {path.name} ({size_mb:.1f} MB) ...")

        try:
            dataset = joblib.load(path)
        except Exception as exc:
            log(f"  ERROR loading {path.name}: {type(exc).__name__}: {exc}")
            continue

        load_seconds = time.perf_counter() - file_start
        log(f"  Loaded in {load_seconds:.1f}s")

        try:
            frame = get_dataset_frame(dataset, path)
        except Exception as exc:
            log(f"  Skipping: {type(exc).__name__}: {exc}")
            continue

        total_input_rows += len(frame)

        if "smiles" not in frame.columns:
            log("  Skipping: no 'smiles' column")
            continue

        log(f"  Rows in file: {len(frame):,}")
        if "split" in frame.columns:
            split_counts = frame["split"].astype(str).str.lower().value_counts().to_dict()
            log(f"  Split counts: {split_counts}")

        file_selected = 0
        file_new = 0
        file_failures = 0

        row_iter = iter_smiles_for_split(frame, args.split)

        for row_idx, smi in enumerate(row_iter, start=1):
            if args.limit_rows_per_file > 0 and row_idx > args.limit_rows_per_file:
                break

            file_selected += 1
            total_selected_rows += 1

            if row_idx % args.progress_every == 0:
                elapsed = time.perf_counter() - file_start
                log(
                    f"  Progress {path.stem}: selected_rows={row_idx:,}, "
                    f"file_new={file_new:,}, total_new={total_new_smiles:,}, "
                    f"symbols={len(unique_symbols):,}, selfies={len(unique_selfies):,}, "
                    f"failures={sf_failures:,}, elapsed={elapsed:.1f}s"
                )

            if smi in seen_smiles:
                continue

            seen_smiles.add(smi)
            file_new += 1
            total_new_smiles += 1

            selfies = smiles_to_selfies(smi)
            if selfies is None:
                sf_failures += 1
                file_failures += 1
                continue

            if args.mode == "selfies":
                unique_selfies.add(selfies)
            else:
                unique_symbols.update(SELFIES_SYMBOL_RE.findall(selfies))

        elapsed_file = time.perf_counter() - file_start
        log(
            f"  Done {path.stem}: selected={file_selected:,}, "
            f"new_smiles={file_new:,}, failures={file_failures:,}, "
            f"symbols={len(unique_symbols):,}, selfies={len(unique_selfies):,}, "
            f"elapsed={elapsed_file:.1f}s"
        )
        log("")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "symbols":
        values = sorted(unique_symbols)
    else:
        values = sorted(unique_selfies)

    with args.output.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")

    elapsed_total = time.perf_counter() - start_total

    log("Export complete")
    log("===============")
    log(f"Input rows total:       {total_input_rows:,}")
    log(f"Selected rows total:    {total_selected_rows:,}")
    log(f"Unique SMILES:          {len(seen_smiles):,}")
    log(f"SELFIES failures:       {sf_failures:,}")
    log(f"Unique SELFIES:         {len(unique_selfies):,}")
    log(f"Unique symbols:         {len(unique_symbols):,}")
    log(f"Output rows:            {len(values):,}")
    log(f"Output:                 {args.output}")
    log(f"Elapsed:                {elapsed_total:.1f}s")

    if args.mode == "symbols":
        log("")
        log("Use this output with:")
        log(f"  --extra_vocab_symbols_path {args.output}")
    else:
        log("")
        log("Use this output with:")
        log(f"  --extra_vocab_selfies_path {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        raise
