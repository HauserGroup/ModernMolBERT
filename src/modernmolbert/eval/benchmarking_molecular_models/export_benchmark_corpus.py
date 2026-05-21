"""Export primitive symbol coverage data from prepared benchmark datasets.

This is a diagnostic/tokenizer-coverage utility, not a benchmark runner.

It reads prepared `.joblib` benchmark datasets and exports primitive token counts
for either SELFIES or SMILES representation.

SELFIES mode (default):
    Converts each SMILES string to SELFIES via sf.encoder(), then extracts
    primitive SELFIES bracket tokens with SELFIES_SYMBOL_RE.

SMILES mode:
    Extracts primitive SMILES tokens directly from the raw SMILES strings
    using SMILES_RE (same regex used by APEPreTrainedTokenizer). No conversion.

The main use case is checking whether a tokenizer trained on the pretraining
corpus maps valid primitive symbols from downstream datasets to `<unk>`.
Missing symbols can be force-added to the tokenizer vocabulary as atomic symbols
after APE merge training.

Recommended workflow (SELFIES):
    1. Export benchmark symbol counts:
        uv run python -m modernmolbert.eval.benchmarking_molecular_models.export_benchmark_corpus \\
          --prepared_dir data/prepared \\
          --output tokenizer/extra_symbols/benchmark_selfies_symbol_counts.tsv \\
          --mode symbol_counts \\
          --split all

    2. Filter to missing, sufficiently common symbols:
        uv run python -m modernmolbert.tokenization.filter_missing_selfies_symbols \\
          --vocab tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \\
          --symbol_counts tokenizer/extra_symbols/benchmark_selfies_symbol_counts.tsv \\
          --output tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt \\
          --min_count 10

    3. Train tokenizer with forced primitive-symbol coverage:
        uv run python -m modernmolbert.train_ape_tokenizer \\
          --output_vocab_path tokenizer/chembl36_selfies_2m_benchmark_covered_ape_tokenizer.json \\
          --dataset_name data/pretrain/chembl36_selfies \\
          --molecule_column selfies \\
          --representation SELFIES \\
          --tokenizer_train_size 2000000 \\
          --max_vocab_size 5000 \\
          --min_freq_for_merge 2000 \\
          --extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt

Recommended workflow (SMILES):
    1. Export benchmark symbol counts:
        uv run python -m modernmolbert.eval.benchmarking_molecular_models.export_benchmark_corpus \\
          --prepared_dir data/prepared \\
          --output tokenizer/extra_symbols/benchmark_smiles_symbol_counts.tsv \\
          --mode symbol_counts \\
          --representation SMILES \\
          --split all

    2. Filter to missing, sufficiently common symbols:
        uv run python -m modernmolbert.tokenization.filter_missing_selfies_symbols \\
          --vocab tokenizer/my_smiles_ape.json \\
          --symbol_counts tokenizer/extra_symbols/benchmark_smiles_symbol_counts.tsv \\
          --output tokenizer/extra_symbols/benchmark_missing_smiles_symbols_min10.txt \\
          --representation SMILES \\
          --min_count 10

    3. Train tokenizer with forced primitive-symbol coverage:
        uv run python -m modernmolbert.train_ape_tokenizer \\
          --output_vocab_path tokenizer/my_smiles_ape_covered.json \\
          --dataset_name data/pretrain/chembl36_selfies \\
          --molecule_column smiles \\
          --representation SMILES \\
          --tokenizer_train_size 2000000 \\
          --max_vocab_size 5000 \\
          --min_freq_for_merge 2000 \\
          --extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_smiles_symbols_min10.txt

Leakage note:
    This utility should be used to add primitive alphabet symbols only.
    It should not be used to train APE merge rules on benchmark molecules, pretrain
    MLM on benchmark molecules, or use benchmark labels/test performance. Adding
    common valid primitives is tokenizer alphabet coverage, not model selection.
"""

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import joblib
import pandas as pd
import selfies as sf

from modernmolbert.tokenization_ape import SMILES_RE


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
        "--representation",
        choices=["SELFIES", "SMILES"],
        default="SELFIES",
        help=(
            "Molecular representation to tokenize. "
            "'SELFIES' (default) converts SMILES to SELFIES then extracts bracket tokens. "
            "'SMILES' extracts primitive SMILES tokens directly without conversion."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["symbols", "selfies", "symbol_counts"],
        default="symbols",
        help=(
            "'symbols' writes primitive tokens, one per line. "
            "'selfies' writes full SELFIES strings, one per line (SELFIES representation only). "
            "'symbol_counts' writes a TSV of symbol\\tcount sorted by count descending "
            "(input for filter_missing_selfies_symbols.py)."
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


def get_dataset_frame(dataset: Any, path: Path) -> pd.DataFrame:
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

    if args.mode == "selfies" and args.representation == "SMILES":
        raise ValueError("--mode selfies is only valid with --representation SELFIES")

    log(f"Prepared dir:   {args.prepared_dir}")
    log(f"Files found:    {len(joblib_files)}")
    log(f"Representation: {args.representation}")
    log(f"Split:          {args.split}")
    log(f"Mode:           {args.mode}")
    log(f"Output:         {args.output}")
    log("")

    seen_smiles: set[str] = set()
    unique_selfies: set[str] = set()
    symbol_counts: Counter[str] = Counter()

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
                extra = (
                    ""
                    if args.representation == "SMILES"
                    else f", selfies={len(unique_selfies):,}, failures={sf_failures:,}"
                )
                log(
                    f"  Progress {path.stem}: selected_rows={row_idx:,}, "
                    f"file_new={file_new:,}, total_new={total_new_smiles:,}, "
                    f"symbols={len(symbol_counts):,}{extra}, elapsed={elapsed:.1f}s"
                )

            if smi in seen_smiles:
                continue

            seen_smiles.add(smi)
            file_new += 1
            total_new_smiles += 1

            if args.representation == "SMILES":
                symbol_counts.update(m.group() for m in SMILES_RE.finditer(smi))
            else:
                selfies = smiles_to_selfies(smi)
                if selfies is None:
                    sf_failures += 1
                    file_failures += 1
                    continue

                if args.mode == "selfies":
                    unique_selfies.add(selfies)
                else:
                    symbol_counts.update(SELFIES_SYMBOL_RE.findall(selfies))

        elapsed_file = time.perf_counter() - file_start
        extra = (
            ""
            if args.representation == "SMILES"
            else f", failures={file_failures:,}, selfies={len(unique_selfies):,}"
        )
        log(
            f"  Done {path.stem}: selected={file_selected:,}, "
            f"new_smiles={file_new:,}, symbols={len(symbol_counts):,}{extra}, "
            f"elapsed={elapsed_file:.1f}s"
        )
        log("")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "symbol_counts":
        with args.output.open("w", encoding="utf-8") as f:
            f.write("symbol\tcount\n")
            for symbol, count in symbol_counts.most_common():
                f.write(f"{symbol}\t{count}\n")
        values = list(symbol_counts.keys())
    elif args.mode == "symbols":
        values = sorted(symbol_counts.keys())
        with args.output.open("w", encoding="utf-8") as f:
            for value in values:
                f.write(f"{value}\n")
    else:
        values = sorted(unique_selfies)
        with args.output.open("w", encoding="utf-8") as f:
            for value in values:
                f.write(f"{value}\n")

    elapsed_total = time.perf_counter() - start_total

    log("Export complete")
    log("===============")
    log(f"Representation:         {args.representation}")
    log(f"Input rows total:       {total_input_rows:,}")
    log(f"Selected rows total:    {total_selected_rows:,}")
    log(f"Unique SMILES:          {len(seen_smiles):,}")
    if args.representation == "SELFIES":
        log(f"SELFIES failures:       {sf_failures:,}")
        log(f"Unique SELFIES:         {len(unique_selfies):,}")
    log(f"Unique symbols:         {len(symbol_counts):,}")
    log(f"Output rows:            {len(values):,}")
    log(f"Output:                 {args.output}")
    log(f"Elapsed:                {elapsed_total:.1f}s")

    log("")
    log("Use this output with:")
    if args.mode in ("symbols", "symbol_counts"):
        log(f"  --extra_vocab_symbols_path {args.output}")
    else:
        log(f"  --extra_vocab_selfies_path {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        raise
