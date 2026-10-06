"""Count use of benchmark-derived SELFIES vocabulary symbols in local corpora.

Reports input molecules containing any injected symbol for each prepared
benchmark split, plus symbol prevalence in the ChEMBL training split.
No model is loaded. Counts concern current local files, not historical run IDs.
"""

import argparse
from collections import Counter
import csv
from functools import lru_cache
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
import selfies as sf

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset


def load_symbols(path: Path) -> list[str]:
    symbols = [line.strip() for line in path.read_text().splitlines()]
    symbols = [symbol for symbol in symbols if symbol and not symbol.startswith("#")]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("Injected symbol list is empty or contains duplicates")
    return symbols


def count_benchmark(
    prepared_dir: Path, symbol_pattern: re.Pattern[str]
) -> tuple[list[dict[str, object]], Counter[str]]:
    @lru_cache(maxsize=500_000)
    def encode_symbols(smiles: object) -> tuple[str, ...] | None:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        try:
            encoded = sf.encoder(smiles)
        except Exception:
            return None
        if not encoded:
            return None
        return tuple(symbol_pattern.findall(encoded))

    rows = []
    global_molecule_counts: Counter[str] = Counter()
    paths = [
        p
        for p in sorted(prepared_dir.glob("*.json"))
        if not p.name.endswith(".manifest.json") and p.name != "migration_manifest.json"
    ]
    if not paths:
        raise FileNotFoundError(f"No prepared datasets in {prepared_dir}")
    for path in paths:
        dataset = Dataset.deserialize_legacy(path)
        row_split = ["unspecified"] * len(dataset.data)
        for split, indices in dataset.splits.items():
            for index in indices:
                position = int(index)
                if not 0 <= position < len(row_split):
                    raise ValueError(f"{dataset.name}: split index {position} is out of range")
                if row_split[position] != "unspecified":
                    raise ValueError(f"{dataset.name}: row {position} belongs to two splits")
                row_split[position] = str(split)
        by_split: dict[str, Counter[str]] = {}
        for smiles, split in zip(dataset.data["smiles"], row_split, strict=True):
            counts = by_split.setdefault(split, Counter())
            counts["n_inputs"] += 1
            symbols = encode_symbols(smiles)
            if symbols is None:
                counts["n_conversion_failure"] += 1
                continue
            if symbols:
                counts["n_with_any_injected_symbol"] += 1
                global_molecule_counts.update(set(symbols))
        for split, counts in sorted(by_split.items()):
            rows.append({"dataset": dataset.name, "split": split, **counts})
        print(dataset.name, flush=True)
    return rows, global_molecule_counts


def count_pretraining(parquet_path: Path, symbol_pattern: re.Pattern[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    file = pq.ParquetFile(parquet_path)
    for batch in file.iter_batches(batch_size=32768, columns=["selfies"]):
        for value in batch.column(0).to_pylist():
            counts["n_inputs"] += 1
            if not isinstance(value, str):
                counts["n_empty_or_nonstring"] += 1
                continue
            symbols = symbol_pattern.findall(value)
            if symbols:
                counts["n_with_any_injected_symbol"] += 1
                counts.update(set(symbols))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        type=Path,
        default=Path("tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt"),
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--pretrain-parquet",
        type=Path,
        default=Path("data/pretrain/chembl36_selfies/train.parquet"),
    )
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--symbol-output", type=Path, required=True)
    args = parser.parse_args()
    symbols = load_symbols(args.symbols)
    pattern = re.compile(
        "|".join(re.escape(symbol) for symbol in sorted(symbols, key=len, reverse=True))
    )
    benchmark, benchmark_symbols = count_benchmark(args.prepared_dir, pattern)
    training = count_pretraining(args.pretrain_parquet, pattern)
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    for row in benchmark:
        row.setdefault("n_conversion_failure", 0)
        row.setdefault("n_with_any_injected_symbol", 0)
    with args.benchmark_output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "split",
                "n_inputs",
                "n_conversion_failure",
                "n_with_any_injected_symbol",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark)
    args.symbol_output.parent.mkdir(parents=True, exist_ok=True)
    with args.symbol_output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "n_train_molecules_with_symbol", "n_benchmark_rows_with_symbol"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "symbol": symbol,
                "n_train_molecules_with_symbol": training[symbol],
                "n_benchmark_rows_with_symbol": benchmark_symbols[symbol],
            }
            for symbol in symbols
        )
    print(
        json.dumps(
            {
                "injected_symbols": len(symbols),
                "pretraining_rows": training["n_inputs"],
                "pretraining_molecules_with_any_injected_symbol": training[
                    "n_with_any_injected_symbol"
                ],
                "benchmark_split_rows": len(benchmark),
                "benchmark_molecules_with_any_injected_symbol": sum(
                    int(row.get("n_with_any_injected_symbol", 0)) for row in benchmark
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
