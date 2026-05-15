#!/usr/bin/env python3
"""Create an extra-vocabulary SELFIES symbol file for tokenizer training.
This utility filters a SELFIES symbol-count TSV against an existing tokenizer
vocabulary and writes missing symbols above a frequency threshold. The output is
intended for `train_ape_tokenizer --extra_vocab_symbols_path`.

# This script takes:
1. an existing tokenizer vocabulary JSON
2. a benchmark SELFIES symbol-count TSV

and writes a plain text file containing only symbols that are:

- absent from the tokenizer vocabulary
- observed at least `--min_count` times

The output is intended for:

    --extra_vocab_symbols_path

in `modernmolbert.train_ape_tokenizer`.

Input TSV format:
    symbol<TAB>count

Example:
    [C@@H1]    18109
    [C@H1]     17718
    [/C]       15660

Output format:
    one primitive SELFIES symbol per line

Example:
    [C@@H1]
    [C@H1]
    [/C]

Recommended cutoff:
    --min_count 10

Rationale:
    Common stereochemistry and directional-bond SELFIES symbols should not map to
    `<unk>`. Very rare one-off metals, isotopes, or unusual charged species can
    usually remain unknown unless they are frequent enough to matter.

Important:
    The output should contain primitive SELFIES symbols only, not full SELFIES
    molecule strings.

```bash

uv run python -m modernmolbert.tokenization.filter_missing_selfies_symbols \
  --vocab tokenizer/chembl36_selfies_2m_ape_tokenizer.json \
  --symbol_counts tokenizer/extra_symbols/benchmark_selfies_symbol_counts.tsv \
  --output tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt \
  --min_count 10

```
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument(
        "--symbol_counts",
        type=Path,
        required=True,
        help="TSV with columns: symbol, count. Header optional.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_count", type=int, default=10)
    return parser.parse_args()


def read_symbol_counts(path: Path) -> list[tuple[str, int]]:
    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        symbol, count_text = parts[0], parts[1]

        if symbol.lower() == "symbol":
            continue

        try:
            count = int(count_text)
        except ValueError:
            continue

        rows.append((symbol, count))

    return rows


def main() -> None:
    args = parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    counts = read_symbol_counts(args.symbol_counts)

    selected = [
        (symbol, count)
        for symbol, count in counts
        if count >= args.min_count and symbol not in vocab
    ]

    selected = sorted(selected, key=lambda x: (-x[1], x[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        f.write("# Missing SELFIES primitive symbols selected from benchmark diagnostics.\n")
        f.write(f"# min_count={args.min_count}\n")
        for symbol, _count in selected:
            f.write(f"{symbol}\n")

    print(f"Read symbols:       {len(counts)}")
    print(f"Selected missing:   {len(selected)}")
    print(f"Output:             {args.output}")
    print("Top selected:")
    for symbol, count in selected[:50]:
        print(f"  {symbol}\t{count}")


if __name__ == "__main__":
    main()
