"""Audit prepared benchmark SMILES against the shipped APE vocabulary without loading a model.

The archived checkpoint tokenizers silently drop SELFIES component dots; the
current tokenizer reports them as unknown. Counts are by prepared input row and
split, before model embedding. They do not establish historical result coverage.
"""

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import selfies as sf

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.tokenization_ape import ape_tokenize


def audit_smiles(
    smiles: object, vocab: dict[str, int], max_span: int, max_length: int
) -> dict[str, int]:
    counts = {"conversion_failure": 0, "disconnected": 0, "unknown": 0, "truncated": 0}
    if not isinstance(smiles, str) or not smiles.strip():
        counts["conversion_failure"] = 1
        return counts
    try:
        encoded = sf.encoder(smiles)
    except Exception:
        counts["conversion_failure"] = 1
        return counts
    if not encoded:
        counts["conversion_failure"] = 1
        return counts
    tokens = ape_tokenize(encoded, vocab, "SELFIES", max_piece_span=max_span)
    counts["disconnected"] = int("." in encoded)
    counts["unknown"] = int("<unk>" in tokens)
    counts["truncated"] = int(len(tokens) + 2 > max_length)
    return counts


def audit_dataset(dataset: Dataset, vocab: dict[str, int], max_span: int, max_length: int):
    n = len(dataset.data)
    index_split = [None] * n
    for split, indices in dataset.splits.items():
        for index in indices:
            position = int(index)
            if not 0 <= position < n:
                raise ValueError(f"{dataset.name}: {split} index {position} out of range")
            if index_split[position] is not None:
                raise ValueError(f"{dataset.name}: input {position} belongs to two splits")
            index_split[position] = str(split)
    # Prepared files can retain rows omitted from every benchmark split.
    # Keep them visible as "unspecified" instead of assigning them to test.

    @lru_cache(maxsize=500_000)
    def inspect(smiles):
        return audit_smiles(smiles, vocab, max_span, max_length)

    results = {}
    for smiles, split in zip(dataset.data["smiles"], index_split, strict=True):
        key = split or "unspecified"
        row = results.setdefault(
            key,
            {
                "dataset": dataset.name,
                "split": key,
                "n_inputs": 0,
                "n_conversion_failure": 0,
                "n_disconnected": 0,
                "n_unknown_strict_parser": 0,
                "n_truncated": 0,
            },
        )
        row["n_inputs"] += 1
        for field, count in inspect(smiles).items():
            output = "n_unknown_strict_parser" if field == "unknown" else f"n_{field}"
            row[output] += count
    return list(results.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--vocab", type=Path, default=Path("tokenizer/chembl36_selfies_2m_ape_max2_min3000.json")
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_length < 3:
        raise ValueError("max-length must leave room for BOS, content, and EOS")
    raw = args.vocab.read_bytes()
    payload = json.loads(raw)
    vocab = payload.get("vocab", payload)
    symbols = re.compile(r"\[[^\]]+\]|\.")
    max_span = max((len(symbols.findall(token)) for token in vocab), default=1)
    paths = sorted(args.prepared_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No prepared benchmark JSON files in {args.prepared_dir}")
    rows = []
    for path in paths:
        dataset = Dataset.deserialize_legacy(path)
        result = audit_dataset(dataset, vocab, max_span, args.max_length)
        rows.extend(result)
        print(f"{dataset.name}: {result}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "n_inputs",
        "n_conversion_failure",
        "n_disconnected",
        "n_unknown_strict_parser",
        "n_truncated",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Vocabulary SHA256: {hashlib.sha256(raw).hexdigest()}")
    print(f"Wrote {len(rows)} split rows to {args.output}")


if __name__ == "__main__":
    main()
