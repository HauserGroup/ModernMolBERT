"""Count tokenizer failures and truncation on Parquet SELFIES corpora, without model loading."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from modernmolbert.tokenization_ape import ape_tokenize


def audit_strings(strings, vocab, *, max_length=128, lengths=None):
    """Count failures; if ``lengths`` is a Counter, also tally token lengths incl. BOS/EOS."""
    counts = Counter()
    unknown_symbols = Counter()
    symbols = re.compile(r"\[[^\]]+\]|\.")
    max_span = max((len(symbols.findall(token)) for token in vocab), default=1)
    for value in strings:
        counts["n_inputs"] += 1
        if not isinstance(value, str) or not value:
            counts["n_empty_or_nonstring"] += 1
            continue
        pieces = symbols.findall(value)
        counts["n_disconnected"] += int("." in value)
        counts["n_malformed"] += int("".join(pieces) != value)
        tokens = ape_tokenize(value, vocab, "SELFIES", max_piece_span=max_span)
        counts["n_unknown"] += int("<unk>" in tokens)
        counts["n_truncated_at_max_length"] += int(len(tokens) + 2 > max_length)
        if lengths is not None:
            lengths[len(tokens) + 2] += 1
        unknown_symbols.update(p for p in pieces if p not in vocab)
    return dict(counts), dict(unknown_symbols)


def length_summary(lengths):
    """Mean, percentiles and maximum of a {length: count} distribution."""
    total = sum(lengths.values())
    if not total:
        return {}
    summary = {
        "n": total,
        "mean": round(sum(length * n for length, n in lengths.items()) / total, 3),
        "max": max(lengths),
    }
    running = 0
    quantiles = iter([(0.5, "p50"), (0.95, "p95"), (0.99, "p99")])
    quantile, name = next(quantiles)
    for length in sorted(lengths):
        running += lengths[length]
        while running >= quantile * total:
            summary[name] = length
            quantile, name = next(quantiles, (2.0, ""))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--column", default="selfies")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.vocab.read_bytes()
    payload = json.loads(raw)
    vocab = payload.get("vocab", payload)
    report = {
        "vocab": str(args.vocab),
        "vocab_sha256": hashlib.sha256(raw).hexdigest(),
        "max_length_including_bos_eos": args.max_length,
        "scope": "Current tokenizer and local data; not proof of historical checkpoint data identity.",
        "inputs": [],
    }
    for path in args.parquet:
        counts = Counter()
        unknown = Counter()
        lengths = Counter()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32768, columns=[args.column]):
            batch_counts, batch_unknown = audit_strings(
                batch.column(0).to_pylist(), vocab, max_length=args.max_length, lengths=lengths
            )
            counts.update(batch_counts)
            unknown.update(batch_unknown)
        result = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "counts": dict(counts),
            "token_length_including_bos_eos": length_summary(lengths),
            "out_of_vocab_primitive_occurrences": dict(unknown),
        }
        report["inputs"].append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
