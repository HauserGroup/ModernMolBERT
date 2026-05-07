#!/usr/bin/env python3
"""Create a fast SELFIES symbol-level tokenizer vocabulary.

This does not train APE merges. It simply collects bracketed SELFIES symbols
from PubChem10M_SMILES_SELFIES and writes a tokenizer JSON compatible with
APETokenizer.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SELFIES_RE = re.compile(r"\[[^\]]+\]")

SPECIAL_TOKENS = {
    "<s>": 0,
    "<pad>": 1,
    "</s>": 2,
    "<unk>": 3,
    "<mask>": 4,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default="mikemayuare/PubChem10M_SMILES_SELFIES"
    )
    parser.add_argument(
        "--output_vocab_path", default="tokenizer/selfies_ape_tokenizer.json"
    )
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    output_vocab_path = Path(args.output_vocab_path)
    output_vocab_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_name, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    counts = Counter()
    seen = 0

    for row in ds:
        s = str(row.get("SELFIES", "")).strip()
        if not s:
            continue

        tokens = SELFIES_RE.findall(s)
        if not tokens:
            continue

        counts.update(tokens)
        seen += 1

        if seen >= args.n:
            break

    if not counts:
        raise RuntimeError("No SELFIES tokens found. Check dataset column/schema.")

    vocab = dict(SPECIAL_TOKENS)

    # Stable order: most common first, then lexical for ties.
    sorted_tokens = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    for token, _freq in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    output_vocab_path.write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sha = file_sha256(output_vocab_path)
    metadata = {
        "representation": "SELFIES",
        "tokenizer_type": "SELFIES_SYMBOL_LEVEL",
        "tokenizer_sha256": sha,
        "dataset_name": args.dataset_name,
        "sample_size": seen,
        "num_selfies_symbols": len(counts),
        "vocab_size": len(vocab),
        "note": "Fast interim tokenizer: bracketed SELFIES symbols only; no APE merges.",
    }

    metadata_path = output_vocab_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote vocab:    {output_vocab_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Sampled rows:   {seen}")
    print(f"SELFIES symbols:{len(counts)}")
    print(f"Vocab size:     {len(vocab)}")
    print(f"SHA256:         {sha}")


if __name__ == "__main__":
    main()
