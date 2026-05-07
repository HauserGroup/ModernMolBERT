#!/usr/bin/env python3
"""Train an APE tokenizer for SELFIES and emit metadata."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from modernmolbert.ape_tokenizer import APETokenizer
from modernmolbert.utils import (
    SELFIES_REPRESENTATION,
    collect_corpus_for_tokenizer,
    default_selfies_tokenizer_path,
    file_sha256,
    metadata_path_for_vocab,
    resolve_special_ids,
    tokenizer_vocab_size,
    validate_selfies_sample_shape,
    write_tokenizer_metadata,
)

DATASET_NAME = "mikemayuare/PubChem10M_SMILES_SELFIES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SELFIES APE tokenizer and save metadata.",
    )
    parser.add_argument(
        "--output_vocab_path",
        type=str,
        default=str(default_selfies_tokenizer_path()),
        help="Where to write tokenizer vocabulary JSON.",
    )
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--representation",
        type=str,
        choices=[SELFIES_REPRESENTATION],
        default=SELFIES_REPRESENTATION,
    )
    parser.add_argument("--tokenizer_train_size", type=int, default=2_000_000)
    parser.add_argument("--max_vocab_size", type=int, default=5000)
    parser.add_argument("--min_freq_for_merge", type=int, default=2000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--ape_source",
        type=str,
        default="modernmolbert.local",
        help="Version/commit descriptor for tokenizer implementation provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_vocab_path = Path(args.output_vocab_path)
    output_vocab_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = collect_corpus_for_tokenizer(
        dataset_name=args.dataset_name,
        representation=args.representation,
        n=args.tokenizer_train_size,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
    )
    validate_selfies_sample_shape(corpus[: min(512, len(corpus))])

    tokenizer = APETokenizer()
    tokenizer.train(
        corpus,
        max_vocab_size=args.max_vocab_size,
        min_freq_for_merge=args.min_freq_for_merge,
        save_checkpoint=False,
    )
    tokenizer.save_vocabulary(str(output_vocab_path))

    vocab_size = tokenizer_vocab_size(tokenizer)
    special_ids = resolve_special_ids(tokenizer)
    metadata_path = metadata_path_for_vocab(output_vocab_path)

    metadata = {
        "representation": SELFIES_REPRESENTATION,
        "dataset_name": args.dataset_name,
        "tokenizer_train_size": args.tokenizer_train_size,
        "max_vocab_size": args.max_vocab_size,
        "min_freq_for_merge": args.min_freq_for_merge,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "seed": args.seed,
        "ape_source": args.ape_source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "tokenizer_path": str(output_vocab_path),
        "tokenizer_sha256": file_sha256(output_vocab_path),
        "creation_command": "python -m modernmolbert.train_ape_tokenizer",
    }
    write_tokenizer_metadata(metadata_path, metadata)

    print("Tokenizer training complete.")
    print(f"Tokenizer vocabulary: {output_vocab_path}")
    print(f"Tokenizer metadata: {metadata_path}")
    print(f"Vocab size: {vocab_size}")


if __name__ == "__main__":
    main()
