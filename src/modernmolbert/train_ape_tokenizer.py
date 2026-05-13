#!/usr/bin/env python3
"""Train an APE tokenizer for SELFIES and emit metadata."""

import argparse
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    collect_corpus_for_tokenizer,
    default_selfies_tokenizer_path,
    file_sha256,
    infer_selfies_column,
    metadata_path_for_vocab,
    resolve_special_ids,
    tokenizer_vocab_size,
    validate_selfies_sample_shape,
    write_tokenizer_metadata,
)

DATASET_NAME = PUBCHEM10M_DATASET


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
        "--selfies_column",
        type=str,
        default=None,
        help="Column containing SELFIES strings. Defaults by dataset.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "Local Arrow dataset directory. If omitted, auto-detect a matching dataset in data/."
        ),
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help=(
            "Optional parquet file path or glob to stream directly. "
            "When set, this takes precedence over --dataset_name/--data_dir."
        ),
    )
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
    load_dotenv()
    args = parse_args()
    resolved_column = infer_selfies_column(args.dataset_name, args.selfies_column)

    output_vocab_path = Path(args.output_vocab_path)
    output_vocab_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = collect_corpus_for_tokenizer(
        dataset_name=args.dataset_name,
        representation=resolved_column,
        n=args.tokenizer_train_size,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
        data_files=args.data_files,
    )
    validate_selfies_sample_shape(corpus[: min(512, len(corpus))])

    tokenizer = APEPreTrainedTokenizer(representation=SELFIES_REPRESENTATION)
    tokenizer.train(
        corpus,
        max_vocab_size=args.max_vocab_size,
        min_freq_for_merge=args.min_freq_for_merge,
        save_checkpoint=False,
    )

    # Phase 1: write the vocab to disk.
    tokenizer.save_vocabulary_file(output_vocab_path)

    # Phase 2: compute the SHA from the file that is now on disk.
    vocab_sha256 = file_sha256(output_vocab_path)
    vocab_size = tokenizer_vocab_size(tokenizer)
    special_ids = resolve_special_ids(tokenizer)
    metadata_path = metadata_path_for_vocab(output_vocab_path)

    # Phase 3: write metadata — SHA reflects the final on-disk vocab.
    metadata = {
        "representation": SELFIES_REPRESENTATION,
        "dataset_name": args.dataset_name,
        "selfies_column": resolved_column,
        "tokenizer_train_size": args.tokenizer_train_size,
        "max_vocab_size": args.max_vocab_size,
        "min_freq_for_merge": args.min_freq_for_merge,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "seed": args.seed,
        "ape_source": args.ape_source,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "tokenizer_path": str(output_vocab_path),
        "tokenizer_sha256": vocab_sha256,
        "creation_command": "python -m modernmolbert.train_ape_tokenizer",
    }
    write_tokenizer_metadata(metadata_path, metadata)

    print("Tokenizer training complete.")
    print(f"Tokenizer vocabulary: {output_vocab_path}")
    print(f"Tokenizer metadata: {metadata_path}")
    print(f"Vocab size: {vocab_size}")


if __name__ == "__main__":
    main()
