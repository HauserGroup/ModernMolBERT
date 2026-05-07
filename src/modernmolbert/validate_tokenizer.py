#!/usr/bin/env python3
"""Validate SELFIES tokenizer quality before model training."""

import argparse
from pathlib import Path

from modernmolbert.ape_tokenizer import APETokenizer
from modernmolbert.utils import (
    SELFIES_REPRESENTATION,
    assert_metadata_representation,
    compute_tokenization_stats,
    default_selfies_tokenizer_path,
    encode_sequence,
    file_sha256,
    get_streaming_dataset,
    load_tokenizer_metadata,
    metadata_path_for_vocab,
    normalize_sequence,
    resolve_special_ids,
    sample_jsonl_sequences,
    tokenizer_vocab_size,
    validate_selfies_sample_shape,
)

DATASET_NAME = "mikemayuare/PubChem10M_SMILES_SELFIES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate tokenizer metadata and tokenization quality for SELFIES.",
    )
    parser.add_argument(
        "--tokenizer_vocab_path",
        type=str,
        default=str(default_selfies_tokenizer_path()),
    )
    parser.add_argument(
        "--tokenizer_metadata_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=[SELFIES_REPRESENTATION],
        default=SELFIES_REPRESENTATION,
    )
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--unk_rate_threshold", type=float, default=0.001)
    parser.add_argument("--truncation_warn_threshold", type=float, default=0.05)
    parser.add_argument(
        "--fixture_jsonl",
        type=str,
        default=None,
        help="Optional local JSONL file for offline validation.",
    )
    return parser.parse_args()


def _sample_sequences(args: argparse.Namespace) -> list[str]:
    if args.fixture_jsonl:
        return sample_jsonl_sequences(
            Path(args.fixture_jsonl),
            representation=args.representation,
            n=args.n,
        )

    ds = get_streaming_dataset(
        dataset_name=args.dataset_name,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
    )
    rows: list[str] = []
    for row in ds:
        seq = normalize_sequence(row, args.representation)
        if seq is None:
            continue
        rows.append(seq)
        if len(rows) >= args.n:
            break
    return rows


def _assert_ethanol_not_unknown(
    tokenizer: APETokenizer, special_ids: dict[str, int]
) -> None:
    ethanol = "[C][C][O]"
    encoded = encode_sequence(tokenizer, ethanol, max_seq_length=256)["input_ids"]
    non_special = [x for x in encoded if x not in set(special_ids.values())]
    if not non_special:
        raise ValueError("Tokenizer produced no usable tokens for ethanol SELFIES.")
    unk_id = special_ids["unk_token"]
    unk_rate = sum(1 for x in non_special if x == unk_id) / len(non_special)
    if unk_rate > 0.05:
        raise ValueError(
            f"Tokenizer encodes [C][C][O] with high <unk> rate: {unk_rate:.4f}"
        )


def main() -> None:
    args = parse_args()

    vocab_path = Path(args.tokenizer_vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocabulary not found: {vocab_path}")

    metadata_path = (
        Path(args.tokenizer_metadata_path)
        if args.tokenizer_metadata_path
        else metadata_path_for_vocab(vocab_path)
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Tokenizer metadata missing. Expected file: {metadata_path}"
        )

    metadata = load_tokenizer_metadata(metadata_path)
    assert_metadata_representation(
        metadata, expected_representation=args.representation
    )

    recorded_sha = str(metadata.get("tokenizer_sha256", ""))
    actual_sha = file_sha256(vocab_path)
    if recorded_sha and recorded_sha != actual_sha:
        raise ValueError(
            "Tokenizer hash mismatch between metadata and file. "
            f"metadata={recorded_sha} file={actual_sha}"
        )

    tokenizer = APETokenizer()
    tokenizer.load_vocabulary(str(vocab_path))

    special_ids = resolve_special_ids(tokenizer)
    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size < 100:
        raise ValueError(f"Suspiciously small vocabulary size: {vocab_size}")

    sequences = _sample_sequences(args)
    validate_selfies_sample_shape(sequences)
    _assert_ethanol_not_unknown(tokenizer, special_ids)

    stats = compute_tokenization_stats(
        tokenizer=tokenizer,
        sequences=sequences,
        max_seq_length=args.max_seq_length,
        special_ids=special_ids,
    )

    if stats["unk_rate"] > args.unk_rate_threshold:
        raise ValueError(
            f"Tokenizer unknown-token rate too high: {stats['unk_rate']:.6f} "
            f"(threshold {args.unk_rate_threshold:.6f})"
        )
    if stats["empty_sequence_rate"] > 0.0:
        raise ValueError("Tokenizer produced empty tokenized sequences.")
    if stats["mostly_unknown_rate"] > 0.01:
        raise ValueError(
            "Too many sequences are mostly unknown tokens: "
            f"{stats['mostly_unknown_rate']:.4f}"
        )

    print(f"representation: {args.representation}")
    print(f"tokenizer_path: {vocab_path}")
    print(f"tokenizer_metadata_path: {metadata_path}")
    print(f"vocab_size: {vocab_size}")
    print(f"special_ids: {special_ids}")
    print(f"sample_size: {int(stats['sample_size'])}")
    print(f"unk_rate: {stats['unk_rate']:.6f}")
    print(f"mean_len: {stats['mean_len']:.2f}")
    print(f"p50_len: {stats['p50_len']:.0f}")
    print(f"p95_len: {stats['p95_len']:.0f}")
    print(f"p99_len: {stats['p99_len']:.0f}")
    print(f"truncation_rate@{args.max_seq_length}: {stats['truncation_rate']:.6f}")

    if stats["truncation_rate"] > args.truncation_warn_threshold:
        print(
            "warning: truncation rate is above threshold "
            f"({stats['truncation_rate']:.4f} > {args.truncation_warn_threshold:.4f})"
        )


if __name__ == "__main__":
    main()
