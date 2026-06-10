#!/usr/bin/env python3
"""Validate APE tokenizer quality before model training."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    SMILES_REPRESENTATION,
    assert_metadata_representation,
    assert_representation_compatible,
    compute_tokenization_stats,
    default_tokenizer_path,
    encode_sequence,
    file_sha256,
    get_streaming_dataset,
    infer_molecule_column,
    load_tokenizer_metadata,
    metadata_path_for_vocab,
    normalize_sequence,
    resolve_special_ids,
    sample_jsonl_sequences,
    tokenizer_vocab_size,
    validate_sample_shape,
)

DATASET_NAME = PUBCHEM10M_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate APE tokenizer metadata and tokenization quality.",
    )
    parser.add_argument(
        "--tokenizer_vocab_path",
        type=str,
        default=None,
        help="Path to tokenizer vocabulary JSON. Defaults by representation.",
    )
    parser.add_argument(
        "--tokenizer_metadata_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=[SELFIES_REPRESENTATION, SMILES_REPRESENTATION],
        default=SELFIES_REPRESENTATION,
    )
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--molecule_column",
        type=str,
        default=None,
        help="Column containing molecule strings. Defaults by dataset and representation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to sample from.",
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
            "Optional parquet glob/path for direct streaming (e.g. "
            "hf://datasets/<repo>/data/train-*.parquet)."
        ),
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--unk_rate_threshold", type=float, default=0.001)
    parser.add_argument("--truncation_warn_threshold", type=float, default=0.05)
    parser.add_argument(
        "--warn_only",
        action="store_true",
        help="Report validation failures as warnings instead of exiting nonzero.",
    )
    parser.add_argument(
        "--show_unknown_examples",
        type=int,
        default=0,
        help="Print up to N sample sequences that contain <unk> tokens.",
    )
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
            representation=args.molecule_column,
            n=args.n,
        )

    ds = get_streaming_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
        data_files=args.data_files,
    )
    rows: list[str] = []
    try:
        with tqdm(total=args.n, desc="Sampling sequences", unit="seq") as pbar:
            for row in ds:
                seq = normalize_sequence(row, args.molecule_column)
                if seq is None:
                    continue
                rows.append(seq)
                pbar.update(1)
                if len(rows) >= args.n:
                    break
    finally:
        # Best-effort cleanup so streaming layers don't continue retry noise
        # after validation has already reached a terminal state.
        del ds
    return rows


def _fail_or_warn(args: argparse.Namespace, message: str) -> bool:
    if args.warn_only:
        print(f"WARNING: {message}", flush=True)
        return True
    print(f"ERROR: {message}", flush=True)
    raise SystemExit(1)


def _print_unknown_examples(
    tokenizer: APEPreTrainedTokenizer,
    sequences: list[str],
    special_ids: dict[str, int],
    max_seq_length: int,
    n: int,
) -> None:
    if n <= 0:
        return
    unk_id = special_ids["unk_token"]
    shown = 0
    for seq in sequences:
        encoded = encode_sequence(
            tokenizer,
            seq,
            max_seq_length=max_seq_length,
        )["input_ids"]
        if unk_id not in encoded:
            continue
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        print("UNKNOWN EXAMPLE")
        print(f"sequence: {seq[:300]}")
        print(f"ids: {encoded[:80]}")
        print(f"tokens: {tokens[:80]}")
        shown += 1
        if shown >= n:
            break


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.tokenizer_vocab_path is None:
        args.tokenizer_vocab_path = str(default_tokenizer_path(args.representation))

    args.molecule_column = infer_molecule_column(
        args.dataset_name, args.representation, args.molecule_column
    )

    vocab_path = Path(args.tokenizer_vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocabulary not found: {vocab_path}")

    metadata_path = (
        Path(args.tokenizer_metadata_path)
        if args.tokenizer_metadata_path
        else metadata_path_for_vocab(vocab_path)
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Tokenizer metadata missing. Expected file: {metadata_path}")

    metadata = load_tokenizer_metadata(metadata_path)
    assert_metadata_representation(metadata, expected_representation=args.representation)

    recorded_sha = str(metadata.get("tokenizer_sha256", ""))
    actual_sha = file_sha256(vocab_path)
    if recorded_sha and recorded_sha != actual_sha:
        raise ValueError(
            "Tokenizer hash mismatch between metadata and file. "
            f"metadata={recorded_sha} file={actual_sha}"
        )

    tokenizer = APEPreTrainedTokenizer(representation=args.representation)
    tokenizer.load_vocabulary_file(vocab_path)

    special_ids = resolve_special_ids(tokenizer)
    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size < 100:
        raise ValueError(f"Suspiciously small vocabulary size: {vocab_size}")

    warning_count = 0

    sequences = _sample_sequences(args)
    validate_sample_shape(sequences, args.representation)
    try:
        assert_representation_compatible(tokenizer, special_ids, args.representation)
    except ValueError as exc:
        warning_count += int(_fail_or_warn(args, str(exc)))

    stats = compute_tokenization_stats(
        tokenizer=tokenizer,
        sequences=sequences,
        max_seq_length=args.max_seq_length,
        special_ids=special_ids,
    )

    print(f"representation: {args.representation}")
    print(f"molecule_column: {args.molecule_column}")
    print(f"split: {args.split}")
    print(f"dataset_name: {args.dataset_name}")
    if args.data_files:
        print(f"data_files: {args.data_files}")
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

    if stats["unk_rate"] > args.unk_rate_threshold:
        if args.show_unknown_examples > 0:
            _print_unknown_examples(
                tokenizer=tokenizer,
                sequences=sequences,
                special_ids=special_ids,
                max_seq_length=args.max_seq_length,
                n=args.show_unknown_examples,
            )
        warning_count += int(
            _fail_or_warn(
                args,
                "Tokenizer unknown-token rate too high: "
                f"{stats['unk_rate']:.6f} "
                f"(threshold {args.unk_rate_threshold:.6f})",
            )
        )
    if stats["empty_sequence_rate"] > 0.0:
        warning_count += int(_fail_or_warn(args, "Tokenizer produced empty tokenized sequences."))
    if stats["mostly_unknown_rate"] > 0.01:
        warning_count += int(
            _fail_or_warn(
                args,
                f"Too many sequences are mostly unknown tokens: {stats['mostly_unknown_rate']:.4f}",
            )
        )

    if stats["truncation_rate"] > args.truncation_warn_threshold:
        print(
            "warning: truncation rate is above threshold "
            f"({stats['truncation_rate']:.4f} > {args.truncation_warn_threshold:.4f})"
        )

    if warning_count > 0 and args.warn_only:
        print(f"validation completed with {warning_count} warning(s)", flush=True)


if __name__ == "__main__":
    main()
