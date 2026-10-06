#!/usr/bin/env python3
"""Validate an APE or BPE tokenizer before model training.

Checks the file against its metadata, then tokenizes a sample of molecules and
reports unknown-token, silent-loss, length and truncation statistics.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from tqdm.auto import tqdm

from modernmolbert.tokenization.load import (
    SMIRK,
    load_verified_tokenizer,
    tokenizer_algorithm,
    tokenizer_representation,
)
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    EXPECTED_SPECIAL_IDS,
    SELFIES_REPRESENTATION,
    SMILES_REPRESENTATION,
    assert_metadata_representation,
    assert_representation_compatible,
    compute_tokenization_stats,
    encode_sequence,
    get_streaming_dataset,
    infer_molecule_column,
    normalize_sequence,
    resolve_special_ids,
    sample_jsonl_sequences,
    tokenizer_vocab_size,
    validate_sample_shape,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DATASET_NAME = PUBCHEM10M_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate tokenizer metadata and tokenization quality.",
    )
    parser.add_argument(
        "--tokenizer_vocab_path",
        type=str,
        required=True,
        help="Tokenizer file: an APE vocabulary JSON or a BPE/SMIRK tokenizer.json.",
    )
    parser.add_argument(
        "--tokenizer_metadata_path",
        type=str,
        default=None,
        help="Tokenizer metadata JSON. Defaults to <file>.metadata.json.",
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=[SELFIES_REPRESENTATION, SMILES_REPRESENTATION],
        default=None,
        help="Expected representation; fails if the metadata records another. "
        "Defaults to the metadata's.",
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
    parser.add_argument("--mostly_unknown_threshold", type=float, default=0.01)
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
            column=args.molecule_column,
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
    tokenizer: "PreTrainedTokenizerBase",
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

    # A hash mismatch always fails, even under --warn_only.
    tokenizer, metadata, vocab_path, metadata_path = load_verified_tokenizer(
        args.tokenizer_vocab_path,
        args.tokenizer_metadata_path,
        log=lambda message: print(message, flush=True),
    )
    if args.representation is not None:
        assert_metadata_representation(metadata, expected_representation=args.representation)
    args.representation = tokenizer_representation(metadata)
    args.molecule_column = infer_molecule_column(
        args.dataset_name, args.representation, args.molecule_column
    )

    special_ids = resolve_special_ids(tokenizer)
    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size < 100:
        raise ValueError(f"Suspiciously small vocabulary size: {vocab_size}")

    warning_count = 0

    if tokenizer_algorithm(metadata) != SMIRK and special_ids != EXPECTED_SPECIAL_IDS:
        warning_count += int(
            _fail_or_warn(
                args,
                f"Unexpected special token IDs: {special_ids}; expected {EXPECTED_SPECIAL_IDS}. "
                "Model config and inference depend on these positions.",
            )
        )
    metadata_special_ids = metadata.get("special_ids")
    if tokenizer_algorithm(metadata) == SMIRK and not isinstance(metadata_special_ids, dict):
        warning_count += int(_fail_or_warn(args, "SMIRK metadata must pin its special token IDs."))
    if metadata_special_ids is not None and metadata_special_ids != special_ids:
        warning_count += int(
            _fail_or_warn(
                args,
                f"Tokenizer special IDs {special_ids} disagree with metadata {metadata_special_ids}.",
            )
        )

    source = args.fixture_jsonl or args.dataset_name
    sequences = _sample_sequences(args)
    if not sequences:
        _fail_or_warn(args, f"No sequences sampled from {source}; nothing to validate.")
        return

    try:
        validate_sample_shape(sequences, args.representation)
    except ValueError as exc:
        warning_count += int(_fail_or_warn(args, str(exc)))

    try:
        assert_representation_compatible(tokenizer, special_ids, args.representation)
    except ValueError as exc:
        warning_count += int(_fail_or_warn(args, str(exc)))

    try:
        stats = compute_tokenization_stats(
            tokenizer=tokenizer,
            sequences=sequences,
            max_seq_length=args.max_seq_length,
            special_ids=special_ids,
        )
    except ValueError as exc:
        _fail_or_warn(args, str(exc))
        return

    print(f"algorithm: {tokenizer_algorithm(metadata)}")
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
    print(f"silent_loss_rate: {stats['silent_loss_rate']:.6f}")
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
    if stats["silent_loss_rate"] > 0.0:
        warning_count += int(
            _fail_or_warn(
                args,
                "Tokenization silently changed some inputs (no <unk>, but the tokens do not "
                f"rebuild the string): rate {stats['silent_loss_rate']:.6f}",
            )
        )
    if stats["empty_sequence_rate"] > 0.0:
        warning_count += int(_fail_or_warn(args, "Tokenizer produced empty tokenized sequences."))
    if stats["mostly_unknown_rate"] > args.mostly_unknown_threshold:
        warning_count += int(
            _fail_or_warn(
                args,
                "Too many sequences are mostly unknown tokens: "
                f"{stats['mostly_unknown_rate']:.4f} "
                f"(threshold {args.mostly_unknown_threshold:.4f})",
            )
        )

    if stats["truncation_rate"] > args.truncation_warn_threshold:
        # Soft signal by design (threshold name says "warn"): never hard-fails,
        # but is counted so the --warn_only summary stays accurate.
        print(
            "WARNING: truncation rate is above threshold "
            f"({stats['truncation_rate']:.4f} > {args.truncation_warn_threshold:.4f})",
            flush=True,
        )
        warning_count += 1

    if warning_count > 0 and args.warn_only:
        print(f"validation completed with {warning_count} warning(s)", flush=True)


if __name__ == "__main__":
    main()
