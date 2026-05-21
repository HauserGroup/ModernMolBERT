#!/usr/bin/env python3
"""Train an APE tokenizer for SELFIES and emit metadata.

# Conservative
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max4.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 500000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000 \
  --max_merge_pieces 4 \
  --extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt \
  --seed 42

# Moderate
  uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max8.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 500000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000 \
  --max_merge_pieces 8 \
  --extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt \
  --seed 42

# Final
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 2000 \
  --min_freq_for_merge 3000 \
  --max_merge_pieces 2 \
  --extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt \
  --seed 42


# Validate
uv run python -m modernmolbert.validate_tokenizer \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --split train \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.metadata.json \
  --n 10000 \
  --max_seq_length 256

# Ideally
mean_len: 25–60
p95_len: <150
unk_rate: 0
truncation_rate@256: ~0

# Or

unk_rate = 0
mostly_unknown_rate = 0
truncation_rate@256 ≈ 0
mean_len not absurdly low
p95 comfortably below 256

max8: mean_len ≈ 20–50, p95 < 150
max4: mean_len ≈ 35–80, p95 < 200

mean_len still around 10–15 for max8
  → still too compressed

mean_len above 100 with high p95 for max4
  → maybe too fragmented/slow

unk_rate > 0
  → coverage problem

large difference between ChEMBL validation and benchmark molecules
  → add missing symbols or broaden tokenizer corpus

"""

import argparse
import re
from datetime import UTC, datetime
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
SELFIES_SYMBOL_RE = re.compile(r"\[[^\]]+\]")


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
    # Temporarily disabled: current implementation only writes vocab snapshots
    # and does not support true resumeable tokenizer training checkpoints.
    # parser.add_argument(
    #     "--save_checkpoint",
    #     action="store_true",
    #     help="Periodically save intermediate tokenizer checkpoints during APE merge training.",
    # )
    # parser.add_argument(
    #     "--checkpoint_path",
    #     type=str,
    #     default="tokenizer/checkpoints",
    #     help="Directory where intermediate checkpoints are saved when --save_checkpoint is set.",
    # )
    # parser.add_argument(
    #     "--checkpoint_interval",
    #     type=int,
    #     default=500,
    #     help="Checkpoint interval in learned vocabulary entries.",
    # )
    parser.add_argument(
        "--extra_vocab_symbols_path",
        type=Path,
        default=None,
        help=(
            "Optional text file with one primitive SELFIES token per line, e.g. "
            "[C@@H1]. These tokens are force-added after APE merge training and "
            "before saving the final vocabulary. Do not pass full SELFIES molecule "
            "strings here."
        ),
    )
    parser.add_argument(
        "--extra_vocab_selfies_path",
        type=Path,
        default=None,
        help=(
            "Optional text file with one full SELFIES molecule string per line. "
            "All bracketed primitive SELFIES symbols are extracted and force-added "
            "after APE merge training. Prefer --extra_vocab_symbols_path when you "
            "already have a symbol list."
        ),
    )
    parser.add_argument(
        "--max_merge_pieces",
        type=int,
        default=8,
        help=(
            "Maximum number of primitive SELFIES/SMILES pieces allowed in one learned "
            "APE merge token. Use 0 or negative to disable."
        ),
    )
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show tqdm progress bar while collecting corpus.",
    )
    parser.add_argument(
        "--ape_source",
        type=str,
        default="modernmolbert.local",
        help="Version/commit descriptor for tokenizer implementation provenance.",
    )
    return parser.parse_args()


def load_extra_vocab_symbols(
    *,
    symbols_path: Path | None,
    selfies_path: Path | None,
) -> list[str]:
    """Load additional vocabulary symbols to force into the tokenizer.

    symbols_path expects one token per line, e.g.:
        [C@@H1]
        [C@H1]
        [/C]

    selfies_path expects one SELFIES string per line. All bracketed SELFIES
    primitive symbols are extracted.
    """

    symbols: set[str] = set()

    if symbols_path is not None:
        for line in symbols_path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                symbols.add(token)

    if selfies_path is not None:
        for line in selfies_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            symbols.update(SELFIES_SYMBOL_RE.findall(text))

    return sorted(symbols)


def validate_selfies_symbols(symbols: list[str]) -> None:
    """Fail early if extra SELFIES symbols are malformed."""

    malformed = [symbol for symbol in symbols if SELFIES_SYMBOL_RE.fullmatch(symbol) is None]
    if malformed:
        examples = ", ".join(malformed[:20])
        raise ValueError(
            "extra vocab symbols must be SELFIES bracket tokens like [C@@H1]. "
            f"Malformed examples: {examples}"
        )


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
        show_progress=args.show_progress,
    )
    print(f"Corpus collected: {len(corpus)} sequences", flush=True)

    validate_selfies_sample_shape(corpus[: min(512, len(corpus))])

    max_merge_pieces = args.max_merge_pieces
    if max_merge_pieces is not None and max_merge_pieces <= 0:
        max_merge_pieces = None

    tokenizer = APEPreTrainedTokenizer(representation=SELFIES_REPRESENTATION)
    tokenizer.train(
        corpus,
        representation=args.representation,
        max_vocab_size=args.max_vocab_size,
        min_freq_for_merge=args.min_freq_for_merge,
        max_merge_pieces=max_merge_pieces,
        # Temporarily disabled until tokenizer checkpointing can support true resume.
        # save_checkpoint=args.save_checkpoint,
        # checkpoint_path=args.checkpoint_path,
        # checkpoint_interval=args.checkpoint_interval,
    )

    extra_symbols = load_extra_vocab_symbols(
        symbols_path=args.extra_vocab_symbols_path,
        selfies_path=args.extra_vocab_selfies_path,
    )
    validate_selfies_symbols(extra_symbols)

    added_extra_symbols = tokenizer.add_tokens_to_vocabulary(extra_symbols)

    if extra_symbols:
        print(
            "Extra vocab coverage: "
            f"requested={len(extra_symbols)}, added={added_extra_symbols}, "
            f"already_present={len(extra_symbols) - added_extra_symbols}",
            flush=True,
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
        "max_merge_pieces": max_merge_pieces,
        "extra_vocab_symbols_path": (
            str(args.extra_vocab_symbols_path)
            if args.extra_vocab_symbols_path is not None
            else None
        ),
        "extra_vocab_selfies_path": (
            str(args.extra_vocab_selfies_path)
            if args.extra_vocab_selfies_path is not None
            else None
        ),
        "extra_vocab_symbols_requested": len(extra_symbols),
        "extra_vocab_symbols_added": added_extra_symbols,
        "creation_command": "python -m modernmolbert.train_ape_tokenizer",
    }
    write_tokenizer_metadata(metadata_path, metadata)

    print("Tokenizer training complete.", flush=True)
    print(f"Tokenizer vocabulary: {output_vocab_path}", flush=True)
    print(f"Tokenizer metadata: {metadata_path}", flush=True)
    print(f"Vocab size: {vocab_size}", flush=True)
    print(f"Vocab SHA256: {vocab_sha256}", flush=True)
    print(f"Dataset: {args.dataset_name} (column: {resolved_column})", flush=True)
    print(f"Training size: {args.tokenizer_train_size}", flush=True)
    print(f"Max vocab: {args.max_vocab_size}, Min freq: {args.min_freq_for_merge}", flush=True)
    print(f"Extra vocab symbols requested: {len(extra_symbols)}", flush=True)
    print(f"Extra vocab symbols added: {added_extra_symbols}", flush=True)


if __name__ == "__main__":
    main()
