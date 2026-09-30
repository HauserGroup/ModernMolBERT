#!/usr/bin/env python3
"""Train an APE or BPE tokenizer on SELFIES or SMILES and write its metadata.

APE merges whole primitive symbols: SELFIES bracket symbols, or SMILES atoms,
bonds and ring labels. BPE merges single characters and may split a symbol.
Both learn merges from a sample of the training split. With
``--corpus_primitive_parquet``, every primitive symbol (APE) or character (BPE)
of the full training split also enters the vocabulary, so no training molecule
maps to ``<unk>``.

The vocabulary goes to ``--output_vocab_path`` (an APE vocabulary JSON or a BPE
``tokenizer.json``), with a ``.metadata.json`` next to it that records the
settings and the file's SHA-256. Recipes are in docs/tokenizer.md; check every
new tokenizer with ``python -m modernmolbert.validate_tokenizer``.

Example, the corpus-only APE tokenizer of the revision run:

uv run python -m modernmolbert.train_tokenizer \\
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \\
  --dataset_name data/pretrain/chembl36_selfies \\
  --molecule_column selfies --representation SELFIES \\
  --tokenizer_train_size 2000000 --max_vocab_size 2000 \\
  --min_freq_for_merge 3000 --max_merge_pieces 2 --seed 42 \\
  --corpus_primitive_parquet data/pretrain/chembl36_selfies/train.parquet
"""

import argparse
import hashlib
import re
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from modernmolbert.tokenization.ape import train_ape
from modernmolbert.tokenization.load import ALGORITHMS, APE, BPE, load_tokenizer
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer, pre_tokenize_molecule
from modernmolbert.utils import (
    EXPECTED_SPECIAL_IDS,
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    SMILES_REPRESENTATION,
    SPECIAL_TOKENS,
    assert_special_ids,
    collect_corpus_for_tokenizer,
    file_sha256,
    get_git_revision,
    infer_molecule_column,
    metadata_path_for_vocab,
    resolve_special_ids,
    tokenizer_vocab_size,
    validate_sample_shape,
    write_tokenizer_metadata,
)

DATASET_NAME = PUBCHEM10M_DATASET
SELFIES_SYMBOL_RE = re.compile(r"\[[^\]]+\]")
# The APE cap on primitive symbols per merged token when none is given.
DEFAULT_MAX_MERGE_PIECES = 8


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an APE or BPE molecular tokenizer and save its metadata.",
    )
    parser.add_argument("--algorithm", type=str.upper, choices=ALGORITHMS, default=APE)
    parser.add_argument(
        "--output_vocab_path",
        type=Path,
        required=True,
        help="Tokenizer file to write: an APE vocabulary JSON or a BPE tokenizer.json.",
    )
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--molecule_column",
        type=str,
        default=None,
        help="Column containing molecule strings. Defaults by dataset and representation.",
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
        choices=[SELFIES_REPRESENTATION, SMILES_REPRESENTATION],
        default=SELFIES_REPRESENTATION,
    )
    parser.add_argument("--tokenizer_train_size", type=int, default=2_000_000)
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=5000,
        help=(
            "APE: stop merging at this many tokens, not counting special tokens. "
            "BPE: final vocabulary size, including the five special tokens."
        ),
    )
    parser.add_argument(
        "--min_freq_for_merge",
        type=int,
        default=2000,
        help="Stop merging when the most frequent pair occurs fewer times than this.",
    )
    parser.add_argument(
        "--max_merge_pieces",
        type=int,
        default=None,
        help=(
            "APE only: maximum number of primitive symbols in one merged token "
            f"(default {DEFAULT_MAX_MERGE_PIECES}). Use 0 or negative to disable."
        ),
    )
    parser.add_argument(
        "--corpus_primitive_parquet",
        type=Path,
        default=None,
        help=(
            "Optional local training-split Parquet file. Every primitive symbol (APE) or "
            "character (BPE) in its molecule column enters the vocabulary. Use this for "
            "corpus-only coverage without benchmark-derived symbols."
        ),
    )
    parser.add_argument(
        "--extra_vocab_symbols_path",
        type=Path,
        default=None,
        help=(
            "APE only; off by default. Text file with one primitive token per line "
            "(SELFIES: e.g. [C@@H1]; SMILES: e.g. [Fe+3]). These tokens are "
            "force-added after APE merge training and before saving the final "
            "vocabulary. Do not pass full molecule strings here. For SELFIES "
            "representation, tokens are validated as bracket tokens."
        ),
    )
    parser.add_argument(
        "--extra_vocab_selfies_path",
        type=Path,
        default=None,
        help=(
            "APE only; off by default. Text file with one full SELFIES molecule "
            "string per line. All bracketed primitive SELFIES symbols are extracted "
            "and force-added after APE merge training. Prefer "
            "--extra_vocab_symbols_path when you already have a symbol list."
        ),
    )
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--aligned_sample_parquet",
        type=Path,
        default=None,
        help="Local paired-molecule Parquet for one shared row-index sample across representations.",
    )
    parser.add_argument(
        "--sample_indices_path",
        type=Path,
        default=None,
        help="Persist/reuse the exact zero-based row indices of --aligned_sample_parquet.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show tqdm progress bar while collecting corpus.",
    )
    parser.add_argument(
        "--ape_source",
        type=str,
        default="modernmolbert.local",
        help="APE only: version or commit descriptor recorded for provenance.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training arguments before any data is collected."""
    positive = {
        "tokenizer_train_size": args.tokenizer_train_size,
        "max_vocab_size": args.max_vocab_size,
        "min_freq_for_merge": args.min_freq_for_merge,
        "shuffle_buffer_size": args.shuffle_buffer_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")
    aligned_sample_parquet = getattr(args, "aligned_sample_parquet", None)
    sample_indices_path = getattr(args, "sample_indices_path", None)
    if (aligned_sample_parquet is None) != (sample_indices_path is None):
        raise ValueError("--aligned_sample_parquet and --sample_indices_path are required together")
    if aligned_sample_parquet is not None and (
        args.corpus_primitive_parquet is None
        or Path(args.corpus_primitive_parquet).resolve() != Path(aligned_sample_parquet).resolve()
    ):
        raise ValueError("Aligned sampling requires the same full-Parquet primitive scan")

    injects_symbols = (
        args.extra_vocab_symbols_path is not None or args.extra_vocab_selfies_path is not None
    )
    if args.algorithm == BPE:
        if injects_symbols:
            raise ValueError("--extra_vocab_* options inject APE symbols; BPE does not take them")
        if args.max_merge_pieces is not None:
            raise ValueError("--max_merge_pieces applies to APE only")

    if args.representation == SMILES_REPRESENTATION and args.extra_vocab_selfies_path is not None:
        raise ValueError(
            "--extra_vocab_selfies_path extracts SELFIES bracket symbols and is only "
            "valid with --representation SELFIES. Use --extra_vocab_symbols_path for SMILES."
        )

    if args.corpus_primitive_parquet is not None and injects_symbols:
        raise ValueError(
            "Corpus-only primitive coverage cannot be combined with extra symbol injection"
        )


def load_extra_vocab_symbols(
    *,
    symbols_path: Path | None,
    selfies_path: Path | None,
) -> list[str]:
    """Load additional vocabulary symbols to force into the tokenizer.

    symbols_path expects one primitive token per line, e.g.:
        [C@@H1]   (SELFIES bracket token)
        [Fe+3]    (SMILES bracket atom)

    selfies_path expects one full SELFIES string per line. All bracketed
    SELFIES primitive symbols are extracted. Only valid for SELFIES representation.
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


def _parquet_molecules(path: Path, column: str) -> Iterator[tuple[int, str]]:
    """Yield (row number from 1, molecule string); fail on an empty or missing value."""
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"Corpus primitive source not found: {path}")
    parquet = pq.ParquetFile(path)
    if column not in parquet.schema_arrow.names:
        raise ValueError(f"Missing {column!r} column in {path}")
    row = 0
    for batch in parquet.iter_batches(batch_size=32768, columns=[column]):
        for value in batch.column(0).to_pylist():
            row += 1
            if not isinstance(value, str) or not value:
                raise ValueError(f"Empty or non-string molecule in {path} at row {row}")
            yield row, value


def collect_corpus_primitives(
    path: Path, column: str, representation: str
) -> tuple[list[str], int]:
    """Collect every primitive symbol of a local training split, including SELFIES dots."""
    symbols: set[str] = set()
    n_rows = 0
    for n_rows, value in _parquet_molecules(path, column):
        try:
            pieces = pre_tokenize_molecule(value, representation)
        except ValueError as exc:
            raise ValueError(f"Malformed {representation} in {path} at row {n_rows}") from exc
        if "".join(pieces) != value:
            raise ValueError(f"Tokenizer loses {representation} content in {path} at row {n_rows}")
        symbols.update(pieces)
    if not n_rows:
        raise ValueError(f"No training molecules in {path}")
    return sorted(symbols), n_rows


def collect_corpus_characters(path: Path, column: str) -> tuple[list[str], int]:
    """Collect every character of a local training split."""
    characters: set[str] = set()
    n_rows = 0
    for _, value in _parquet_molecules(path, column):
        n_rows += 1
        characters.update(value)
    if not n_rows:
        raise ValueError(f"No training molecules in {path}")
    return sorted(characters), n_rows


def _scan_record(path: Path, n_rows: int, n_distinct: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "n_rows": n_rows,
        "n_distinct_primitives": n_distinct,
    }


def collect_aligned_sample(
    parquet_path: Path, indices_path: Path, column: str, sample_size: int, seed: int
) -> tuple[list[str], dict[str, Any]]:
    """Use one persisted permutation of physical source rows for all four tokenizers."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(parquet_path)
    if column not in source.schema_arrow.names:
        raise ValueError(f"Missing {column!r} column in {parquet_path}")
    n_rows = source.metadata.num_rows
    if not 0 < sample_size <= n_rows:
        raise ValueError(f"Tokenizer sample size {sample_size} exceeds {n_rows} source rows")
    if indices_path.exists():
        indices = np.load(indices_path, allow_pickle=False)
    else:
        indices = np.random.default_rng(seed).permutation(n_rows)[:sample_size].astype("<i8")
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(indices_path, indices, allow_pickle=False)
    if (
        indices.dtype != np.dtype("<i8")
        or indices.shape != (sample_size,)
        or len(np.unique(indices)) != sample_size
        or np.any(indices < 0)
        or np.any(indices >= n_rows)
    ):
        raise ValueError("Persisted tokenizer sample indices are invalid for this source")
    expected = np.random.default_rng(seed).permutation(n_rows)[:sample_size]
    if not np.array_equal(indices, expected):
        raise ValueError("Persisted tokenizer sample indices differ from the declared seed")

    values = (
        pq.read_table(parquet_path, columns=[column]).column(0).take(pa.array(indices)).to_pylist()
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Aligned sample has missing {column!r} values")
    return values, {
        "source_parquet": str(parquet_path),
        "source_sha256": file_sha256(parquet_path),
        "source_rows": n_rows,
        "sample_size": sample_size,
        "sample_seed": seed,
        "sample_order": "numpy.default_rng(seed).permutation(source_rows)[:sample_size]",
        "row_indices_path": str(indices_path),
        "row_indices_sha256": file_sha256(indices_path),
        "row_indices_order_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def build_ape(args: argparse.Namespace, corpus: list[str], column: str) -> dict[str, Any]:
    """Learn and save an APE vocabulary; return its metadata fields."""
    max_merge_pieces = (
        DEFAULT_MAX_MERGE_PIECES if args.max_merge_pieces is None else args.max_merge_pieces
    )
    if max_merge_pieces <= 0:
        max_merge_pieces = None

    frequency = train_ape(
        corpus,
        args.representation,
        max_vocab_size=args.max_vocab_size,
        min_freq_for_merge=args.min_freq_for_merge,
        max_merge_pieces=max_merge_pieces,
        log=log,
    )
    specials = sorted(EXPECTED_SPECIAL_IDS, key=lambda name: EXPECTED_SPECIAL_IDS[name])
    vocab = {SPECIAL_TOKENS[name]: EXPECTED_SPECIAL_IDS[name] for name in specials}
    vocab.update({token: index for index, token in enumerate(frequency, start=len(vocab))})
    tokenizer = APEPreTrainedTokenizer(vocab=vocab, representation=args.representation)
    tokenizer.vocabulary_frequency = frequency

    extra_symbols = load_extra_vocab_symbols(
        symbols_path=args.extra_vocab_symbols_path,
        selfies_path=args.extra_vocab_selfies_path,
    )
    if args.representation == SELFIES_REPRESENTATION:
        validate_selfies_symbols(extra_symbols)
    added_extra_symbols = tokenizer.add_tokens_to_vocabulary(extra_symbols)
    if extra_symbols:
        log(
            "Extra vocab coverage: "
            f"requested={len(extra_symbols)}, added={added_extra_symbols}, "
            f"already_present={len(extra_symbols) - added_extra_symbols}"
        )

    corpus_primitive_scan = None
    if args.corpus_primitive_parquet is not None:
        primitives, n_rows = collect_corpus_primitives(
            args.corpus_primitive_parquet, column, args.representation
        )
        corpus_primitive_scan = {
            **_scan_record(args.corpus_primitive_parquet, n_rows, len(primitives)),
            "n_added_after_merge_learning": tokenizer.add_tokens_to_vocabulary(primitives),
        }
        log(f"Corpus primitive coverage: {corpus_primitive_scan}")

    tokenizer.save_vocabulary_file(args.output_vocab_path)
    return {
        "ape_source": args.ape_source,
        "max_merge_pieces": max_merge_pieces,
        "extra_vocab_symbols_path": (
            str(args.extra_vocab_symbols_path) if args.extra_vocab_symbols_path else None
        ),
        "extra_vocab_selfies_path": (
            str(args.extra_vocab_selfies_path) if args.extra_vocab_selfies_path else None
        ),
        "extra_vocab_symbols_requested": len(extra_symbols),
        "extra_vocab_symbols_added": added_extra_symbols,
        "corpus_primitive_scan": corpus_primitive_scan,
    }


def build_bpe(args: argparse.Namespace, corpus: list[str], column: str) -> dict[str, Any]:
    """Learn and save a character-level BPE tokenizer; return its metadata fields."""
    import tokenizers

    from modernmolbert.tokenization.bpe import train_bpe

    alphabet: list[str] = []
    corpus_primitive_scan = None
    if args.corpus_primitive_parquet is not None:
        alphabet, n_rows = collect_corpus_characters(args.corpus_primitive_parquet, column)
        sample_characters: set[str] = set()
        for molecule in corpus:
            sample_characters.update(molecule)
        corpus_primitive_scan = {
            **_scan_record(args.corpus_primitive_parquet, n_rows, len(alphabet)),
            "n_added_to_initial_alphabet": len(set(alphabet) - sample_characters),
        }
        log(f"Corpus character coverage: {corpus_primitive_scan}")

    tokenizer = train_bpe(
        corpus,
        vocab_size=args.max_vocab_size,
        min_frequency=args.min_freq_for_merge,
        alphabet=alphabet,
    )
    tokenizer.save(str(args.output_vocab_path))
    return {
        "tokenizers_version": tokenizers.__version__,
        "max_merge_pieces": None,
        "extra_vocab_symbols_path": None,
        "extra_vocab_selfies_path": None,
        "extra_vocab_symbols_requested": 0,
        "extra_vocab_symbols_added": 0,
        "corpus_primitive_scan": corpus_primitive_scan,
    }


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    validate_args(args)

    column = infer_molecule_column(args.dataset_name, args.representation, args.molecule_column)
    args.output_vocab_path.parent.mkdir(parents=True, exist_ok=True)

    sample_provenance = None
    if args.aligned_sample_parquet is not None:
        corpus, sample_provenance = collect_aligned_sample(
            args.aligned_sample_parquet,
            args.sample_indices_path,
            column,
            args.tokenizer_train_size,
            args.seed,
        )
    else:
        corpus = collect_corpus_for_tokenizer(
            dataset_name=args.dataset_name,
            column=column,
            n=args.tokenizer_train_size,
            seed=args.seed,
            buffer_size=args.shuffle_buffer_size,
            data_dir=args.data_dir,
            data_files=args.data_files,
            show_progress=args.show_progress,
        )
    log(f"Corpus collected: {len(corpus)} sequences")
    validate_sample_shape(corpus[: min(512, len(corpus))], args.representation)

    build = build_bpe if args.algorithm == BPE else build_ape
    fields = build(args, corpus, column)

    # Describe the file as written: its hash, size and special-token IDs.
    tokenizer = load_tokenizer(
        args.output_vocab_path,
        {"algorithm": args.algorithm, "representation": args.representation},
    )
    special_ids = resolve_special_ids(tokenizer)
    assert_special_ids(special_ids)
    vocab_size = tokenizer_vocab_size(tokenizer)
    vocab_sha256 = file_sha256(args.output_vocab_path)
    metadata_path = metadata_path_for_vocab(args.output_vocab_path)
    metadata = {
        "algorithm": args.algorithm,
        "representation": args.representation,
        "dataset_name": args.dataset_name,
        "molecule_column": column,
        "tokenizer_train_size": args.tokenizer_train_size,
        "max_vocab_size": args.max_vocab_size,
        "min_freq_for_merge": args.min_freq_for_merge,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "seed": args.seed,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "tokenizer_path": str(args.output_vocab_path),
        "tokenizer_sha256": vocab_sha256,
        "aligned_sample": sample_provenance,
        **fields,
        "argv": list(argv if argv is not None else sys.argv),
        "git": get_git_revision(),
        "creation_command": "python -m modernmolbert.train_tokenizer",
    }
    write_tokenizer_metadata(metadata_path, metadata)

    log("Tokenizer training complete.")
    log(f"Algorithm: {args.algorithm}, representation: {args.representation}")
    log(f"Tokenizer file: {args.output_vocab_path}")
    log(f"Tokenizer metadata: {metadata_path}")
    log(f"Vocab size: {vocab_size}")
    log(f"Vocab SHA256: {vocab_sha256}")
    log(f"Dataset: {args.dataset_name} (column: {column})")
    log(f"Training size: {args.tokenizer_train_size}")
    log(f"Max vocab: {args.max_vocab_size}, Min freq: {args.min_freq_for_merge}")


if __name__ == "__main__":
    main()
