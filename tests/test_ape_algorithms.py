"""Check the APE trainer against a small, independent merge specification."""

import random
from argparse import Namespace
from collections import Counter
import json

import pytest

from modernmolbert.tokenization.ape import train_ape
from modernmolbert.tokenization_ape import pre_tokenize_molecule
from modernmolbert.train_tokenizer import build_ape


def _reference_train(
    corpus: list[str],
    representation: str,
    *,
    max_vocab_size: int,
    min_freq_for_merge: int,
    max_merge_pieces: int | None,
) -> dict[str, int]:
    """Count adjacent candidates, then replace non-overlapping sites left to right."""
    sequences: list[list[str]] = []
    vocabulary: dict[str, int] = {}
    for molecule in corpus:
        try:
            pieces = pre_tokenize_molecule(molecule, representation)
        except ValueError:
            continue
        for piece in pieces:
            vocabulary.setdefault(piece, 0)
        if pieces:
            sequences.append(pieces)
    if not vocabulary:
        raise ValueError("Cannot train APE tokenizer on an empty corpus.")

    while len(vocabulary) < max_vocab_size:
        counts: Counter[tuple[str, str]] = Counter()
        for pieces in sequences:
            for pair in zip(pieces, pieces[1:], strict=False):
                if max_merge_pieces is not None:
                    merged_pieces = pre_tokenize_molecule("".join(pair), representation)
                    if len(merged_pieces) > max_merge_pieces:
                        continue
                counts[pair] += 1
        if not counts:
            break
        pair = max(counts, key=counts.__getitem__)
        if counts[pair] < min_freq_for_merge:
            break

        merged = "".join(pair)
        vocabulary.setdefault(merged, 0)
        next_sequences: list[list[str]] = []
        for pieces in sequences:
            result = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == pair:
                    result.append(merged)
                    index += 2
                else:
                    result.append(pieces[index])
                    index += 1
            next_sequences.append(result)
        sequences = next_sequences

    surviving = Counter(piece for sequence in sequences for piece in sequence)
    return {piece: surviving[piece] for piece in vocabulary}


def test_overlapping_pair_frequency_counts_applied_merges():
    result = train_ape(
        ["[C][C][C][C]"],
        "SELFIES",
        max_vocab_size=10,
        min_freq_for_merge=1,
        max_merge_pieces=2,
        log=lambda _: None,
    )
    assert result == {"[C]": 0, "[C][C]": 2}


def test_minimum_frequency_counts_overlapping_candidates():
    result = train_ape(
        ["[C][C][C][C]"],
        "SELFIES",
        max_vocab_size=10,
        min_freq_for_merge=3,
        max_merge_pieces=2,
        log=lambda _: None,
    )
    assert result == {"[C]": 0, "[C][C]": 2}


def test_empty_or_malformed_corpus_has_no_vocabulary():
    with pytest.raises(ValueError, match="empty corpus"):
        train_ape(
            ["not SELFIES", ""],
            "SELFIES",
            max_vocab_size=10,
            min_freq_for_merge=1,
            max_merge_pieces=2,
            log=lambda _: None,
        )


def test_training_cli_writes_realized_frequencies(tmp_path):
    vocab_path = tmp_path / "ape.json"
    args = Namespace(
        representation="SELFIES",
        max_vocab_size=10,
        min_freq_for_merge=1,
        max_merge_pieces=2,
        output_vocab_path=vocab_path,
        extra_vocab_symbols_path=None,
        extra_vocab_selfies_path=None,
        corpus_primitive_parquet=None,
        ape_source="test",
    )
    build_ape(args, ["[C][C][C][C]"], "selfies")

    assert json.loads(vocab_path.read_text()) == {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[C][C]": 6,
    }
    assert json.loads((tmp_path / "ape_freq.json").read_text()) == {
        "[C]": 0,
        "[C][C]": 2,
    }


def test_ape_engine_matches_independent_reference():
    cases = [
        (["[C]", "[O]", "bad SELFIES"], "SELFIES"),
        (["[C][O]", "[O][C]", "[C][O]"], "SELFIES"),
        (["[C][C][C][C][C]", "[C][C]"], "SELFIES"),
        (["CCO", "OCC", "CCN"], "SMILES"),
    ]
    for representation, alphabet in (
        ("SELFIES", ["[C]", "[O]", "[N]", "[=O]", "."]),
        ("SMILES", ["C", "O", "N", "=", "(", ")"]),
    ):
        for seed in range(20):
            rng = random.Random(seed)
            corpus = [
                "".join(rng.choices(alphabet, k=rng.randrange(1, 8)))
                for _ in range(rng.randrange(5, 20))
            ]
            cases.append((corpus, representation))

    for corpus, representation in cases:
        for cap in (None, 2, 4):
            settings = {
                "max_vocab_size": 18,
                "min_freq_for_merge": 2,
                "max_merge_pieces": cap,
            }
            expected = _reference_train(corpus, representation, **settings)
            actual = train_ape(corpus, representation, log=lambda _: None, **settings)
            assert list(actual.items()) == list(expected.items()), (
                representation,
                cap,
                corpus,
            )


def test_oversized_pair_table_is_rejected_before_allocation():
    with pytest.raises(ValueError, match="oversized dense pair tables"):
        train_ape(
            ["[C][O]"],
            "SELFIES",
            max_vocab_size=10_000,
            min_freq_for_merge=1,
            max_merge_pieces=2,
            log=lambda _: None,
        )
