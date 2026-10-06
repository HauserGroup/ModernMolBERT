import json

import pytest

from modernmolbert.tokenization.ape import train_ape
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


def _trained_tokenizer(
    corpus: list[str],
    *,
    representation: str = "SELFIES",
    max_vocab_size: int = 32,
    min_freq_for_merge: int = 2,
) -> APEPreTrainedTokenizer:
    frequency = train_ape(
        corpus,
        representation,
        max_vocab_size=max_vocab_size,
        min_freq_for_merge=min_freq_for_merge,
        max_merge_pieces=8,
        log=lambda _: None,
    )
    tokenizer = APEPreTrainedTokenizer(representation=representation)
    tokenizer.vocabulary = {
        **tokenizer.special_tokens,
        **{token: index for index, token in enumerate(frequency, start=5)},
    }
    tokenizer.vocabulary_frequency = frequency
    return tokenizer


def test_ape_train_terminates_on_tiny_corpus():
    corpus = ["[C][C][O]", "[C][O][C]", "[C][C][C]"] * 20
    tokenizer = _trained_tokenizer(corpus)

    assert len(tokenizer.vocabulary) > len(tokenizer.special_tokens)


def test_ape_train_skips_malformed_rows_without_crashing():
    # A stray non-bracket row must be skipped, not abort the whole run.
    corpus = ["[C][C][O]", "not selfies", "[C][O][C]", "[C][C][C]"] * 20
    tokenizer = _trained_tokenizer(corpus)

    assert len(tokenizer.vocabulary) > len(tokenizer.special_tokens)


def test_ape_tokenize_maps_malformed_selfies_to_unk():
    tokenizer = APEPreTrainedTokenizer()

    ids = tokenizer.encode("totally not selfies", add_special_tokens=False)

    assert ids == [tokenizer.unk_token_id]


def test_ape_train_freq_debits_merged_constituents():
    corpus = ["[C][C]"] * 10
    tokenizer = _trained_tokenizer(corpus)

    freq = tokenizer.vocabulary_frequency
    # The merge consumes every [C] (20 occurrences across 10 merges of [C]+[C]).
    assert freq["[C][C]"] == 10
    assert freq["[C]"] == 0
    # Debited primitive stays in the vocab so single-atom coverage survives.
    assert "[C]" in tokenizer.vocabulary
    assert all(count >= 0 for count in freq.values())


def test_load_vocabulary_rejects_duplicate_ids(tmp_path) -> None:
    vocab_path = tmp_path / "bad_vocab.json"
    vocab_path.write_text(
        '{"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 3}',
        encoding="utf-8",
    )
    tokenizer = APEPreTrainedTokenizer()

    with pytest.raises(ValueError, match="unique"):
        tokenizer.load_vocabulary_file(vocab_path)


def test_ape_train_does_not_merge_across_molecule_boundaries():
    # Each sample has exactly one token; any pair merge would require crossing
    # molecule boundaries and should therefore never happen.
    corpus = ["[C]", "[O]"] * 40
    tokenizer = _trained_tokenizer(corpus)

    assert "[C][O]" not in tokenizer.vocabulary
    assert "[O][C]" not in tokenizer.vocabulary


def test_ape_train_preserves_first_seen_pair_tie_order():
    corpus = ["[C][O][C][O]", "[O][C][N]", "[C][O][N]"] * 3
    tokenizer = _trained_tokenizer(corpus, max_vocab_size=20)

    learned_tokens = [
        token
        for token in tokenizer.vocabulary
        if not (token.startswith("<") and token.endswith(">"))
    ]
    assert learned_tokens == [
        "[C]",
        "[O]",
        "[N]",
        "[C][O]",
        "[C][O][C][O]",
        "[O][C]",
        "[O][C][N]",
        "[C][O][N]",
    ]


def test_smiles_pre_tokenize_preserves_chemical_tokens():
    tokenizer = APEPreTrainedTokenizer(representation="SMILES")

    tokens = tokenizer.pre_tokenize("ClC[C@H](Br)C1=CC=CC=C1")

    assert tokens[:6] == ["Cl", "C", "[C@H]", "(", "Br", ")"]
    assert tokenizer.pre_tokenize("C%12CCCCC%12")[1] == "%12"
    assert tokenizer.pre_tokenize(r"C/C=C\C")[5] == "\\"


def test_smiles_encoding_uses_ape_merges_over_smiles_tokens():
    tokenizer = APEPreTrainedTokenizer(representation="SMILES")
    tokenizer.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "C": 5,
        "O": 6,
        "CC": 7,
        "CCO": 8,
    }
    tokenizer.update_reverse_vocabulary()

    ids = tokenizer.encode("CCO", add_special_tokens=True)

    assert ids == [0, 8, 2]


def test_train_supports_smiles_representation():
    corpus = ["CCO", "CCN", "CCC"] * 20
    tokenizer = _trained_tokenizer(corpus, representation="SMILES")

    assert tokenizer.representation == "SMILES"
    assert "C" in tokenizer.vocabulary
    assert any(token.startswith("CC") for token in tokenizer.vocabulary)


def test_save_vocabulary_writes_expected_freq_file(tmp_path):
    tokenizer = APEPreTrainedTokenizer()
    vocab_path = tmp_path / "toy_tokenizer.json"

    tokenizer.save_vocabulary_file(vocab_path)

    assert vocab_path.exists()
    assert (tmp_path / "toy_tokenizer_freq.json").exists()


def test_from_pretrained_restores_reverse_vocabulary(tmp_path):
    tokenizer = APEPreTrainedTokenizer(representation="SMILES")
    tokenizer.save_pretrained(str(tmp_path))

    loaded = APEPreTrainedTokenizer.from_pretrained(str(tmp_path))

    bos_id = loaded.vocabulary[loaded.bos_token]
    assert loaded.convert_ids_to_tokens([bos_id]) == [loaded.bos_token]
    assert loaded.representation == "SMILES"


def _write_vocab(path, extra_tokens):
    path.write_text(
        json.dumps(
            {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<mask>": 4,
                **extra_tokens,
            }
        ),
        encoding="utf-8",
    )
