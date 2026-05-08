import pytest

from modernmolbert.ape_tokenizer import APETokenizer


def test_ape_train_terminates_on_tiny_corpus():
    tokenizer = APETokenizer()
    corpus = ["[C][C][O]", "[C][O][C]", "[C][C][C]"] * 20

    tokenizer.train(
        corpus=corpus,
        max_vocab_size=32,
        min_freq_for_merge=2,
        save_checkpoint=False,
    )

    assert len(tokenizer.vocabulary) > len(tokenizer.special_tokens)


def test_ape_train_does_not_merge_across_molecule_boundaries():
    tokenizer = APETokenizer()
    # Each sample has exactly one token; any pair merge would require crossing
    # molecule boundaries and should therefore never happen.
    corpus = ["[C]", "[O]"] * 40

    tokenizer.train(
        corpus=corpus,
        max_vocab_size=32,
        min_freq_for_merge=2,
        save_checkpoint=False,
    )

    assert "[C][O]" not in tokenizer.vocabulary
    assert "[O][C]" not in tokenizer.vocabulary


def test_save_vocabulary_writes_expected_freq_file(tmp_path):
    tokenizer = APETokenizer()
    vocab_path = tmp_path / "toy_tokenizer.json"

    tokenizer.save_vocabulary(str(vocab_path))

    assert vocab_path.exists()
    assert (tmp_path / "toy_tokenizer_freq.json").exists()


def test_from_pretrained_restores_reverse_vocabulary(tmp_path):
    tokenizer = APETokenizer()
    tokenizer.save_pretrained(str(tmp_path))

    loaded = APETokenizer.from_pretrained(str(tmp_path))

    bos_id = loaded.vocabulary[loaded.bos_token]
    assert loaded.convert_ids_to_tokens([bos_id]) == [loaded.bos_token]


def test_get_special_tokens_mask_returns_input_length_masks():
    tokenizer = APETokenizer()
    token_ids = [tokenizer.bos_token_id, 10, tokenizer.eos_token_id]

    with_specials = tokenizer.get_special_tokens_mask(
        token_ids,
        already_has_special_tokens=True,
    )
    without_specials = tokenizer.get_special_tokens_mask(
        token_ids,
        already_has_special_tokens=False,
    )

    assert len(with_specials) == len(token_ids)
    assert len(without_specials) == len(token_ids)
    assert with_specials == [1, 0, 1]
    assert without_specials == [0, 0, 0]


def test_unk_token_id_matches_special_tokens_mapping():
    tokenizer = APETokenizer()
    assert tokenizer.unk_token_id == tokenizer.special_tokens[tokenizer.unk_token]


def test_pad_pads_labels_with_ignore_index():
    tokenizer = APETokenizer()
    batch = [
        {"input_ids": [0, 5, 2], "labels": [0, 5, 2]},
        {"input_ids": [0, 6, 7, 2], "labels": [0, 6, 7, 2]},
    ]

    out = tokenizer.pad(batch, return_tensors=None)

    assert out["labels"][0] == [0, 5, 2, -100]
    assert out["labels"][1] == [0, 6, 7, 2]


def test_train_from_iterator_raises_not_implemented():
    tokenizer = APETokenizer()
    with pytest.raises(NotImplementedError):
        tokenizer.train_from_iterator(iter(["[C][C][O]"]))
