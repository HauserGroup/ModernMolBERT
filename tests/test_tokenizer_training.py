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
