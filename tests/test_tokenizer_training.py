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
