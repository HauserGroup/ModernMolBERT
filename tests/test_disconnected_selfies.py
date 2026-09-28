from modernmolbert.tokenization_ape import ape_tokenize, pre_tokenize_molecule


def test_disconnected_selfies_preserves_component_separator():
    assert pre_tokenize_molecule("[C].[O]", "SELFIES") == ["[C]", ".", "[O]"]
    vocab = {"[C]": 5, ".": 6, "[O]": 7}
    assert ape_tokenize("[C].[O]", vocab, "SELFIES") == ["[C]", ".", "[O]"]


def test_missing_separator_is_an_explicit_unknown_not_silent_join():
    assert ape_tokenize("[C].[O]", {"[C]": 5, "[O]": 6}, "SELFIES") == ["[C]", "<unk>", "[O]"]
    assert ape_tokenize("[C]oops[O]", {"[C]": 5, "[O]": 6}, "SELFIES") == ["<unk>"]
