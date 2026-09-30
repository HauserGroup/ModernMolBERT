from audit_task_overlap import connectivity_keys, pair_overlap


def test_stereoisomers_share_a_connectivity_key():
    keys = connectivity_keys(["C[C@H](O)F", "C[C@@H](O)F", "CCO", "not_a_smiles"])
    assert len(keys) == 2


def test_pair_overlap_flags_family_members():
    test_keys = {"a": {"x", "y", "z"}, "b": {"y", "z"}, "c": {"q"}}
    overlap = pair_overlap(test_keys, {"a": "F", "b": "F"}).set_index(["dataset_a", "dataset_b"])
    assert overlap.loc[("a", "b"), "n_shared"] == 2
    assert overlap.loc[("a", "b"), "shared_fraction_of_smaller"] == 1.0
    assert bool(overlap.loc[("a", "b"), "same_family"])
    assert not bool(overlap.loc[("a", "c"), "same_family"])
    assert overlap.loc[("b", "c"), "family_b"] == "c"
