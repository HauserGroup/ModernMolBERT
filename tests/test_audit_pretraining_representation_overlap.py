from audit_pretraining_representation_overlap import annotate_rows, inspect_smiles, summaries


def test_tautomers_share_inchikey_without_sharing_selfies():
    train_smiles = "Oc1nc(O)nc(O)n1"
    test_smiles = "O=c1[nH]c(=O)[nH]c(=O)[nH]1"
    train_key, train_selfies = inspect_smiles(train_smiles)
    test_key, test_selfies = inspect_smiles(test_smiles)
    assert train_key == test_key
    assert train_selfies != test_selfies
    assert train_key and train_selfies and test_selfies

    rows: list[dict[str, object]] = [
        {
            "dataset": "toy",
            "standard_inchi_key": train_key,
            "smiles": train_smiles,
            "eval_selfies": train_selfies,
        },
        {
            "dataset": "toy",
            "standard_inchi_key": test_key,
            "smiles": test_smiles,
            "eval_selfies": test_selfies,
        },
    ]
    annotate_rows(
        rows,
        {train_key: {train_selfies}},
        {train_key: {train_smiles}},
        {train_smiles: {train_selfies}},
    )
    summary = summaries(rows, "a" * 64)[0]
    assert summary["n_inchikey_overlap"] == 2
    assert summary["n_canonical_smiles_overlap"] == 1
    assert summary["n_inchikey_only_overlap"] == 1
    assert summary["n_canonical_smiles_overlap_selfies_differ"] == 0
    assert summary["n_inchikey_overlap_selfies_differ"] == 1
