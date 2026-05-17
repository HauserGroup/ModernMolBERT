import pytest

from modernmolbert.rdkit_safety import looks_like_smiles


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "c1ccccc1",
        "C[C@H](N)C(=O)O",
        "[NH4+]",
        "[13CH4]",
        "ClCBr",
        "C%12CCCCC%12",
    ],
)
def test_looks_like_smiles_accepts_common_valid_syntax(smiles: str) -> None:
    assert looks_like_smiles(smiles) is True


@pytest.mark.parametrize(
    "smiles",
    [
        "",
        "   ",
        None,
        "not_a_smiles",
        "not-a-smiles",
        "C%1",
        "C[",
        "[AlH6]",
        "C]",
        "C[C",
    ],
)
def test_looks_like_smiles_rejects_obviously_invalid_text(smiles: object) -> None:
    assert looks_like_smiles(smiles) is False
