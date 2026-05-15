"""Tests for MLM-FG functional-group masking utilities."""

import pytest
import torch

from modernmolbert.masking import (
    FUNCTIONAL_GROUP_SMARTS,
    build_fg_masked_indices,
    get_molecule_substructures,
    map_atoms_to_ape_positions,
)


# ---------------------------------------------------------------------------
# get_molecule_substructures
# ---------------------------------------------------------------------------


def test_benzene_ring():
    groups = get_molecule_substructures("c1ccccc1")
    # All 6 ring atoms in one group.
    assert len(groups) == 1
    assert groups[0] == frozenset(range(6))


def test_ethanol_no_ring():
    groups = get_molecule_substructures("CCO")
    # No ring; should have functional group or singletons covering all 3 atoms.
    all_atoms = set()
    for g in groups:
        all_atoms |= g
    assert all_atoms == {0, 1, 2}


def test_all_atoms_covered():
    smiles = "CC(=O)Nc1ccc(cc1)O"  # paracetamol
    groups = get_molecule_substructures(smiles)
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    covered = set()
    for g in groups:
        covered |= g
    assert covered == set(range(n_atoms))


def test_no_overlap_between_groups():
    groups = get_molecule_substructures("CC(=O)Nc1ccc(cc1)O")
    seen: set[int] = set()
    for g in groups:
        assert g.isdisjoint(seen), "Atom assigned to multiple groups"
        seen |= g


def test_invalid_smiles_returns_empty():
    assert get_molecule_substructures("not_valid!!!") == []


def test_functional_group_smarts_all_valid():
    from rdkit.Chem import MolFromSmarts

    for smarts in FUNCTIONAL_GROUP_SMARTS:
        assert MolFromSmarts(smarts) is not None, f"Invalid SMARTS: {smarts}"


# ---------------------------------------------------------------------------
# map_atoms_to_ape_positions
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_vocab():
    """Tiny APE vocab: specials + a few SELFIES primitives."""
    return {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[N]": 7,
        "[=O]": 8,
        "[C][C]": 9,  # merged token
    }


def test_ethanol_ape_mapping(minimal_vocab):
    import selfies as sf

    selfies_str = sf.encoder("CCO")
    result = map_atoms_to_ape_positions(selfies_str, minimal_vocab)
    assert result is not None
    # All substructures together must cover APE positions for all 3 atoms.
    all_positions: set[int] = set()
    for group in result:
        all_positions |= group
    assert len(all_positions) > 0


def test_invalid_selfies_returns_none(minimal_vocab):
    # "[C][C][O]" as SELFIES decodes fine; test a string that fails RDKit
    result = map_atoms_to_ape_positions("not_valid_selfies", minimal_vocab)
    assert result is None


def test_empty_selfies_returns_none(minimal_vocab):
    result = map_atoms_to_ape_positions("", minimal_vocab)
    assert result is None


# ---------------------------------------------------------------------------
# build_fg_masked_indices
# ---------------------------------------------------------------------------


def test_build_fg_masked_indices_shape():
    groups = [{1, 2}, {3}]
    result = build_fg_masked_indices(
        groups, seq_len=6, mlm_probability=1.0, special_token_positions={0, 5}
    )
    assert len(result) == 6


def test_build_fg_masked_indices_prob_one_masks_all_eligible():
    torch.manual_seed(0)
    groups = [{1, 2}, {3}]
    result = build_fg_masked_indices(
        groups, seq_len=6, mlm_probability=1.0, special_token_positions={0, 5}
    )
    # Positions 1, 2, 3 in groups and eligible — must be masked.
    assert result[1] and result[2] and result[3]
    # Position 4 not in any group — not masked.
    assert not result[4]


def test_build_fg_masked_indices_prob_zero_masks_nothing():
    torch.manual_seed(0)
    groups = [{1, 2}, {3}]
    result = build_fg_masked_indices(
        groups, seq_len=6, mlm_probability=0.0, special_token_positions={0, 5}
    )
    assert not any(result)


def test_build_fg_masked_indices_respects_special_positions():
    groups = [{0, 1}]  # group spans a special position
    result = build_fg_masked_indices(
        groups, seq_len=4, mlm_probability=1.0, special_token_positions={0}
    )
    assert not result[0]  # special — never masked
    assert result[1]  # eligible — masked


# ---------------------------------------------------------------------------
# FunctionalGroupMLMCollator integration
# ---------------------------------------------------------------------------


def test_fg_collator_output_shape_matches_standard():
    import selfies as sf
    from modernmolbert.train_selfies_ape_modernbert import (
        FunctionalGroupMLMCollator,
        MolecularMLMCollator,
    )

    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[N]": 7,
        "[=O]": 8,
        "[C][C]": 9,
    }
    special_ids = [0, 1, 2, 3, 4]

    def make_example(smiles):
        sel = sf.encoder(smiles)
        # Manually encode: BOS + primitive tokens + EOS (no APE merging for simplicity)
        from modernmolbert.tokenization_ape import pre_tokenize_molecule

        primitives = pre_tokenize_molecule(sel, "SELFIES")
        ids = [0] + [vocab.get(t, 3) for t in primitives] + [2]
        return {"input_ids": ids, "selfies": sel}

    examples = [make_example("CCO"), make_example("CC(=O)O"), make_example("c1ccccc1")]

    standard = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=10,
        mlm_probability=0.3,
        special_token_ids=special_ids,
    )
    fg = FunctionalGroupMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=10,
        mlm_probability=0.3,
        special_token_ids=special_ids,
        tokenizer_vocab=vocab,
    )

    torch.manual_seed(0)
    std_batch = standard(examples)
    torch.manual_seed(0)
    fg_batch = fg(examples)

    for key in ("input_ids", "attention_mask", "labels"):
        assert fg_batch[key].shape == std_batch[key].shape, f"Shape mismatch on {key}"
        assert fg_batch[key].dtype == std_batch[key].dtype, f"Dtype mismatch on {key}"

    # Padding positions must be -100 in labels.
    assert torch.all(fg_batch["labels"][fg_batch["attention_mask"] == 0] == -100)


def test_fg_collator_fallback_on_missing_selfies():
    from modernmolbert.train_selfies_ape_modernbert import FunctionalGroupMLMCollator

    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "[C]": 5}
    # No 'selfies' key — collator must fall back to standard masking without error.
    examples = [{"input_ids": [0, 5, 5, 2]}]

    fg = FunctionalGroupMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=6,
        mlm_probability=0.5,
        special_token_ids=[0, 1, 2, 3, 4],
        tokenizer_vocab=vocab,
    )
    torch.manual_seed(0)
    batch = fg(examples)
    assert batch["input_ids"].shape == (1, 4)
