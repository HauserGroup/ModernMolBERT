"""Functional-group masking utilities for MLM-FG pretraining.

MLM-FG (Fine-Grained MLM) masks entire chemical substructures — rings and
functional groups — rather than independent tokens. This forces the model to
predict structurally coherent units.

Public API used by FunctionalGroupMLMCollator in train_selfies_ape_modernbert:
  - get_molecule_substructures(smiles)
  - map_atoms_to_ape_positions(smiles, selfies_str, vocab)
"""

from __future__ import annotations

import re
from typing import Any

_STRUCTURAL_TOKEN_RE = re.compile(r"^\[(#?Branch\d*|#?Ring\d*|epsilon|nop)\]$")

# SMARTS patterns from the reference MLM-FG implementation.
FUNCTIONAL_GROUP_SMARTS: list[str] = [
    "[#6]-[#8]-[#6]",  # ether
    "[#6](=[#8])-[#8]-[#6]",  # ester
    "[#6](=[#8])-[#1]",  # aldehyde
    "[#6](=[#8])-[#6]",  # ketone
    "[#6](=[#8])-[#8][#1]",  # carboxylic acid
    "[#6](=[#8])-[#7]",  # amide
    "[#6]-[#8][#1]",  # hydroxyl
    "[#6]-[#7]",  # amine
    "[#6]-[F,Cl,Br,I]",  # halide
    "*C(C)=O",  # acetyl
    "*=O",  # carbonyl
]


def get_molecule_substructures(smiles: str) -> list[frozenset[int]]:
    """Return atom-index groups for rings, functional groups, and singletons.

    Each atom appears in exactly one group (first match wins). Unmatched atoms
    become singleton groups.

    Returns empty list on any RDKit failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MolFromSmarts

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        n_atoms = mol.GetNumAtoms()
        assigned: set[int] = set()
        groups: list[frozenset[int]] = []

        # Rings first.
        for ring in mol.GetRingInfo().AtomRings():
            ring_set = frozenset(ring) - assigned
            if ring_set:
                groups.append(ring_set)
                assigned |= ring_set

        # Functional groups.
        for smarts in FUNCTIONAL_GROUP_SMARTS:
            pattern = MolFromSmarts(smarts)
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                group = frozenset(match) - assigned
                if group:
                    groups.append(group)
                    assigned |= group

        # Remaining atoms become singletons.
        for atom_idx in range(n_atoms):
            if atom_idx not in assigned:
                groups.append(frozenset({atom_idx}))

        return groups
    except Exception:
        return []


def _primitive_spans_for_ape(
    selfies_str: str,
    vocab: dict[str, int],
) -> list[tuple[int, int]]:
    """Return (start, end) primitive-token spans, one per APE output token.

    Simulates ape_tokenize() greedy-match while tracking which primitive
    tokens were consumed by each APE token.
    """
    from modernmolbert.tokenization_ape import pre_tokenize_molecule

    primitives = pre_tokenize_molecule(selfies_str, "SELFIES")
    if not primitives:
        return []

    spans: list[tuple[int, int]] = []
    i = 0
    n = len(primitives)
    while i < n:
        matched = False
        for j in range(n, i, -1):
            candidate = "".join(primitives[i:j])
            if candidate in vocab:
                spans.append((i, j))
                i = j
                matched = True
                break
        if not matched:
            # unk — consume one primitive
            spans.append((i, i + 1))
            i += 1

    return spans


def map_atoms_to_ape_positions(
    selfies_str: str,
    vocab: dict[str, int],
) -> list[set[int]] | None:
    """Map chemical substructures to sets of APE token positions.

    Decodes selfies_str → SMILES via sf.decoder() and uses that SMILES for
    RDKit substructure detection. This guarantees that RDKit atom index k
    corresponds to SELFIES atom token k, because sf.decoder() preserves the
    SELFIES traversal order. Using a separately stored canonical SMILES is
    unsafe here: its atom numbering may differ from the SELFIES encoding order.

    Returns a list parallel to get_molecule_substructures(): each element is
    a set of 0-based APE sequence positions (excluding BOS/EOS offsets —
    the caller is responsible for applying those).

    Returns None if mapping fails for any reason, signalling the caller to
    fall back to standard token-level masking.
    """
    try:
        import selfies as sf
        from modernmolbert.tokenization_ape import pre_tokenize_molecule

        # Decode to get SMILES whose atom order matches the SELFIES token order.
        smiles = sf.decoder(selfies_str)
        if not smiles:
            return None

        substructures = get_molecule_substructures(smiles)
        if not substructures:
            return None

        primitives = pre_tokenize_molecule(selfies_str, "SELFIES")
        if not primitives:
            return None

        spans = _primitive_spans_for_ape(selfies_str, vocab)
        if not spans:
            return None

        # Identify which primitive positions are atom-producing tokens.
        atom_prim_positions: list[int] = [
            i for i, tok in enumerate(primitives) if not _STRUCTURAL_TOKEN_RE.match(tok)
        ]

        # Verify atom count matches between decoded SMILES and SELFIES.
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n_atoms = mol.GetNumAtoms()
        if len(atom_prim_positions) != n_atoms:
            return None

        # Build primitive index → APE token index mapping.
        primitive_to_ape: dict[int, int] = {}
        for ape_idx, (start, end) in enumerate(spans):
            for prim_idx in range(start, end):
                primitive_to_ape[prim_idx] = ape_idx

        # Map each substructure to APE positions.
        result: list[set[int]] = []
        for substructure in substructures:
            ape_positions: set[int] = set()
            for atom_idx in substructure:
                prim_pos = atom_prim_positions[atom_idx]
                ape_pos = primitive_to_ape.get(prim_pos)
                if ape_pos is not None:
                    ape_positions.add(ape_pos)
            result.append(ape_positions)

        return result

    except Exception:
        return None


def build_fg_masked_indices(
    substructure_ape_positions: list[set[int]],
    seq_len: int,
    mlm_probability: float,
    special_token_positions: set[int],
    *,
    rng: Any = None,
) -> list[bool]:
    """Sample substructures and return a per-position mask list.

    Each substructure is selected as a unit with probability mlm_probability.
    Special token positions are never masked.

    Args:
        substructure_ape_positions: output of map_atoms_to_ape_positions,
            shifted by +1 for the BOS token prepended by the tokenizer.
        seq_len: total sequence length including BOS/EOS/padding.
        mlm_probability: per-substructure sampling probability.
        special_token_positions: positions to always leave unmasked.
        rng: optional torch.Generator for reproducibility.

    Returns:
        Boolean list of length seq_len.
    """
    import torch

    masked = [False] * seq_len
    for group in substructure_ape_positions:
        if torch.bernoulli(torch.tensor(mlm_probability), generator=rng).item():
            for pos in group:
                if pos < seq_len and pos not in special_token_positions:
                    masked[pos] = True
    return masked
