"""Tests for span and hetero_span masking strategies in MolecularMLMCollator."""

import pytest
import torch

from modernmolbert.collator import MolecularMLMCollator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SPECIAL_IDS = [0, 1, 2, 3, 4]  # bos, pad, eos, unk, mask
VOCAB_SIZE = 100


def _make_collator(
    strategy: str = "standard",
    mlm_probability: float = 0.15,
    span_max_length: int = 6,
    span_p_geom: float = 0.4,
    heteroatom_start_weight: float = 2.0,
    ids_to_tokens: dict | None = None,
    vocab_size: int = VOCAB_SIZE,
) -> MolecularMLMCollator:
    if ids_to_tokens is None:
        ids_to_tokens = {i: f"[T{i}]" for i in range(10, vocab_size)}
        ids_to_tokens.update({0: "<s>", 1: "<pad>", 2: "</s>", 3: "<unk>", 4: "<mask>"})
    return MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=vocab_size,
        mlm_probability=mlm_probability,
        special_token_ids=SPECIAL_IDS,
        masking_strategy=strategy,
        span_p_geom=span_p_geom,
        span_max_length=span_max_length,
        heteroatom_start_weight=heteroatom_start_weight,
        ids_to_tokens=ids_to_tokens,
    )


def _examples():
    # bos=0, eos=2, real tokens 5-9
    return [
        {"input_ids": [0, 5, 6, 7, 8, 9, 2]},
        {"input_ids": [0, 6, 7, 8, 2]},
        {"input_ids": [0, 5, 6, 7, 8, 9, 5, 6, 7, 8, 2]},
    ]


# ---------------------------------------------------------------------------
# Output shape / dtype (same contract as standard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "kwargs"),
    [
        ("span", {"span_p_geom": 0.0}),
        ("span", {"span_p_geom": 1.0}),
        ("span", {"span_max_length": 0}),
        ("hetero_span", {"heteroatom_start_weight": 0.0}),
    ],
)
def test_invalid_span_params_raise(strategy, kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _make_collator(strategy, **kwargs)


# ---------------------------------------------------------------------------
# Invariants that must hold for all strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["standard", "span", "hetero_span"])
def test_special_tokens_never_masked(strategy):
    torch.manual_seed(0)
    coll = _make_collator(strategy, mlm_probability=0.3)
    batch = coll(_examples())
    from torch.nn.utils.rnn import pad_sequence

    examples = _examples()
    ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
    original = pad_sequence(ids, batch_first=True, padding_value=1)
    special_mask = torch.zeros_like(original, dtype=torch.bool)
    for sid in SPECIAL_IDS:
        special_mask |= original.eq(sid)
    assert torch.all(batch["labels"][special_mask] == -100), (
        f"{strategy}: special token was included in masked positions"
    )


# ---------------------------------------------------------------------------
# span-specific: contiguous spans
# ---------------------------------------------------------------------------


def test_span_produces_contiguous_runs():
    """At least some masked runs in span mode should be length > 1."""
    torch.manual_seed(0)
    coll = _make_collator("span", mlm_probability=0.5, span_max_length=4)
    # Use a long sequence to give spans room to form.
    examples = [{"input_ids": [0] + list(range(5, 25)) + [2]}]
    found_multi = False
    for _ in range(20):
        batch = coll(examples)
        masked = batch["labels"][0] != -100
        run_len = 0
        for m in masked.tolist():
            if m:
                run_len += 1
                if run_len >= 2:
                    found_multi = True
                    break
            else:
                run_len = 0
        if found_multi:
            break
    assert found_multi, "span strategy never produced a run of length >= 2"


def test_span_max_length_clamps_individual_draw():
    """Single-draw budget: max contiguous run ≤ span_max_length."""

    def max_run(labels_row: torch.Tensor) -> int:
        masked = (labels_row != -100).tolist()
        best = cur = 0
        for m in masked:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best

    # 10 eligible positions, mlm_probability=0.05 → budget = max(1, round(10*0.05)) = 1
    for max_len in [1, 3]:
        for seed in range(4):
            torch.manual_seed(seed)
            coll = _make_collator(
                "span",
                mlm_probability=0.05,
                span_max_length=max_len,
                span_p_geom=0.01,  # very low p → geometric returns large pre-clamp values
            )
            examples = [{"input_ids": [0] + list(range(5, 15)) + [2]}]
            batch = coll(examples)
            run = max_run(batch["labels"][0])
            assert run <= max_len, (
                f"seed={seed}, span_max_length={max_len}: max run {run} exceeds span_max_length"
            )


# ---------------------------------------------------------------------------
# hetero_span: weight tensor
# ---------------------------------------------------------------------------


def test_build_token_start_weights_respects_selfies_atom_boundaries_and_isotopes():
    ids_to_tokens = {
        0: "<s>",
        1: "<pad>",
        2: "</s>",
        3: "<unk>",
        4: "<mask>",
        5: "[C]",
        6: "[15N]",
        7: "[=18O]",
        8: "[/123I]",
        9: "[Na]",
        10: "[Fe]",
        11: "[Sn]",
        12: "[Branch1]",
        13: "[C][=O]",
        14: "[ClH0]",
    }
    coll = _make_collator(
        "hetero_span",
        ids_to_tokens=ids_to_tokens,
        vocab_size=15,
        heteroatom_start_weight=3.0,
    )
    weights = coll._token_start_weights
    assert weights is not None

    for tok_id in (6, 7, 8, 13, 14):
        assert weights[tok_id].item() == 3.0, f"{ids_to_tokens[tok_id]} should be elevated"
    for tok_id in (5, 9, 10, 11, 12):
        assert weights[tok_id].item() == 1.0, f"{ids_to_tokens[tok_id]} should not be elevated"


def test_hetero_span_sampling_uses_hetero_start_weights(monkeypatch):
    ids_to_tokens = {
        0: "<s>",
        1: "<pad>",
        2: "</s>",
        3: "<unk>",
        4: "<mask>",
        5: "[C]",
        6: "[N]",
        7: "[C]",
    }
    coll = _make_collator(
        "hetero_span",
        ids_to_tokens=ids_to_tokens,
        vocab_size=8,
        mlm_probability=0.1,
        span_max_length=1,
        heteroatom_start_weight=7.0,
    )

    class OneTokenSpans:
        def sample(self, shape):
            return torch.zeros(shape)

    captured_weights = []

    def choose_hetero_position(weights, num_samples, *args, **kwargs):
        captured_weights.append(weights.detach().clone())
        return torch.tensor([1])

    coll._geom_dist = OneTokenSpans()  # type: ignore[assignment]
    monkeypatch.setattr(torch, "multinomial", choose_hetero_position)

    input_ids_row = torch.tensor([0, 5, 6, 7, 2])
    attention_mask_row = torch.ones_like(input_ids_row)
    special_mask_row = torch.zeros_like(input_ids_row, dtype=torch.bool)
    for sid in SPECIAL_IDS:
        special_mask_row |= input_ids_row.eq(sid)

    masked = coll._sample_span_mask(input_ids_row, attention_mask_row, special_mask_row)

    assert captured_weights, "hetero_span did not call weighted span-start sampling"
    assert captured_weights[0].tolist() == [1.0, 7.0, 1.0]
    assert masked.tolist() == [False, False, True, False, False]
