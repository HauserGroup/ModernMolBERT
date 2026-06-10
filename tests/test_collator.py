import pytest
import torch

from modernmolbert.collator import MolecularMLMCollator


def _examples():
    # 0=<s>, 1=<pad>, 2=</s>, 3=<unk>, 4=<mask>, 5/6/7 are normal tokens.
    return [
        {"input_ids": [0, 5, 6, 2]},
        {"input_ids": [0, 6, 7, 2]},
        {"input_ids": [0, 5, 2]},
    ]


def test_collator_shapes_and_padding_mask_behavior():
    torch.manual_seed(0)
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=32,
        mlm_probability=0.3,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    batch = collator(_examples())
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.shape == attention_mask.shape == labels.shape
    # Any padding positions must be ignored in labels.
    assert torch.all(labels[attention_mask == 0] == -100)


def test_collator_probability_zero_masks_nothing():
    torch.manual_seed(0)
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=32,
        mlm_probability=0.0,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    labels = collator(_examples())["labels"]
    assert torch.all(labels == -100)


def test_collator_probability_one_masks_all_eligible_only():
    torch.manual_seed(0)
    original = _examples()
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=32,
        mlm_probability=1.0,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    batch = collator(original)
    labels = batch["labels"]

    non_ignored = labels != -100

    # All non-special, non-padding positions should be selected for MLM labels.
    original_padded = torch.tensor(
        [
            [0, 5, 6, 2],
            [0, 6, 7, 2],
            [0, 5, 2, 1],
        ]
    )
    eligible = ~torch.isin(original_padded, torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(non_ignored, eligible)


def test_collator_random_replacements_never_use_special_ids():
    torch.manual_seed(2)
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=12,
        mlm_probability=1.0,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    batch = collator(_examples())
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    replaced_random = (labels != -100) & (input_ids != labels) & (input_ids != 4)
    if replaced_random.any():
        assert torch.all(~torch.isin(input_ids[replaced_random], torch.tensor([0, 1, 2, 3, 4])))


def test_collator_forces_at_least_one_mask_when_probability_nonzero():
    torch.manual_seed(7)
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=32,
        mlm_probability=1e-6,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    labels = collator(_examples())["labels"]
    assert int((labels != -100).sum()) >= 1


def test_collator_special_tokens_never_masked_and_vocab_bounds():
    torch.manual_seed(1)
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=11,
        mlm_probability=1.0,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    batch = collator(_examples())
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Special token labels must stay ignored.
    for sid in [0, 1, 2, 3, 4]:
        assert torch.all(labels[labels == sid] == -100)

    # All produced token ids remain within vocabulary bounds.
    assert int(torch.min(input_ids)) >= 0
    assert int(torch.max(input_ids)) < 11

    # With enough masked tokens and fixed seed, at least one <mask> should appear.
    assert int((input_ids == 4).sum()) > 0


def test_collator_random_replacement_candidates_exclude_special_ids():

    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=11,
        mlm_probability=0.15,
        special_token_ids=[0, 1, 2, 3, 4],
    )

    assert collator._eligible_replacement_ids.tolist() == [5, 6, 7, 8, 9, 10]


def test_collator_random_replacement_candidates_require_content_tokens():
    # A vocabulary made entirely of special tokens leaves nothing to sample for
    # random replacement; this is rejected when the collator is constructed.
    with pytest.raises(ValueError, match="No eligible non-special token IDs"):
        MolecularMLMCollator(
            pad_token_id=1,
            mask_token_id=4,
            vocab_size=5,
            mlm_probability=0.15,
            special_token_ids=[0, 1, 2, 3, 4],
        )


def test_standard_masking_replacement_runs_without_special_tokens():
    collator = MolecularMLMCollator(
        pad_token_id=1,
        mask_token_id=4,
        vocab_size=20,
        mlm_probability=0.30,
        special_token_ids=[0, 1, 2, 3, 4],
        masking_strategy="standard",
    )

    examples = [
        {"input_ids": [0, 5, 6, 7, 8, 2]},
        {"input_ids": [0, 9, 10, 11, 12, 2]},
    ]

    batch = collator(examples)

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["labels"].shape

    # At least one token should be selected because the collator enforces that
    # fallback when mlm_probability > 0.
    assert (batch["labels"] != -100).any()

    # Special tokens should not be prediction targets.
    labels = batch["labels"]
    for sid in [0, 1, 2, 3, 4]:
        assert not (labels == sid).any()


class TestHeteroatomRegex:
    """Verify _HETEROATOM_IN_BRACKET does not produce false positives or
    false negatives on SELFIES control symbols, standard atom tokens,
    and APE merged tokens."""

    pattern = MolecularMLMCollator._HETEROATOM_IN_BRACKET

    # --- true heteroatom tokens -------------------------------------------
    @pytest.mark.parametrize(
        "token",
        [
            "[N]",
            "[O]",
            "[S]",
            "[P]",
            "[F]",
            "[Cl]",
            "[Br]",
            "[I]",
            "[Se]",
            "[Si]",
            "[NH2]",  # N followed by uppercase H — should match
            "[NH1]",
            "[=O]",  # bond-prefix variant
            "[#N]",
            "[15N]",  # isotope prefix
            "[/123I]",  # stereo + isotope prefix
            "[SiH4]",  # two-char element with trailing chars
            "[C][N]",  # APE merged token — re.search finds [N]
            "[C][=O]",  # APE merged token — re.search finds [=O]
            "[Branch1][N]",  # APE merged token where heteroatom is in second primitive
        ],
    )
    def test_should_match(self, token):
        assert self.pattern.search(token), (
            f"Expected {token!r} to match as heteroatom-containing, but it did not."
        )

    # --- SELFIES control symbols and non-heteroatom atoms ------------------
    @pytest.mark.parametrize(
        "token",
        [
            "[Branch1]",  # contains "Br" substring — must NOT match
            "[Branch2]",
            "[Branch3]",
            "[=Branch1]",  # bond-prefixed branch
            "[Ring1]",  # contains no heteroatom symbol
            "[Ring2]",
            "[=Ring1]",
            "[Na]",  # N followed by lowercase a — must NOT match
            "[Fe]",  # F followed by lowercase e — must NOT match
            "[Sn]",  # S followed by lowercase n — must NOT match
            "[Sc]",  # S followed by lowercase c — must NOT match
            "[As]",  # intentionally not in covered set
            "[B]",  # boron — intentionally not covered
            "[C]",  # carbon — not a heteroatom
            "[=C]",
            "[#C]",
            "[C][C]",  # APE merged carbon-carbon token — no heteroatom
        ],
    )
    def test_should_not_match(self, token):
        assert not self.pattern.search(token), f"Expected {token!r} NOT to match, but it did."
