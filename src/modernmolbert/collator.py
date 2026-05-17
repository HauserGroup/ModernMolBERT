import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class MolecularMLMCollator(DataCollatorMixin):
    """MLM data collator for SELFIES/APE-tokenized molecular sequences.

    Subclasses `DataCollatorMixin`, so instances are valid HuggingFace data
    collators and can be passed directly to `Trainer(data_collator=...)` or
    any PyTorch `DataLoader`.

    Masking strategies
    ------------------
    standard    : Independent Bernoulli per eligible token (original BERT).
    span        : Contiguous spans from Geometric(span_p_geom), clamped to
                  span_max_length.  Requires span_p_geom in (0,1) and
                  span_max_length >= 1.
    hetero_span : Same as span, but start positions are weighted toward tokens
                  whose string representation contains a heteroatom bracket
                  ([N], [O], [Cl], etc.).  Requires ids_to_tokens to be set
                  and heteroatom_start_weight > 0.

    All strategies apply the BERT corruption rule after selecting positions:
    80 % replaced with mask_token_id, 10 % replaced with a random non-special
    token, 10 % left unchanged.  Special tokens and padding are never masked.

    Example
    -------
    >>> collator = MolecularMLMCollator(
    ...     pad_token_id=1, mask_token_id=4,
    ...     vocab_size=32, mlm_probability=0.15,
    ...     special_token_ids=[0, 1, 2, 3, 4],
    ...     masking_strategy="span",
    ... )
    >>> batch = collator([{"input_ids": [0, 5, 6, 7, 8, 2]},
    ...                   {"input_ids": [0, 9, 10, 2]}])
    >>> batch.keys()
    dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    pad_token_id: int
    mask_token_id: int
    vocab_size: int
    mlm_probability: float
    special_token_ids: list[int]
    masking_strategy: str = "standard"
    span_p_geom: float = 0.4
    span_max_length: int = 6
    heteroatom_start_weight: float = 2.0
    ids_to_tokens: dict[int, str] = field(default_factory=dict)
    return_tensors: str = "pt"

    # ClassVar: excluded from __init__ by dataclass machinery.
    # Ordered longest-first so alternation matches Cl before C, Br before B, Se before S.
    _HETEROATOM_IN_BRACKET: ClassVar[re.Pattern] = re.compile(
        r"\["
        r"[=#/\\@+\-]*"
        r"(?:Cl|Br|Se|Si|[NOSPFI])"
        r"[^\]]*"
        r"\]"
    )

    def __post_init__(self) -> None:
        special_ids = {int(token_id) for token_id in self.special_token_ids}
        eligible = [token_id for token_id in range(self.vocab_size) if token_id not in special_ids]
        self._eligible_replacement_ids = torch.tensor(eligible, dtype=torch.long)

        if self.masking_strategy in {"span", "hetero_span"}:
            if not (0.0 < self.span_p_geom < 1.0):
                raise ValueError("span_p_geom must be in (0, 1)")
            if self.span_max_length < 1:
                raise ValueError("span_max_length must be >= 1")
            self._geom_dist: torch.distributions.Geometric | None = torch.distributions.Geometric(
                torch.tensor(self.span_p_geom)
            )
        else:
            self._geom_dist = None

        if self.masking_strategy == "hetero_span":
            if self.heteroatom_start_weight <= 0.0:
                raise ValueError("heteroatom_start_weight must be > 0")
            self._token_start_weights = self._build_token_start_weights()
        else:
            self._token_start_weights = None

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        ids = [
            torch.tensor(ex["input_ids"], dtype=torch.long)
            for ex in features
            if len(ex["input_ids"]) > 0
        ]

        if not ids:
            raise ValueError("Received an empty batch after tokenization.")

        input_ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()

        special_mask = torch.zeros_like(labels, dtype=torch.bool)
        for sid in self.special_token_ids:
            special_mask |= labels.eq(sid)

        if self.masking_strategy == "standard":
            masked_indices = self._sample_standard_mask(labels, attention_mask, special_mask)
        elif self.masking_strategy in {"span", "hetero_span"}:
            masked_indices = self._sample_batch_span_mask(input_ids, attention_mask, special_mask)
        else:
            raise ValueError(f"Unknown masking_strategy: {self.masking_strategy!r}")

        labels[~masked_indices] = -100

        # 80% of selected tokens become mask tokens.
        replace_draw = torch.rand(labels.shape)
        replace_with_mask = (replace_draw < 0.8) & masked_indices
        input_ids[replace_with_mask] = self.mask_token_id

        # 10% become random tokens.
        # Conditional probability is 0.5 among the remaining 20%, giving 10% overall.
        random_draw = torch.rand(labels.shape)
        replace_with_random = (random_draw < 0.5) & masked_indices & ~replace_with_mask
        if replace_with_random.any():
            eligible_random_ids = self.eligible_random_token_ids(device=input_ids.device)
            random_indices = torch.randint(
                low=0,
                high=len(eligible_random_ids),
                size=labels.shape,
                device=input_ids.device,
            )
            random_words = eligible_random_ids[random_indices]
            input_ids[replace_with_random] = random_words[replace_with_random]

        # Remaining 10% stay unchanged.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _sample_standard_mask(
        self,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        special_mask: torch.Tensor,
    ) -> torch.Tensor:
        eligible = (~special_mask) & attention_mask.bool()
        masked_indices = (torch.rand(labels.shape) < self.mlm_probability) & eligible

        if self.mlm_probability > 0.0 and not masked_indices.any():
            eligible_positions = eligible.nonzero(as_tuple=False)
            if len(eligible_positions) > 0:
                idx = int(torch.randint(len(eligible_positions), (1,)).item())
                row, col = eligible_positions[idx].tolist()
                masked_indices[int(row), int(col)] = True

        return masked_indices

    def _sample_batch_span_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(batch_size):
            row_mask = self._sample_span_mask(
                input_ids_row=input_ids[i],
                attention_mask_row=attention_mask[i],
                special_mask_row=special_mask[i],
            )
            if not row_mask.any():
                eligible = (~special_mask[i] & attention_mask[i].bool()).nonzero(as_tuple=False)
                if len(eligible) > 0:
                    rand_idx = int(torch.randint(len(eligible), (1,)).item())
                    col = int(eligible[rand_idx].item())
                    row_mask[col] = True
            masked_indices[i] = row_mask
        return masked_indices

    def eligible_random_token_ids(
        self,
        device: torch.device | None = None,
    ) -> torch.Tensor:

        if len(self._eligible_replacement_ids) == 0:
            raise ValueError(
                "No eligible non-special token IDs available for MLM random replacement."
            )

        if device is None:
            return self._eligible_replacement_ids

        return self._eligible_replacement_ids.to(device)

    def _build_token_start_weights(self) -> torch.Tensor:
        """Return a (vocab_size,) float weight tensor for heteroatom-biased span starts.

        Covered heteroatom set: N, O, S, P, F, Cl, Br, I, Se, Si.
        Elements not in this set (e.g. B, Sn, As, Ge) receive weight 1.0.
        If intentional coverage of additional elements is needed, extend
        _HETEROATOM_IN_BRACKET accordingly.

        Token IDs in special_token_ids receive weight 0.0 as a defensive guard;
        the eligible-position filter in _sample_span_mask is the primary barrier.
        Tokens matching the heteroatom pattern receive weight heteroatom_start_weight.
        All other tokens receive weight 1.0.
        """
        weights = torch.ones(self.vocab_size, dtype=torch.float32)
        special_ids = set(self.special_token_ids)
        for tok_id, tok_str in self.ids_to_tokens.items():
            if tok_id in special_ids:
                weights[tok_id] = 0.0
            elif self._HETEROATOM_IN_BRACKET.search(tok_str):
                weights[tok_id] = float(self.heteroatom_start_weight)
        return weights

    def _sample_span_mask(
        self,
        input_ids_row: torch.Tensor,
        attention_mask_row: torch.Tensor,
        special_mask_row: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a span-based boolean mask for one sequence.

        Contiguous spans of APE tokens are sampled until the number of newly
        masked positions reaches round(n_eligible × mlm_probability).
        Span lengths are drawn from a Geometric(span_p_geom) distribution and
        clamped to span_max_length. For hetero_span, span-start positions are
        sampled with weights proportional to heteroatom content.

        Adjacent independent spans may produce contiguous masked runs longer than
        span_max_length — the parameter bounds individual draws, not total runs.
        On very short sequences the actual masked fraction may exceed mlm_probability
        because a single span can cover the entire budget in one draw.
        """
        seq_len = input_ids_row.size(0)
        masked = torch.zeros(seq_len, dtype=torch.bool)

        eligible_mask = (~special_mask_row) & attention_mask_row.bool()
        eligible_pos = eligible_mask.nonzero(as_tuple=False).squeeze(1)

        if len(eligible_pos) == 0:
            return masked

        n_eligible = len(eligible_pos)
        budget = max(1, round(n_eligible * self.mlm_probability))

        # Pre-sample all geometric span lengths in one vectorised call.
        max_draws = budget * 5
        assert self._geom_dist is not None
        span_lengths = (
            self._geom_dist.sample((max_draws,)).long() + 1  # shift k≥0 → k≥1
        ).clamp(max=self.span_max_length)

        if self.masking_strategy == "hetero_span" and self._token_start_weights is not None:
            tok_ids_at_eligible = input_ids_row[eligible_pos]
            pos_weights = self._token_start_weights[tok_ids_at_eligible].clone()
        else:
            pos_weights = torch.ones(n_eligible, dtype=torch.float32)

        masked_count = 0

        for draw_idx in range(max_draws):
            if masked_count >= budget:
                break
            if pos_weights.sum().item() == 0.0:
                break

            start_local = int(torch.multinomial(pos_weights, num_samples=1).item())
            start = int(eligible_pos[start_local].item())
            span_len = int(span_lengths[draw_idx].item())
            end = min(start + span_len, seq_len)

            for pos in range(start, end):
                if not attention_mask_row[pos].item() or special_mask_row[pos].item():
                    end = pos
                    break

            if end <= start:
                pos_weights[start_local] = 0.0
                continue

            new_count = int((~masked[start:end]).sum().item())
            masked[start:end] = True
            masked_count += new_count

            # Zero weights for covered eligible positions so subsequent draws
            # explore unmasked territory.
            covered = (eligible_pos >= start) & (eligible_pos < end)
            pos_weights[covered] = 0.0

        return masked
