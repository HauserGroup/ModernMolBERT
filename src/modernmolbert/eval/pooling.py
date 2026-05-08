import torch


def mean_pool_excluding_token_ids(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    excluded_token_ids: set[int] | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Mean-pool token embeddings while optionally excluding special tokens.

    Parameters
    ----------
    last_hidden_state:
        Tensor of shape [batch, seq_len, hidden_dim].

    attention_mask:
        Tensor of shape [batch, seq_len]. Nonzero entries are attended tokens.

    input_ids:
        Optional token-id tensor of shape [batch, seq_len]. Required if
        excluded_token_ids is provided.

    excluded_token_ids:
        Token IDs to remove from the mean, usually BOS/EOS/PAD/MASK/UNK.

    Notes
    -----
    If excluding special tokens leaves a row with no tokens, the function falls
    back to attention-mask pooling for that row.
    """

    content_mask = attention_mask.bool()

    if input_ids is not None and excluded_token_ids:
        for token_id in excluded_token_ids:
            content_mask = content_mask & input_ids.ne(int(token_id))

        empty_rows = content_mask.sum(dim=1).eq(0)
        if empty_rows.any():
            content_mask[empty_rows] = attention_mask.bool()[empty_rows]

    mask = content_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom
