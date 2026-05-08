import torch


def mean_pool_excluding_token_ids(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    excluded_token_ids: set[int] | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
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
