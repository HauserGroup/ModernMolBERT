from __future__ import annotations

import torch


def cls_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts
