from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from modernmolbert.eval.featurizers.modernmolbert_selfies import (
    ModernMolBERTSelfiesFeaturizer,
)


def embed_smiles(
    model_path: str | Path,
    smiles: list[str],
    *,
    tokenizer_path: str | Path | None = None,
    batch_size: int = 128,
    max_length: int | None = None,
    pooling: Literal["mean", "cls"] = "mean",
    device: str = "auto",
    strict: bool = True,
) -> np.ndarray:
    """Embed SMILES with a trained ModernMolBERT checkpoint.

    Returns one row per input molecule. In strict mode, invalid molecules raise
    instead of returning partially filled embeddings.
    """

    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=model_path,
        tokenizer_path=tokenizer_path,
        max_seq_length=256 if max_length is None else max_length,
        pooling=pooling,
        device=device,
        batch_size=batch_size,
    )
    batch = featurizer.featurize_smiles(smiles, batch_size=batch_size)

    if strict and not bool(batch.valid_mask.all()):
        n_invalid = int((~batch.valid_mask).sum())
        raise ValueError(f"ModernMolBERT failed to embed {n_invalid} of {len(smiles)} molecules")

    if bool(batch.valid_mask.all()):
        return batch.X

    out = np.full((len(smiles), batch.X.shape[1]), np.nan, dtype=np.float32)
    out[batch.valid_mask] = batch.X
    return out
