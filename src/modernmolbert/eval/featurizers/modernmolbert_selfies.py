from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel

from modernmolbert.ape_tokenizer import APETokenizer
from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.pooling import mean_pool_excluding_token_ids


@dataclass
class ModernMolBERTSelfiesFeaturizer:
    model_dir: str | Path
    tokenizer_path: str | Path | None = None
    max_seq_length: int = 256
    pooling: Literal["mean", "cls"] = "mean"
    device: str = "auto"
    batch_size: int = 32

    @property
    def name(self) -> str:
        return "modernmolbert_selfies"

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)

        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_dir
        else:
            self.tokenizer_path = Path(self.tokenizer_path)

        self._device = self._resolve_device(self.device)
        self.tokenizer = APETokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.model.to(self._device)
        self.model.eval()

    def featurize_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> FeatureBatch:

        selfies_strings: list[str] = []

        valid_mask = np.zeros(len(smiles), dtype=bool)

        effective_batch_size = batch_size

        selfies_strings: list[str] = []

        valid_mask = np.zeros(len(smiles), dtype=bool)
        selfies_strings: list[str] = []
        valid_mask = np.zeros(len(smiles), dtype=bool)

        for i, smi in enumerate(smiles):
            try:
                import selfies as sf

                encoded = sf.encoder(smi)
                if encoded is None:
                    continue
                selfies_strings.append(encoded)
                valid_mask[i] = True
            except Exception:
                continue

        if not selfies_strings:
            return FeatureBatch(
                X=np.zeros((0, self.model.config.hidden_size), dtype=np.float32),
                valid_mask=valid_mask,
            )

        features = []

        with torch.no_grad():
            for start in range(0, len(selfies_strings), effective_batch_size):
                batch_strings = selfies_strings[start : start + effective_batch_size]

                batch = self.tokenizer(
                    batch_strings,
                    padding=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )

                batch = {
                    key: value.to(self._device)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }

                outputs = self.model(**batch)
                hidden = outputs.last_hidden_state

                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                elif self.pooling == "mean":
                    pooled = mean_pool_excluding_token_ids(
                        last_hidden_state=hidden,
                        attention_mask=batch["attention_mask"],
                        input_ids=batch["input_ids"],
                        excluded_token_ids=self._special_token_ids(),
                    )
                else:
                    raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

                features.append(pooled.detach().cpu().float().numpy())

        X = np.concatenate(features, axis=0).astype(np.float32, copy=False)

        out = FeatureBatch(X=X, valid_mask=valid_mask)
        out.check(n_inputs=len(smiles))
        return out

    def featurize(self, smiles: list[str]) -> FeatureBatch:
        return self.featurize_smiles(smiles)

    def _mean_pool_excluding_specials(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        special_ids = {
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
            getattr(self.tokenizer, "mask_token_id", None),
        }
        special_ids = {int(x) for x in special_ids if x is not None}

        content_mask = attention_mask.bool()

        for token_id in special_ids:
            content_mask &= input_ids != token_id

        # Fallback: if a row has no content tokens, use attention mask.
        empty_rows = content_mask.sum(dim=1) == 0
        if empty_rows.any():
            content_mask[empty_rows] = attention_mask[empty_rows].bool()

        weights = content_mask.unsqueeze(-1).to(hidden.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denom

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    def _special_token_ids(self) -> set[int]:

        ids = {
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
            getattr(self.tokenizer, "mask_token_id", None),
        }

        return {int(x) for x in ids if x is not None}
