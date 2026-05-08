from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from modernmolbert.eval.featurizers.base import FeatureBatch


PoolingMode = Literal["mean", "cls"]


@dataclass(frozen=True)
class HuggingFaceSmilesFeaturizer:
    """Hugging Face encoder featurizer for SMILES language models.

    This is intended for frozen embedding benchmarks using models such as:

      - DeepChem/ChemBERTa-77M-MLM
      - DeepChem/ChemBERTa-10M-MLM
      - DeepChem/ChemBERTa-5M-MLM
      - other AutoTokenizer/AutoModel-compatible SMILES encoders

    It returns one fixed-size vector per valid SMILES string.
    """

    name: str
    model_name_or_path: str
    max_seq_length: int = 256
    pooling: PoolingMode = "mean"
    device: str = "auto"
    trust_remote_code: bool = False
    revision: str | None = None

    def featurize_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> FeatureBatch:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unknown pooling mode: {self.pooling!r}")

        device = _resolve_device(self.device)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )

        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )

        model.to(device)
        model.eval()

        valid_inputs: list[str] = []
        valid_indices: list[int] = []

        for i, smi in enumerate(smiles):
            if smi is None:
                continue

            text = str(smi).strip()
            if not text:
                continue

            valid_inputs.append(text)
            valid_indices.append(i)

        valid_mask = np.zeros(len(smiles), dtype=bool)
        if not valid_inputs:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            out = FeatureBatch(
                X=np.empty((0, hidden_size), dtype=np.float32),
                valid_mask=valid_mask,
                metadata=self._metadata(
                    model=model, tokenizer=tokenizer, device=str(device)
                ),
            )
            out.check(len(smiles))
            return out

        rows: list[np.ndarray] = []

        for start in range(0, len(valid_inputs), batch_size):
            batch_smiles = valid_inputs[start : start + batch_size]

            batch = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            hidden = outputs.last_hidden_state

            if self.pooling == "mean":
                pooled = _mean_pool(hidden, batch["attention_mask"])
            elif self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling mode: {self.pooling!r}")

            rows.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))

        X = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        valid_mask[np.asarray(valid_indices, dtype=int)] = True

        out = FeatureBatch(
            X=X,
            valid_mask=valid_mask,
            metadata=self._metadata(
                model=model, tokenizer=tokenizer, device=str(device)
            ),
        )
        out.check(len(smiles))
        return out

    def _metadata(self, *, model, tokenizer, device: str) -> dict[str, object]:
        n_params = int(sum(p.numel() for p in model.parameters()))

        return {
            "backend": "huggingface_transformers",
            "model_name_or_path": self.model_name_or_path,
            "pooling": self.pooling,
            "max_seq_length": self.max_seq_length,
            "device": device,
            "hidden_size": int(getattr(model.config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "num_parameters": n_params,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }


def _resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom
