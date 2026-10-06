import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.pooling import mean_pool_excluding_token_ids
from modernmolbert.tokenization.load import load_checkpoint_tokenizer
from modernmolbert.utils import SELFIES_REPRESENTATION


@dataclass
class ModernMolBERTSelfiesFeaturizer:
    """Frozen ModernMolBERT embeddings for SMILES inputs.

    The checkpoint's tokenizer fixes the model input: SMILES are encoded as
    SELFIES for a SELFIES checkpoint and passed through unchanged for a SMILES
    checkpoint. Inputs that cannot be converted, or that the tokenizer does not
    reproduce exactly, are marked invalid rather than embedded. Inputs longer
    than the checkpoint's trained context are rejected and counted.
    """

    model_dir: str | Path
    tokenizer_path: str | Path | None = None
    name: str = "modernmolbert_selfies"
    max_seq_length: int | None = None
    pooling: Literal["mean", "cls"] = "mean"
    device: str = "auto"
    batch_size: int = 32

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)

        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_dir
        else:
            self.tokenizer_path = Path(self.tokenizer_path)

        if self.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling!r}")

        self._device = self._resolve_device(self.device)
        self.tokenizer, self.representation = load_checkpoint_tokenizer(self.tokenizer_path)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.model.to(self._device)
        self.model.eval()
        trained_context = int(self.model.config.max_position_embeddings)
        if self.max_seq_length is None:
            self.max_seq_length = trained_context
        elif not 0 < self.max_seq_length <= trained_context:
            raise ValueError(
                f"Embedding context {self.max_seq_length} exceeds trained context "
                f"{trained_context} or is not positive"
            )

    def featurize_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> FeatureBatch:
        import selfies as sf

        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be positive")

        model_inputs: list[str] = []
        valid_mask = np.zeros(len(smiles), dtype=bool)
        n_tokenization_failures = 0
        n_truncated = 0
        special_ids = self._special_token_ids()
        assert self.max_seq_length is not None

        for i, smi in enumerate(smiles):
            if smi is None:
                continue

            text = str(smi).strip()
            if not text:
                continue

            if self.representation == SELFIES_REPRESENTATION:
                try:
                    text = sf.encoder(text)
                except Exception:
                    continue
                if not text:
                    continue

            # Validate the complete string before truncation can hide a failure.
            # Older checkpoint-bundled tokenizers silently discard component dots.
            # An unknown-ID check alone cannot detect that molecular information loss.
            if "".join(self.tokenizer.tokenize(text)) != text:
                n_tokenization_failures += 1
                continue
            content_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not content_ids or any(token_id in special_ids for token_id in content_ids):
                n_tokenization_failures += 1
                continue
            if len(content_ids) + 2 > self.max_seq_length:
                n_truncated += 1
                continue
            model_inputs.append(text)
            valid_mask[i] = True

        hidden_size = int(getattr(self.model.config, "hidden_size", 0))

        if not model_inputs:
            out = FeatureBatch(
                X=np.zeros((0, hidden_size), dtype=np.float32),
                valid_mask=valid_mask,
                metadata=self._metadata(
                    n_inputs=len(smiles),
                    n_valid=0,
                    n_tokenization_failures=n_tokenization_failures,
                    n_truncated=n_truncated,
                ),
            )
            out.check(n_inputs=len(smiles))
            return out

        n_valid = len(model_inputs)
        X = np.empty((n_valid, hidden_size), dtype=np.float32)
        n_batches = math.ceil(n_valid / effective_batch_size)
        row = 0

        with torch.no_grad():
            for start in tqdm(
                range(0, n_valid, effective_batch_size),
                total=n_batches,
                desc="batches",
                unit="batch",
                leave=False,
            ):
                batch_strings = model_inputs[start : start + effective_batch_size]

                batch = self._tokenize_batch(batch_strings)
                batch = {key: value.to(self._device) for key, value in batch.items()}

                outputs = self.model(**batch)
                hidden = outputs.last_hidden_state

                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    pooled = mean_pool_excluding_token_ids(
                        last_hidden_state=hidden,
                        attention_mask=batch["attention_mask"],
                        input_ids=batch["input_ids"],
                        excluded_token_ids=self._special_token_ids(),
                    )

                n_rows = len(batch_strings)
                X[row : row + n_rows] = pooled.detach().cpu().float().numpy()
                row += n_rows

        out = FeatureBatch(
            X=X,
            valid_mask=valid_mask,
            metadata=self._metadata(
                n_inputs=len(smiles),
                n_valid=int(valid_mask.sum()),
                n_tokenization_failures=n_tokenization_failures,
                n_truncated=n_truncated,
            ),
        )
        out.check(n_inputs=len(smiles))
        return out

    def _metadata(
        self,
        *,
        n_inputs: int,
        n_valid: int,
        n_tokenization_failures: int = 0,
        n_truncated: int = 0,
    ) -> dict[str, object]:
        return {
            "featurizer": self.name,
            "backend": "modernmolbert",
            "model_dir": str(self.model_dir),
            "tokenizer_path": str(self.tokenizer_path),
            "representation": self.representation,
            "pooling": self.pooling,
            "pooling_special_tokens_excluded": self.pooling == "mean",
            "max_seq_length": self.max_seq_length,
            "device": str(self._device),
            "hidden_size": int(getattr(self.model.config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(self.model.config, "num_hidden_layers", 0)),
            "vocab_size": int(getattr(self.model.config, "vocab_size", 0)),
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "n_inputs": n_inputs,
            "n_valid": n_valid,
            "n_tokenization_failures": n_tokenization_failures,
            "n_truncated": n_truncated,
            "invalid_fraction": float(1.0 - n_valid / n_inputs) if n_inputs else 0.0,
        }

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
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

    def _tokenize_batch(
        self,
        model_inputs: list[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of model input strings (SELFIES or SMILES)."""

        if isinstance(model_inputs, str):
            raise TypeError("_tokenize_batch expects list[str], not str")

        if not model_inputs:
            raise ValueError("Cannot tokenize an empty batch")

        if type(self.tokenizer).__name__ == "SmirkTokenizerFast":
            # SMIRK 0.3.0's list path is incompatible with Transformers 5.
            # Its single-string path remains correct, so pad those encodings
            # here without changing the tokenizer used by existing models.
            sequences = [
                self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
                for text in model_inputs
            ]
            pad_id = self.tokenizer.pad_token_id
            if not isinstance(pad_id, int):
                raise ValueError("SMIRK tokenizer must have one integer pad token ID")
            longest = max(map(len, sequences))
            input_ids = torch.full((len(sequences), longest), pad_id, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(sequences):
                if self.tokenizer.padding_side == "left":
                    input_ids[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[row, -len(ids) :] = 1
                else:
                    input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[row, : len(ids)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Batch tokenization: one call for the whole list instead of a per-string
        # loop + manual padding. On MPS the model forward is fast enough that the
        # old Python loop became the bottleneck.
        encoded = self.tokenizer(
            model_inputs,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
