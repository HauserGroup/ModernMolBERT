from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.pooling import mean_pool_excluding_token_ids
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


@dataclass
class ModernMolBERTSelfiesFeaturizer:
    model_dir: str | Path
    tokenizer_path: str | Path | None = None
    name: str = "modernmolbert_selfies"
    max_seq_length: int = 256
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
        self.tokenizer = _load_ape_tokenizer(self.tokenizer_path)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.model.to(self._device)
        self.model.eval()

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

        selfies_strings: list[str] = []
        valid_mask = np.zeros(len(smiles), dtype=bool)

        for i, smi in enumerate(smiles):
            if smi is None:
                continue

            text = str(smi).strip()
            if not text:
                continue

            try:
                encoded = sf.encoder(text)
            except Exception:
                continue

            if not encoded:
                continue

            selfies_strings.append(encoded)
            valid_mask[i] = True

        hidden_size = int(getattr(self.model.config, "hidden_size", 0))

        if not selfies_strings:
            out = FeatureBatch(
                X=np.zeros((0, hidden_size), dtype=np.float32),
                valid_mask=valid_mask,
                metadata=self._metadata(
                    n_inputs=len(smiles),
                    n_valid=0,
                ),
            )
            out.check(n_inputs=len(smiles))
            return out

        features: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(selfies_strings), effective_batch_size):
                batch_strings = selfies_strings[start : start + effective_batch_size]

                batch = self._tokenize_selfies_batch(batch_strings)

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

                features.append(pooled.detach().cpu().float().numpy())

        X = np.concatenate(features, axis=0).astype(np.float32, copy=False)

        out = FeatureBatch(
            X=X,
            valid_mask=valid_mask,
            metadata=self._metadata(
                n_inputs=len(smiles),
                n_valid=int(valid_mask.sum()),
            ),
        )
        out.check(n_inputs=len(smiles))
        return out

    def featurize(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> FeatureBatch:
        return self.featurize_smiles(smiles, batch_size=batch_size)

    def _metadata(self, *, n_inputs: int, n_valid: int) -> dict[str, object]:
        return {
            "featurizer": self.name,
            "backend": "modernmolbert_selfies",
            "model_dir": str(self.model_dir),
            "tokenizer_path": str(self.tokenizer_path),
            "pooling": self.pooling,
            "max_seq_length": self.max_seq_length,
            "device": str(self._device),
            "hidden_size": int(getattr(self.model.config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(self.model.config, "num_hidden_layers", 0)),
            "vocab_size": int(getattr(self.model.config, "vocab_size", 0)),
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "n_inputs": n_inputs,
            "n_valid": n_valid,
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

    def _tokenize_selfies_batch(
        self,
        selfies_strings: list[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of SELFIES strings with the APE tokenizer."""

        if isinstance(selfies_strings, str):
            raise TypeError("_tokenize_selfies_batch expects list[str], not str")

        if not selfies_strings:
            raise ValueError("Cannot tokenize an empty SELFIES batch")

        # Batch tokenization: one call for the whole list instead of a per-string
        # loop + manual padding. On MPS the model forward is fast enough that the
        # old Python loop became the bottleneck.
        encoded = self.tokenizer(
            selfies_strings,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


def _load_ape_tokenizer(path: str | Path) -> APEPreTrainedTokenizer:
    """Load APE tokenizer from a file or checkpoint directory.

    Supported inputs:
    - directory containing ape_tokenizer/ AutoTokenizer artifacts
    - directory containing tokenizer.json
    - directory containing vocab.json
    - direct path to tokenizer.json
    - direct path to vocab.json
    """

    path = Path(path)

    if path.is_file():
        tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
        tokenizer.load_vocabulary_file(path)
        return tokenizer

    if path.is_dir():
        auto_dir = path / "ape_tokenizer"
        if auto_dir.exists():
            loaded = AutoTokenizer.from_pretrained(
                str(auto_dir),
                trust_remote_code=True,
            )
            if loaded.__class__.__name__ != "APEPreTrainedTokenizer":
                raise TypeError(f"Expected APEPreTrainedTokenizer, got {type(loaded)!r}")
            return loaded

        tokenizer_json = path / "tokenizer.json"
        vocab_json = path / "vocab.json"

        if tokenizer_json.exists():
            tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
            tokenizer.load_vocabulary_file(tokenizer_json)
            return tokenizer

        if vocab_json.exists():
            tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
            tokenizer.load_vocabulary_file(vocab_json)
            return tokenizer

        raise FileNotFoundError(
            f"No tokenizer vocabulary found in {path}. Expected tokenizer.json or vocab.json."
        )

    raise FileNotFoundError(f"Tokenizer path does not exist: {path}")
