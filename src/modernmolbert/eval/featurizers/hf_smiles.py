from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Literal, Sequence

import numpy as np

from modernmolbert.eval.featurizers.base import FeatureBatch
from modernmolbert.eval.pooling import mean_pool_excluding_token_ids


PoolingMode = Literal["mean", "cls"]
_ATTENTION_ROPE_KEYS = frozenset({"sliding_attention", "full_attention"})


@dataclass(frozen=True)
class HuggingFaceSmilesFeaturizer:
    """Hugging Face encoder featurizer for SMILES language models."""

    name: str
    model_name_or_path: str
    max_seq_length: int = 256
    pooling: PoolingMode = "mean"
    device: str = "auto"
    trust_remote_code: bool = False
    revision: str | None = None

    # Compatibility shim for ModernBERT checkpoints such as MolEncoder whose
    # config contains legacy/extra RoPE keys rejected by strict HF validation.
    sanitize_modernbert_rope: bool = True

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

        model_path = _maybe_prepare_sanitized_model_dir(
            model_name_or_path=self.model_name_or_path,
            revision=self.revision,
            sanitize_modernbert_rope=self.sanitize_modernbert_rope,
        )

        if self.sanitize_modernbert_rope:
            _patch_modernbert_rope_standardization()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
        )

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
        )

        model.to(device)
        model.eval()

        special_token_ids = [
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", [])
            if token_id is not None
        ]

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
                    model=model,
                    tokenizer=tokenizer,
                    device=str(device),
                    resolved_model_path=model_path,
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
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            hidden = _get_last_hidden_state(outputs)

            if self.pooling == "mean":
                pooled = _mean_pool(
                    hidden,
                    batch["attention_mask"],
                    input_ids=batch.get("input_ids"),
                    special_token_ids=special_token_ids,
                )
            else:
                pooled = hidden[:, 0, :]

            rows.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))

        X = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        valid_mask[np.asarray(valid_indices, dtype=int)] = True

        out = FeatureBatch(
            X=X,
            valid_mask=valid_mask,
            metadata=self._metadata(
                model=model,
                tokenizer=tokenizer,
                device=str(device),
                resolved_model_path=model_path,
            ),
        )
        out.check(len(smiles))
        return out

    def _metadata(
        self,
        *,
        model,
        tokenizer,
        device: str,
        resolved_model_path: str,
    ) -> dict[str, object]:
        n_params = int(sum(p.numel() for p in model.parameters()))

        return {
            "backend": "huggingface_transformers",
            "model_name_or_path": self.model_name_or_path,
            "resolved_model_path": resolved_model_path,
            "revision": self.revision,
            "pooling": self.pooling,
            "max_seq_length": self.max_seq_length,
            "device": device,
            "hidden_size": int(getattr(model.config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "num_parameters": n_params,
            "trust_remote_code": self.trust_remote_code,
            "sanitize_modernbert_rope": self.sanitize_modernbert_rope,
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


def _mean_pool(
    last_hidden_state,
    attention_mask,
    input_ids=None,
    special_token_ids: list[int] | None = None,
):

    return mean_pool_excluding_token_ids(
        last_hidden_state=last_hidden_state,
        attention_mask=attention_mask,
        input_ids=input_ids,
        excluded_token_ids=special_token_ids,
    )


def _get_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state

    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        if "hidden_states" in outputs:
            return outputs["hidden_states"][-1]

    if isinstance(outputs, tuple) and len(outputs) > 0:
        return outputs[0]

    raise TypeError(
        "Could not find last hidden state in Hugging Face model output. "
        f"Output type: {type(outputs)!r}"
    )


def _maybe_prepare_sanitized_model_dir(
    *,
    model_name_or_path: str,
    revision: str | None,
    sanitize_modernbert_rope: bool,
) -> str:
    """Return a local model path, patching config.json only when needed.

    Sanitized configs are cached in a deterministic local directory so we avoid
    creating untracked temporary copies on repeated runs.
    """

    source_dir = _resolve_model_dir(
        model_name_or_path=model_name_or_path,
        revision=revision,
    )

    if not sanitize_modernbert_rope:
        return str(source_dir)

    config_path = source_dir / "config.json"
    if not config_path.exists():
        return str(source_dir)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    sanitized = _sanitize_modernbert_rope_config(config)

    if sanitized == config:
        return str(source_dir)

    cache_key = hashlib.sha256(
        (
            str(source_dir.resolve()) + "\n" + json.dumps(sanitized, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()[:16]

    cache_root = Path.home() / ".cache" / "modernmolbert" / "hf_sanitized_models"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_dir = cache_root / f"{source_dir.name}_{cache_key}"

    if cached_dir.exists() and (cached_dir / "config.json").exists():
        return str(cached_dir)

    tmp_dir = cache_root / f".{cached_dir.name}.tmp-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    _copy_model_dir(source_dir, tmp_dir)
    (tmp_dir / "config.json").write_text(
        json.dumps(sanitized, indent=2) + "\n",
        encoding="utf-8",
    )

    tmp_dir.rename(cached_dir)
    return str(cached_dir)


def _resolve_model_dir(
    *,
    model_name_or_path: str,
    revision: str | None,
) -> Path:
    local_path = Path(model_name_or_path)
    if local_path.exists():
        return local_path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
        )
    )


def _sanitize_modernbert_rope_config(config_dict: dict) -> dict:
    cleaned = dict(config_dict)

    # Remove top-level legacy scalar keys.
    cleaned.pop("rope_type", None)
    cleaned.pop("rope_theta", None)

    # Keep only per-attention mappings for both old and new key names.
    for rope_key in ("rope_scaling", "rope_parameters"):
        rope = cleaned.get(rope_key)
        if isinstance(rope, dict):
            cleaned[rope_key] = {
                key: value
                for key, value in rope.items()
                if key in _ATTENTION_ROPE_KEYS and isinstance(value, dict)
            }

    return cleaned


def _copy_model_dir(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _patch_modernbert_rope_standardization() -> None:
    """Patch ModernBERT RoPE handling for strict HF validation.

    Some Transformers/HF version combinations (observed with modern preview
    versions around transformers>=5.8 and strict huggingface_hub dataclasses)
    construct a ModernBERT
    rope_parameters dict with extra top-level keys such as "rope_type".
    Strict validation expects only:

        {
          "sliding_attention": {...},
          "full_attention": {...},
        }

    This compatibility shim keeps only those two per-attention entries.
    """
    try:
        from transformers.models.modernbert.configuration_modernbert import (
            ModernBertConfig,
        )
    except Exception:
        return

    if getattr(ModernBertConfig, "_modernmolbert_rope_patch_applied", False):
        return

    def _clean_or_build_rope(self) -> dict[str, dict[str, float | str]]:
        existing = getattr(self, "rope_parameters", None)

        if existing is None:
            existing = getattr(self, "rope_scaling", None)

        if isinstance(existing, dict):
            cleaned = {
                key: value
                for key, value in existing.items()
                if key in {"sliding_attention", "full_attention"}
                and isinstance(value, dict)
            }

            if set(cleaned) == {"sliding_attention", "full_attention"}:
                return cleaned

        local_theta = getattr(self, "local_rope_theta", 10000.0)
        global_theta = getattr(self, "global_rope_theta", 160000.0)

        return {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": float(local_theta),
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": float(global_theta),
            },
        }

    def standardize_rope_params(self) -> None:
        object.__setattr__(self, "rope_parameters", _clean_or_build_rope(self))

    def convert_rope_params_to_dict(self, **kwargs):
        object.__setattr__(self, "rope_parameters", _clean_or_build_rope(self))

        # Prevent legacy keys from being processed again downstream.
        kwargs.pop("rope_scaling", None)
        kwargs.pop("rope_parameters", None)
        kwargs.pop("rope_type", None)
        kwargs.pop("rope_theta", None)

        return kwargs

    ModernBertConfig.standardize_rope_params = standardize_rope_params
    ModernBertConfig.convert_rope_params_to_dict = convert_rope_params_to_dict
    setattr(ModernBertConfig, "_modernmolbert_rope_patch_applied", True)
