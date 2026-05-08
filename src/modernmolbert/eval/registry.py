import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from modernmolbert.eval.featurizers.base import RepresentationFeaturizer
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.featurizers.rdkit_ecfp import ECFP4Featurizer
from modernmolbert.eval.featurizers.hf_smiles import HuggingFaceSmilesFeaturizer
from modernmolbert.eval.featurizers.modernmolbert_selfies import (
    ModernMolBERTSelfiesFeaturizer,
)


@dataclass(frozen=True)
class FeaturizerSpec:
    name: str
    factory: type[RepresentationFeaturizer] | Callable[..., RepresentationFeaturizer]
    description: str
    required_extra: str | None = None


FEATURIZER_REGISTRY: dict[str, FeaturizerSpec] = {
    "dummy": FeaturizerSpec(
        name="dummy",
        factory=DummyFeaturizer,
        description="Deterministic toy featurizer for tests and plumbing only.",
        required_extra=None,
    ),
    "ecfp4": FeaturizerSpec(
        name="ecfp4",
        factory=ECFP4Featurizer,
        description="RDKit Morgan/ECFP4 fingerprint, radius=2.",
        required_extra="eval-rdkit",
    ),
    "hf_smiles": FeaturizerSpec(
        name="hf_smiles",
        factory=HuggingFaceSmilesFeaturizer,
        description="Generic Hugging Face SMILES encoder featurizer.",
        required_extra="eval-transformers",
    ),
    "modernmolbert_selfies": FeaturizerSpec(
        name="modernmolbert_selfies",
        factory=ModernMolBERTSelfiesFeaturizer,
        description="Trained ModernMolBERT SELFIES encoder.",
        required_extra="eval-transformers",
    ),
}


def list_featurizers() -> list[dict[str, str | None]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "required_extra": spec.required_extra,
        }
        for spec in FEATURIZER_REGISTRY.values()
    ]


def make_featurizer(
    featurizer_type: str,
    **kwargs: Any,
) -> RepresentationFeaturizer:
    if featurizer_type not in FEATURIZER_REGISTRY:
        raise ValueError(
            f"Unknown featurizer {featurizer_type!r}. "
            f"Known featurizers: {sorted(FEATURIZER_REGISTRY)}"
        )

    spec = FEATURIZER_REGISTRY[featurizer_type]
    return spec.factory(**kwargs)


def make_featurizer_from_config(path: str | Path) -> RepresentationFeaturizer:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    featurizer_type = config.get("type") or config.get("name")
    if featurizer_type is None:
        raise ValueError(f"Featurizer config {config_path} is missing 'type' or 'name'")

    kwargs = {
        key: value for key, value in config.items() if key not in {"type", "name"}
    }

    return make_featurizer(featurizer_type, **kwargs)
