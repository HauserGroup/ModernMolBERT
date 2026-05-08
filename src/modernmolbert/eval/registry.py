import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from modernmolbert.eval.featurizers.base import RepresentationFeaturizer
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer


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

    if "type" not in config:
        raise ValueError(f"Featurizer config {config_path} is missing 'type'")

    featurizer_type = config.pop("type")
    return make_featurizer(featurizer_type, **config)
