from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def as_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: as_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [as_config(v) for v in value]
    return value


def load_yaml_config(path: str | Path) -> Config:
    with Path(path).open() as f:
        return as_config(yaml.safe_load(f) or {})


def load_embedding_config(config_dir: str | Path) -> Config:
    return load_yaml_config(Path(config_dir) / "embedding" / "default.yaml")


def load_dataset_registry(config_dir: str | Path) -> Config:
    registry = load_yaml_config(Path(config_dir) / "datasets.yaml")
    if "datasets" not in registry:
        raise ValueError("Dataset registry must contain a top-level 'datasets' mapping.")
    return registry.datasets


def load_dataset_config(config_dir: str | Path, dataset: str) -> Config:
    registry = load_dataset_registry(config_dir)
    dataset_name = Path(dataset).stem
    if dataset_name not in registry:
        raise FileNotFoundError(f"Unknown dataset config: {dataset_name}")
    return registry[dataset_name]


def expand_dataset_selection(config_dir: str | Path, selections: list[str]) -> list[str]:
    registry = load_dataset_registry(config_dir)
    available = sorted(registry.keys())
    selected: list[str] = []

    for selection in selections:
        if selection == "all":
            selected.extend(available)
        elif any(ch in selection for ch in "*?[]"):
            selected.extend(name for name in available if fnmatch(name, selection))
        else:
            selected.append(Path(selection).stem)

    deduped = list(dict.fromkeys(selected))
    missing = [name for name in deduped if name not in available]
    if missing:
        raise FileNotFoundError(f"Unknown dataset config(s): {', '.join(missing)}")
    return deduped
