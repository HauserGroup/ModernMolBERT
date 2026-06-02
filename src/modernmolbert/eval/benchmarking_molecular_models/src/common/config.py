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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


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


def _strip_task_prefix(name: str) -> str:
    for prefix in ("clf_", "reg_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def expand_dataset_selection(config_dir: str | Path, selections: list[str] | str) -> list[str]:
    registry = load_dataset_registry(config_dir)
    available = sorted(registry.keys())
    by_bare_name = {_strip_task_prefix(k): k for k in available}
    selected: list[str] = []

    for selection in as_list(selections):
        if selection == "all":
            selected.extend(available)
        elif any(ch in selection for ch in "*?[]"):
            selected.extend(name for name in available if fnmatch(name, selection))
        else:
            stem = Path(selection).stem
            if stem in available:
                selected.append(stem)
            elif stem in by_bare_name:
                selected.append(by_bare_name[stem])
            else:
                selected.append(stem)  # will be caught by missing check below

    deduped = list(dict.fromkeys(selected))
    missing = [name for name in deduped if name not in available]
    if missing:
        raise FileNotFoundError(f"Unknown dataset config(s): {', '.join(missing)}")
    return deduped
