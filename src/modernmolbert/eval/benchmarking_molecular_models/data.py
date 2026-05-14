from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "bbbp",
    "bace",
    "esol",
]


def default_catalog_path() -> Path:
    return Path(__file__).with_name("datasets.yaml")


def load_dataset_catalog(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load benchmark dataset entries keyed by dataset name."""

    import yaml

    catalog_path = default_catalog_path() if path is None else Path(path)
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset catalog must be a mapping: {catalog_path}")

    entries = raw.get("datasets", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Dataset catalog must contain a non-empty 'datasets' list: {catalog_path}"
        )

    catalog: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset catalog entries must be mappings: {entry!r}")

        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"Dataset catalog entry is missing a name: {entry!r}")
        if name in catalog:
            raise ValueError(f"Duplicate dataset catalog entry: {name}")

        catalog[name] = {str(key): value for key, value in entry.items()}

    return catalog


def select_dataset_configs(
    dataset_names: list[str] | None,
    *,
    catalog_path: str | Path | None = None,
    prepared_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Select dataset configs for the existing ModernMolBERT eval suite runner."""

    catalog = load_dataset_catalog(catalog_path)
    selected_names = dataset_names if dataset_names else DEFAULT_DATASETS

    configs: list[dict[str, Any]] = []
    missing = [name for name in selected_names if name not in catalog]
    if missing:
        raise ValueError(f"Unknown datasets: {missing}. Known datasets: {sorted(catalog)}")

    for name in selected_names:
        config = dict(catalog[name])
        if prepared_root is not None and config.get("loader") == "prepared_moleculenet":
            config["dataset_dir"] = str(Path(prepared_root) / name)
        configs.append(config)

    return configs
