from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from modernmolbert.eval.datasets import EvalDataset


TaskType = Literal["classification", "regression"]
DatasetLoader = Callable[..., EvalDataset]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and loader for a contributed benchmark dataset.

    Supported contributed datasets are regression and binary classification
    datasets. The loader must return an EvalDataset with fixed train/test
    splits and no missing task labels.
    """

    name: str
    task_type: TaskType
    task_names: tuple[str, ...]
    loader: DatasetLoader
    description: str
    source: str | None = None
    citation: str | None = None
    license: str | None = None
    expected_smiles_column: str = "smiles"
    expected_selfies_column: str = "selfies"
    metadata: Mapping[str, Any] = field(default_factory=dict)


DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    """Register a contributed benchmark dataset."""

    spec = _normalized_spec(spec)

    if not spec.name:
        raise ValueError("DatasetSpec.name must be non-empty")

    if spec.name in DATASET_REGISTRY:
        raise ValueError(f"Dataset {spec.name!r} is already registered")

    if spec.task_type not in {"classification", "regression"}:
        raise ValueError(f"Dataset {spec.name!r} has invalid task_type: {spec.task_type!r}")

    if not spec.task_names:
        raise ValueError(f"Dataset {spec.name!r} must define at least one task")

    if any(not task for task in spec.task_names):
        raise ValueError(f"Dataset {spec.name!r} contains an empty task name")

    if not callable(spec.loader):
        raise ValueError(f"Dataset {spec.name!r} loader must be callable")

    DATASET_REGISTRY[spec.name] = spec


def list_registered_datasets() -> list[dict[str, Any]]:
    """List registered contributed datasets."""

    return [
        {
            "name": spec.name,
            "task_type": spec.task_type,
            "task_names": list(spec.task_names),
            "description": spec.description,
            "source": spec.source,
            "citation": spec.citation,
            "license": spec.license,
        }
        for spec in DATASET_REGISTRY.values()
    ]


def load_registered_dataset(
    name: str,
    *,
    root: str | Path | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> EvalDataset:
    """Load a registered dataset by name."""

    name = name.strip()

    if name not in DATASET_REGISTRY:
        if DATASET_REGISTRY:
            valid = ", ".join(sorted(DATASET_REGISTRY))
            raise ValueError(f"Unknown registered dataset {name!r}. Valid choices: {valid}")
        raise ValueError(f"Unknown registered dataset {name!r}. No datasets are registered.")

    spec = DATASET_REGISTRY[name]
    loader_kwargs = {} if kwargs is None else dict(kwargs)

    if root is not None:
        loader_kwargs.setdefault("root", Path(root))

    dataset = spec.loader(**loader_kwargs)

    if dataset.name != spec.name:
        raise ValueError(
            f"Registered loader for {spec.name!r} returned dataset named {dataset.name!r}"
        )

    if dataset.task_type != spec.task_type:
        raise ValueError(
            f"Registered loader for {spec.name!r} returned task_type "
            f"{dataset.task_type!r}, expected {spec.task_type!r}"
        )

    if tuple(dataset.task_names) != spec.task_names:
        raise ValueError(
            f"Registered loader for {spec.name!r} returned tasks "
            f"{dataset.task_names!r}, expected {list(spec.task_names)!r}"
        )

    dataset.check()
    _validate_no_missing_task_labels(dataset)
    _validate_binary_classification_labels(dataset)

    return dataset


def _normalized_spec(spec: DatasetSpec) -> DatasetSpec:
    return DatasetSpec(
        name=spec.name.strip(),
        task_type=spec.task_type,
        task_names=tuple(task.strip() for task in spec.task_names),
        loader=spec.loader,
        description=spec.description.strip(),
        source=spec.source,
        citation=spec.citation,
        license=spec.license,
        expected_smiles_column=spec.expected_smiles_column,
        expected_selfies_column=spec.expected_selfies_column,
        metadata=spec.metadata,
    )


def _validate_no_missing_task_labels(dataset: EvalDataset) -> None:
    for split_name, frame in dataset.iter_splits(include_valid=True):
        for task in dataset.task_names:
            if frame[task].isna().any():
                raise ValueError(
                    f"Registered dataset {dataset.name!r} returned missing labels "
                    f"in split {split_name!r}, task {task!r}. Loaders must drop "
                    "missing labels before returning EvalDataset."
                )


def _validate_binary_classification_labels(dataset: EvalDataset) -> None:
    if dataset.task_type != "classification":
        return

    for split_name, frame in dataset.iter_splits(include_valid=True):
        for task in dataset.task_names:
            values = set(frame[task].astype(int).unique().tolist())
            invalid = values - {0, 1}
            if invalid:
                raise ValueError(
                    f"Registered classification dataset {dataset.name!r} returned "
                    f"non-binary labels in split {split_name!r}, task {task!r}: "
                    f"{sorted(invalid)}"
                )
