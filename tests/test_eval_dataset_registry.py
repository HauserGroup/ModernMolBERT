from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.eval.dataset_registry import (
    DATASET_REGISTRY,
    DatasetSpec,
    load_registered_dataset,
    register_dataset,
)
from modernmolbert.eval.datasets import make_eval_dataset_from_splits


@pytest.fixture(autouse=True)
def restore_dataset_registry():
    old = dict(DATASET_REGISTRY)
    DATASET_REGISTRY.clear()
    yield
    DATASET_REGISTRY.clear()
    DATASET_REGISTRY.update(old)


def test_register_and_load_dataset(tmp_path: Path) -> None:
    def load_toy(*, root: Path):
        train = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]})
        test = pd.DataFrame({"smiles": ["CCC"], "label": [1]})

        return make_eval_dataset_from_splits(
            name="toy",
            task_type="classification",
            task_names="label",
            train=train,
            test=test,
        )

    register_dataset(
        DatasetSpec(
            name="toy",
            task_type="classification",
            task_names=("label",),
            loader=load_toy,
            description="Toy registered dataset.",
        )
    )

    dataset = load_registered_dataset("toy", root=tmp_path)

    assert dataset.name == "toy"
    assert dataset.task_type == "classification"
    assert dataset.task_names == ["label"]


def test_register_dataset_rejects_duplicate() -> None:
    def load_toy():
        train = pd.DataFrame({"smiles": ["CCO"], "label": [1]})
        test = pd.DataFrame({"smiles": ["CCN"], "label": [0]})
        return make_eval_dataset_from_splits(
            name="toy",
            task_type="classification",
            task_names="label",
            train=train,
            test=test,
        )

    spec = DatasetSpec(
        name="toy",
        task_type="classification",
        task_names=("label",),
        loader=load_toy,
        description="Toy registered dataset.",
    )

    register_dataset(spec)

    with pytest.raises(ValueError, match="already registered"):
        register_dataset(spec)


def test_load_registered_dataset_checks_returned_name(tmp_path: Path) -> None:
    def load_bad(*, root: Path):
        train = pd.DataFrame({"smiles": ["CCO"], "label": [1]})
        test = pd.DataFrame({"smiles": ["CCN"], "label": [0]})
        return make_eval_dataset_from_splits(
            name="wrong_name",
            task_type="classification",
            task_names="label",
            train=train,
            test=test,
        )

    register_dataset(
        DatasetSpec(
            name="toy",
            task_type="classification",
            task_names=("label",),
            loader=load_bad,
            description="Bad registered dataset.",
        )
    )

    with pytest.raises(ValueError, match="returned dataset named"):
        load_registered_dataset("toy", root=tmp_path)


def test_load_registered_dataset_checks_returned_tasks(tmp_path: Path) -> None:
    def load_bad(*, root: Path):
        train = pd.DataFrame({"smiles": ["CCO"], "wrong": [1]})
        test = pd.DataFrame({"smiles": ["CCN"], "wrong": [0]})
        return make_eval_dataset_from_splits(
            name="toy",
            task_type="classification",
            task_names="wrong",
            train=train,
            test=test,
        )

    register_dataset(
        DatasetSpec(
            name="toy",
            task_type="classification",
            task_names=("label",),
            loader=load_bad,
            description="Bad registered dataset.",
        )
    )

    with pytest.raises(ValueError, match="returned tasks"):
        load_registered_dataset("toy", root=tmp_path)


def test_register_dataset_normalizes_name_and_tasks() -> None:
    def load_toy():
        train = pd.DataFrame({"smiles": ["CCO"], "label": [1]})
        test = pd.DataFrame({"smiles": ["CCN"], "label": [0]})
        return make_eval_dataset_from_splits(
            name="toy",
            task_type="classification",
            task_names="label",
            train=train,
            test=test,
        )

    register_dataset(
        DatasetSpec(
            name=" toy ",
            task_type="classification",
            task_names=(" label ",),
            loader=load_toy,
            description=" Toy dataset ",
        )
    )

    assert "toy" in DATASET_REGISTRY
    assert DATASET_REGISTRY["toy"].task_names == ("label",)


def test_load_registered_dataset_rejects_missing_labels(tmp_path: Path) -> None:
    def load_bad(*, root: Path):
        train = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, None]})
        test = pd.DataFrame({"smiles": ["CCC"], "label": [1]})
        return make_eval_dataset_from_splits(
            name="toy",
            task_type="classification",
            task_names="label",
            train=train,
            test=test,
        )

    register_dataset(
        DatasetSpec(
            name="toy",
            task_type="classification",
            task_names=("label",),
            loader=load_bad,
            description="Bad toy dataset.",
        )
    )

    with pytest.raises(ValueError, match="missing labels"):
        load_registered_dataset("toy", root=tmp_path)


def test_load_registered_dataset_rejects_non_binary_labels(tmp_path: Path) -> None:
    def load_bad(*, root: Path):
        train = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 2]})
        test = pd.DataFrame({"smiles": ["CCC"], "label": [1]})
        return make_eval_dataset_from_splits(
            name="toy",
            task_type="classification",
            task_names="label",
            train=train,
            test=test,
        )

    register_dataset(
        DatasetSpec(
            name="toy",
            task_type="classification",
            task_names=("label",),
            loader=load_bad,
            description="Bad toy dataset.",
        )
    )

    with pytest.raises(ValueError, match="non-binary labels"):
        load_registered_dataset("toy", root=tmp_path)


def test_load_registered_dataset_unknown_empty_registry_message() -> None:
    with pytest.raises(ValueError, match="No datasets are registered"):
        load_registered_dataset("missing")
