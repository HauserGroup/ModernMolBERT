import pandas as pd
import numpy as np

from importlib import import_module
from os.path import join
from typing import Any, Literal

from .types import Dataset

# Tuple of train, val, test indices
Splits = dict[str, list[int]]


def ogb_solver(name: str, root: str, load_graphs: bool = False) -> tuple[pd.DataFrame, Splits]:
    from ogb.graphproppred import PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name=name, root=root)

    smiles = pd.read_csv(f"{root}/{name.replace('-', '_')}/mapping/mol.csv.gz").drop(
        columns=["mol_id"]
    )
    if load_graphs:
        smiles["graph"] = pd.Series(
            [
                {
                    "edge_index": data.edge_index.numpy(),
                    "edge_feat": data.edge_attr.numpy(),
                    "node_feat": data.x.numpy(),
                    "num_nodes": data.num_nodes,
                    "x": data.x.numpy(),  # <- important, Torch tensor breaks joblib
                    "y": data.y.numpy(),
                }
                for data in dataset
            ],
            index=smiles.index,
            dtype="object",
        )

    raw_splits = dataset.get_idx_split()
    splits: Splits = {
        "train": list(raw_splits["train"].tolist()),
        "valid": list(raw_splits["valid"].tolist()),
        "test": list(raw_splits["test"].tolist()),
    }
    return smiles, splits


def load_tdc_module_dataset(
    module, name: str, root: str, label: str | None = None, load_graphs: bool = False
) -> tuple[pd.DataFrame, Splits]:
    if label is not None:
        kwargs = {"label_name": label}
    else:
        kwargs = {}
    data = module(name=name, path=root, **kwargs)
    splits = data.get_split(method="scaffold")
    splits["train"]["split"] = "train"
    splits["valid"]["split"] = "train"
    splits["test"]["split"] = "test"

    dataset = pd.concat([splits["train"], splits["valid"], splits["test"]]).reset_index(drop=True)

    if load_graphs:
        from ogb.utils import smiles2graph

        MolConvert = import_module("tdc.chem_utils").MolConvert

        converter = MolConvert(src="SMILES", dst="PyG")
        dataset["graph"] = dataset["Drug"].map(
            lambda smiles: {
                **smiles2graph(smiles),
                "x": converter(smiles).x,
                "y": np.array([[0]]),
            }
        )

    splits = {
        "train": dataset[dataset["split"] == "train"].index.tolist(),
        "valid": [],
        "test": dataset[dataset["split"] == "test"].index.tolist(),
    }

    return dataset.rename(columns={"Drug": "smiles"}).drop(
        ["Drug_ID"], axis=1, errors="ignore"
    ), splits


def tdc_admet_solver(
    module, name: str, root: str, load_graphs: bool = False
) -> tuple[pd.DataFrame, Splits]:
    data = module(name=name, path=root)
    split = data.get_split()

    train, valid, test = split["train"], split["valid"], split["test"]

    dataset = pd.concat([train, valid, test]).reset_index(drop=True)

    if load_graphs:
        from ogb.utils import smiles2graph

        MolConvert = import_module("tdc.chem_utils").MolConvert

        converter = MolConvert(src="SMILES", dst="PyG")
        dataset["graph"] = dataset["Drug"].map(
            lambda smiles: {
                **smiles2graph(smiles),
                "x": converter(smiles).x,
                "y": np.array([[0]]),
            }
        )

    admet_group = import_module("tdc.benchmark_group").admet_group

    group = admet_group(path="data/")
    benchmark = group.get(name)

    return (
        dataset.rename(columns={"Drug": "smiles"}).drop(["Drug_ID"], axis=1, errors="ignore"),
        {
            "train": list(
                benchmark["train_val"].merge(
                    dataset.reset_index().groupby(["Drug_ID", "Drug"]).first(),
                    on=["Drug_ID", "Drug"],
                )["index"]
            ),
            "valid": [],
            "test": list(
                benchmark["test"].merge(
                    dataset.reset_index().groupby(["Drug_ID", "Drug"]).first(),
                    on=["Drug_ID", "Drug"],
                )["index"]
            ),
        },
    )


def get_tdc_group(group_name: str):
    single_pred = import_module("tdc.single_pred")
    ADME = single_pred.ADME
    HTS = single_pred.HTS
    Tox = single_pred.Tox

    return {
        "ADME": ADME,
        "TOX": Tox,
        "HTS": HTS,
    }[group_name]


def get_tdc_solver(benchmark: str):
    return {
        "admet": tdc_admet_solver,
    }[benchmark]


def build_dataset(name: str, task: str, raw_data: pd.DataFrame, splits: Splits) -> Dataset:
    from rdkit import Chem

    raw_data["smiles"] = raw_data["smiles"].map(Chem.CanonSmiles)
    task_lower = task.lower()
    if task_lower not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task}")
    typed_task: Literal["classification", "regression"] = (
        "classification" if task_lower == "classification" else "regression"
    )
    return Dataset(name=name, data=raw_data, splits=splits, task=typed_task)


def load(dataset_config: Any, raw_dir: str, resolve_from_cwd: bool = True) -> Dataset:
    if resolve_from_cwd:
        raw_dir = join(".", raw_dir)

    if dataset_config.source.name == "TDC" and "benchmark" in dataset_config.source:
        module = get_tdc_group(dataset_config.source.group)
        solver = get_tdc_solver(dataset_config.source.benchmark)
        raw_data, splits = solver(module, dataset_config.name, raw_dir, load_graphs=True)
    elif dataset_config.source.name == "TDC":
        raw_data, splits = load_tdc_module_dataset(
            module=get_tdc_group(dataset_config.source.group),
            name=dataset_config.source.collection_name
            if "labels" in dataset_config.source
            else dataset_config.name,
            root=raw_dir,
            label=dataset_config.source.labels if "labels" in dataset_config.source else None,
            load_graphs=True,
        )
    elif dataset_config.source.name == "OGB":
        raw_data, splits = ogb_solver(dataset_config.name, raw_dir, load_graphs=True)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_config.source.name}")

    return build_dataset(
        name=dataset_config.name, task=dataset_config.task, raw_data=raw_data, splits=splits
    )
