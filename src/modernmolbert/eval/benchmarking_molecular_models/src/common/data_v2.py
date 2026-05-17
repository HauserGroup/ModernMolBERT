import pandas as pd

from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
from os.path import join
from random import Random
from typing import Any, Literal
import warnings

from modernmolbert.common.rdkit_safety import looks_like_smiles

from .types import Dataset

# Tuple of train, val, test indices
Splits = dict[str, list[int]]


TDC_METADATA_PATCHES: dict[str, dict[str, Any]] = {
    "pampa_ncats": {"group": "ADME", "id": 6695858, "type": "tab"},
    "approved_pampa_ncats": {"group": "ADME", "id": 6695857, "type": "tab"},
    "herg_karim": {"group": "Tox", "id": 6822246, "type": "tab"},
}


def patch_tdc_metadata(name: str) -> None:
    patch = TDC_METADATA_PATCHES.get(name.lower())
    if patch is None:
        return

    metadata = import_module("tdc.metadata")
    canonical_name = name.lower()
    group = patch["group"]

    if canonical_name not in metadata.dataset_names[group]:
        metadata.dataset_names[group].append(canonical_name)
    if canonical_name not in metadata.dataset_list:
        metadata.dataset_list.append(canonical_name)

    metadata.name2id[canonical_name] = patch["id"]
    metadata.name2type[canonical_name] = patch["type"]


@contextmanager
def torch_load_with_legacy_ogb_defaults():
    import torch

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


@contextmanager
def suppress_outdated_pkg_resources_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        yield


def ogb_solver(name: str, root: str) -> tuple[pd.DataFrame, Splits]:
    """
    Solve an OGB dataset. We don't really do graphs
    """
    with suppress_outdated_pkg_resources_warning():
        try:
            from ogb.graphproppred import PygGraphPropPredDataset as GraphDataset  # type: ignore[import]
        except ImportError:
            from ogb.graphproppred import GraphPropPredDataset as GraphDataset  # type: ignore[import]

        with torch_load_with_legacy_ogb_defaults():
            dataset: Any = GraphDataset(name=name, root=root)

    smiles = pd.read_csv(f"{root}/{name.replace('-', '_')}/mapping/mol.csv.gz").drop(
        columns=["mol_id"]
    )

    raw_splits = dataset.get_idx_split()
    splits: Splits = {
        "train": list(raw_splits["train"].tolist()),
        "valid": list(raw_splits["valid"].tolist()),
        "test": list(raw_splits["test"].tolist()),
    }
    return smiles, splits


def create_tdc_scaffold_split(
    data: Any,
    *,
    seed: int = 42,
    frac: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create a TDC-compatible scaffold split without TDC's broken error handler."""

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")  # type: ignore

    if frac is None:
        frac = [0.7, 0.1, 0.2]

    entity = data.entity1_name
    df = data.get_data(format="df")
    random = Random(seed)

    scaffolds: dict[str, set[int]] = defaultdict(set)
    error_smiles = 0
    for idx, smiles in enumerate(df[entity].values):
        text = str(smiles).strip()
        if not looks_like_smiles(text):
            error_smiles += 1
            continue

        try:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                error_smiles += 1
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )
        except Exception:
            error_smiles += 1
            continue
        scaffolds[scaffold].add(idx)

    split_total = len(df) - error_smiles
    train_size = int(split_total * frac[0])
    valid_size = int(split_total * frac[1])

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []

    big_index_sets = []
    small_index_sets = []
    for index_set in scaffolds.values():
        if (
            len(index_set) > valid_size / 2
            or len(index_set) > (split_total - train_size - valid_size) / 2
        ):
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)

    for index_set in big_index_sets + small_index_sets:
        ordered_index_set = sorted(index_set)
        if len(train) + len(ordered_index_set) <= train_size:
            train += ordered_index_set
        elif frac[2] == 0 or len(valid) + len(ordered_index_set) <= valid_size:
            valid += ordered_index_set
        else:
            test += ordered_index_set

    if error_smiles:
        print(
            f"Warning: TDC scaffold split omitted {error_smiles} SMILES for "
            "which RDKit could not generate a scaffold.",
            flush=True,
        )

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[valid].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }


def load_tdc_module_dataset(
    module,
    name: str,
    root: str,
    label: str | None = None,
) -> tuple[pd.DataFrame, Splits]:
    if label is not None:
        kwargs = {"label_name": label}
    else:
        kwargs = {}

    patch_tdc_metadata(name)
    data = module(name=name, path=root, **kwargs)

    splits = create_tdc_scaffold_split(data)

    splits["train"]["split"] = "train"
    splits["valid"]["split"] = "train"
    splits["test"]["split"] = "test"

    dataset = pd.concat(
        [splits["train"], splits["valid"], splits["test"]],
        ignore_index=True,
    )

    index_splits: Splits = {
        "train": dataset[dataset["split"] == "train"].index.tolist(),
        "valid": [],
        "test": dataset[dataset["split"] == "test"].index.tolist(),
    }

    dataset = dataset.rename(columns={"Drug": "smiles"}).drop(
        ["Drug_ID"],
        axis=1,
        errors="ignore",
    )

    return dataset, index_splits


def tdc_admet_solver(module, name: str, root: str) -> tuple[pd.DataFrame, Splits]:
    data = module(name=name, path=root)
    split = data.get_split()

    train, valid, test = split["train"], split["valid"], split["test"]

    dataset = pd.concat([train, valid, test]).reset_index(drop=True)

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


def canonicalize_smiles(smiles: Any) -> str | None:
    from rdkit import Chem

    text = str(smiles).strip()
    if not looks_like_smiles(text):
        return None

    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonicalize_dataset_smiles(
    name: str,
    raw_data: pd.DataFrame,
    splits: Splits,
) -> tuple[pd.DataFrame, Splits]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # type: ignore

    raw_data = raw_data.reset_index(drop=True)
    canonical_smiles = raw_data["smiles"].map(canonicalize_smiles)
    valid_mask = canonical_smiles.notna()

    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        print(
            f"Warning: Dropped {invalid_count} molecules from {name!r} because RDKit "
            "could not canonicalize their SMILES.",
            flush=True,
        )

    filtered_data = raw_data.loc[valid_mask].copy()
    filtered_data["smiles"] = canonical_smiles.loc[valid_mask].to_numpy()

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_data.index)}
    remapped_splits: Splits = {
        split_name: [old_to_new[idx] for idx in indices if idx in old_to_new]
        for split_name, indices in splits.items()
    }

    return filtered_data.reset_index(drop=True), remapped_splits


def build_dataset(name: str, task: str, raw_data: pd.DataFrame, splits: Splits) -> Dataset:
    raw_data, splits = canonicalize_dataset_smiles(name, raw_data, splits)
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
        raw_data, splits = solver(module, dataset_config.name, raw_dir)
    elif dataset_config.source.name == "TDC":
        raw_data, splits = load_tdc_module_dataset(
            module=get_tdc_group(dataset_config.source.group),
            name=dataset_config.source.collection_name
            if "labels" in dataset_config.source
            else dataset_config.name,
            root=raw_dir,
            label=dataset_config.source.labels if "labels" in dataset_config.source else None,
        )
    elif dataset_config.source.name == "OGB":
        raw_data, splits = ogb_solver(dataset_config.name, raw_dir)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_config.source.name}")

    return build_dataset(
        name=dataset_config.name, task=dataset_config.task, raw_data=raw_data, splits=splits
    )
