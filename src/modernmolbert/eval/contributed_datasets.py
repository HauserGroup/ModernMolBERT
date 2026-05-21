from pathlib import Path

import pandas as pd

from modernmolbert.eval.dataset_registry import DatasetSpec, register_dataset  # noqa: F401
from modernmolbert.eval.datasets import EvalDataset, make_eval_dataset_from_splits


def _drop_missing_labels(
    frame: pd.DataFrame,
    *,
    task_names: list[str],
) -> pd.DataFrame:
    """Drop rows with missing task labels.

    Contributed datasets should not return NaN labels inside EvalDataset.
    If raw source files contain missing labels, loaders should drop those rows
    before constructing EvalDataset.
    """

    out = frame.copy()

    for task in task_names:
        out[task] = pd.to_numeric(out[task], errors="coerce")

    return out.dropna(subset=task_names).reset_index(drop=True)


def _validate_binary_labels(
    frame: pd.DataFrame,
    *,
    task_name: str,
    split_name: str,
) -> None:
    """Validate that a binary-classification split contains only 0/1 labels."""

    values = set(frame[task_name].astype(int).unique().tolist())
    invalid = values - {0, 1}
    if invalid:
        raise ValueError(
            f"{split_name} split has non-binary labels for {task_name!r}: "
            f"{sorted(invalid)}. Binary classification labels must be 0/1."
        )


def load_example_activity_dataset(*, root: str | Path) -> EvalDataset:
    """Example contributed binary-classification dataset loader.

    This is the reference pattern for contributed datasets.

    Expected files:

        root/
          train.csv
          valid.csv      # optional
          test.csv

    Required columns:

        smiles, active

    Requirements:

    - `smiles` must contain molecular SMILES strings.
    - `active` must be binary: 0 or 1.
    - rows with missing labels are dropped by this loader.
    - returned EvalDataset splits must not contain NaN labels.
    - task weights are not supported.
    """

    root = Path(root)
    task_names = ["active"]

    train = pd.read_csv(root / "train.csv")
    valid_path = root / "valid.csv"
    valid = pd.read_csv(valid_path) if valid_path.exists() else None
    test = pd.read_csv(root / "test.csv")

    train = _drop_missing_labels(train, task_names=task_names)
    test = _drop_missing_labels(test, task_names=task_names)
    if valid is not None:
        valid = _drop_missing_labels(valid, task_names=task_names)

    _validate_binary_labels(train, task_name="active", split_name="train")
    _validate_binary_labels(test, task_name="active", split_name="test")
    if valid is not None:
        _validate_binary_labels(valid, task_name="active", split_name="valid")

    return make_eval_dataset_from_splits(
        name="example_activity",
        task_type="classification",
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column="smiles",
        metadata={
            "source": "Example only",
            "split_source": "predefined_split",
            "missing_label_policy": "Rows with missing labels are dropped in the loader.",
            "label_definition": "active is binary: 0=inactive, 1=active.",
        },
    )


def load_esol(*, root: str | Path) -> EvalDataset:
    root = Path(root)
    task_names = ["target"]

    train = pd.read_csv(root / "train.csv")
    valid_path = root / "valid.csv"
    valid = pd.read_csv(valid_path) if valid_path.exists() else None
    test = pd.read_csv(root / "test.csv")

    train = _drop_missing_labels(train, task_names=task_names)
    test = _drop_missing_labels(test, task_names=task_names)
    if valid is not None:
        valid = _drop_missing_labels(valid, task_names=task_names)

    return make_eval_dataset_from_splits(
        name="esol",
        task_type="regression",
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column="smiles",
        metadata={"source": "MoleculeNet ESOL", "missing_label_policy": "Dropped"},
    )


def load_clintox(*, root: str | Path) -> EvalDataset:
    root = Path(root)
    task_names = ["FDA_APPROVED"]

    train = pd.read_csv(root / "train.csv")
    valid_path = root / "valid.csv"
    valid = pd.read_csv(valid_path) if valid_path.exists() else None
    test = pd.read_csv(root / "test.csv")

    train = _drop_missing_labels(train, task_names=task_names)
    test = _drop_missing_labels(test, task_names=task_names)
    if valid is not None:
        valid = _drop_missing_labels(valid, task_names=task_names)

    _validate_binary_labels(train, task_name="FDA_APPROVED", split_name="train")
    _validate_binary_labels(test, task_name="FDA_APPROVED", split_name="test")
    if valid is not None:
        _validate_binary_labels(valid, task_name="FDA_APPROVED", split_name="valid")

    return make_eval_dataset_from_splits(
        name="clintox",
        task_type="classification",
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column="smiles",
        metadata={"source": "MoleculeNet ClinTox", "missing_label_policy": "Dropped"},
    )


def load_tdc_herg_blockers(*, root: str | Path) -> EvalDataset:
    root = Path(root)
    task_names = ["herg_blocker"]

    def read_split(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        return frame.rename(columns={"Drug": "smiles", "Y": "herg_blocker"})

    train = read_split(root / "train.csv")
    valid_path = root / "valid.csv"
    valid = read_split(valid_path) if valid_path.exists() else None
    test = read_split(root / "test.csv")

    train = _drop_missing_labels(train, task_names=task_names)
    test = _drop_missing_labels(test, task_names=task_names)
    if valid is not None:
        valid = _drop_missing_labels(valid, task_names=task_names)

    _validate_binary_labels(train, task_name="herg_blocker", split_name="train")
    _validate_binary_labels(test, task_name="herg_blocker", split_name="test")
    if valid is not None:
        _validate_binary_labels(valid, task_name="herg_blocker", split_name="valid")

    return make_eval_dataset_from_splits(
        name="tdc_herg_blockers",
        task_type="classification",
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column="smiles",
        metadata={
            "source": "Therapeutic Data Commons (TDC)",
            "source_url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-blockers",
            "license": "CC-BY-4.0",
            "split_source": "tdc_scaffold_split",
            "missing_label_policy": "Rows with missing labels are dropped in the loader.",
            "label_definition": "herg_blocker is binary: 0=non-blocker, 1=hERG blocker.",
        },
    )


def load_tdc_caco2_wang(*, root: str | Path) -> EvalDataset:
    root = Path(root)
    task_names = ["caco2_permeability"]

    def read_split(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        return frame.rename(columns={"Drug": "smiles", "Y": "caco2_permeability"})

    train = read_split(root / "train.csv")
    valid_path = root / "valid.csv"
    valid = read_split(valid_path) if valid_path.exists() else None
    test = read_split(root / "test.csv")

    train = _drop_missing_labels(train, task_names=task_names)
    test = _drop_missing_labels(test, task_names=task_names)
    if valid is not None:
        valid = _drop_missing_labels(valid, task_names=task_names)

    return make_eval_dataset_from_splits(
        name="tdc_caco2_wang",
        task_type="regression",
        task_names=task_names,
        train=train,
        valid=valid,
        test=test,
        smiles_column="smiles",
        metadata={
            "source": "Therapeutic Data Commons (TDC)",
            "source_url": "https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al",
            "license": "CC-BY-4.0",
            "split_source": "tdc_scaffold_split",
            "missing_label_policy": "Rows with missing labels are dropped in the loader.",
            "label_definition": (
                "caco2_permeability is the continuous TDC-provided regression target "
                "for Caco-2 cell effective permeability."
            ),
        },
    )


def register_contributed_datasets() -> None:
    """Register project-maintained contributed datasets.

    Add real contributed datasets here. Registration is explicit so importing
    modernmolbert.eval does not read local files or perform expensive work.
    """

    # Example registration pattern. Keep commented because this is documentation,
    # not a real bundled dataset.
    #
    # register_dataset(
    #     DatasetSpec(
    #         name="example_activity",
    #         task_type="classification",
    #         task_names=("active",),
    #         loader=load_example_activity_dataset,
    #         description="Example binary activity dataset.",
    #         source="https://example.org",
    #         citation="Example et al. 2026",
    #         license="CC-BY-4.0",
    #     )
    # )

    register_dataset(
        DatasetSpec(
            name="esol",
            task_type="regression",
            task_names=("target",),
            loader=load_esol,
            description="ESOL water solubility regression from MoleculeNet.",
            source="https://moleculenet.org/datasets-1",
            citation="Wu et al. MoleculeNet: a benchmark for molecular machine learning. Chemical Science (2018).",
            license="MIT",
        )
    )

    register_dataset(
        DatasetSpec(
            name="clintox",
            task_type="classification",
            task_names=("FDA_APPROVED",),
            loader=load_clintox,
            description="ClinTox FDA approval classification from MoleculeNet.",
            source="https://moleculenet.org/datasets-1",
            citation="Wu et al. MoleculeNet: a benchmark for molecular machine learning. Chemical Science (2018).",
            license="MIT",
        )
    )

    register_dataset(
        DatasetSpec(
            name="tdc_herg_blockers",
            task_type="classification",
            task_names=("herg_blocker",),
            loader=load_tdc_herg_blockers,
            description="TDC hERG blocker binary classification benchmark.",
            source="https://tdcommons.ai/single_pred_tasks/tox/#herg-blockers",
            citation="Therapeutic Data Commons",
            license="CC-BY-4.0",
        )
    )

    register_dataset(
        DatasetSpec(
            name="tdc_caco2_wang",
            task_type="regression",
            task_names=("caco2_permeability",),
            loader=load_tdc_caco2_wang,
            description="TDC Caco2 Wang Caco-2 permeability regression benchmark.",
            source="https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al",
            citation="Wang et al. 2016; Therapeutic Data Commons",
            license="CC-BY-4.0",
        )
    )

    return None
