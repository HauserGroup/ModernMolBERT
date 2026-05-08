from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class MolNetSpec:
    """Description of one supported MoleculeNet dataset."""

    name: str
    loader_name: str
    task_type: TaskType
    preferred_metric: str


CORE_SPECS: dict[str, MolNetSpec] = {
    "esol": MolNetSpec(
        name="esol",
        loader_name="load_delaney",
        task_type="regression",
        preferred_metric="rmse",
    ),
    "freesolv": MolNetSpec(
        name="freesolv",
        loader_name="load_freesolv",
        task_type="regression",
        preferred_metric="rmse",
    ),
    "lipophilicity": MolNetSpec(
        name="lipophilicity",
        loader_name="load_lipo",
        task_type="regression",
        preferred_metric="rmse",
    ),
    "bbbp": MolNetSpec(
        name="bbbp",
        loader_name="load_bbbp",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
    "bace": MolNetSpec(
        name="bace",
        loader_name="load_bace_classification",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
    "clintox": MolNetSpec(
        name="clintox",
        loader_name="load_clintox",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
    "tox21": MolNetSpec(
        name="tox21",
        loader_name="load_tox21",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
    "sider": MolNetSpec(
        name="sider",
        loader_name="load_sider",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
}


EXTENDED_SPECS: dict[str, MolNetSpec] = {
    "hiv": MolNetSpec(
        name="hiv",
        loader_name="load_hiv",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
    "muv": MolNetSpec(
        name="muv",
        loader_name="load_muv",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
    "toxcast": MolNetSpec(
        name="toxcast",
        loader_name="load_toxcast",
        task_type="classification",
        preferred_metric="roc_auc",
    ),
}


ALL_SPECS: dict[str, MolNetSpec] = {
    **CORE_SPECS,
    **EXTENDED_SPECS,
}


def prepare_many(
    *,
    dataset_names: Sequence[str],
    output_root: Path,
    deepchem_data_dir: Path,
    deepchem_save_dir: Path,
    split: str = "scaffold",
    keep_invalid: bool = False,
) -> None:
    """Prepare several MoleculeNet datasets as local sanitized Parquet files."""

    output_root.mkdir(parents=True, exist_ok=True)
    deepchem_data_dir.mkdir(parents=True, exist_ok=True)
    deepchem_save_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        if dataset_name not in ALL_SPECS:
            valid = ", ".join(sorted(ALL_SPECS))
            raise ValueError(
                f"Unknown dataset {dataset_name!r}. Valid choices: {valid}"
            )

        prepare_dataset(
            spec=ALL_SPECS[dataset_name],
            output_root=output_root,
            deepchem_data_dir=deepchem_data_dir,
            deepchem_save_dir=deepchem_save_dir,
            split=split,
            keep_invalid=keep_invalid,
        )


def prepare_dataset(
    *,
    spec: MolNetSpec,
    output_root: Path,
    deepchem_data_dir: Path,
    deepchem_save_dir: Path,
    split: str = "scaffold",
    keep_invalid: bool = False,
) -> None:
    """Prepare one MoleculeNet dataset.

    We load unsplit data from DeepChem, sanitize once, then split locally.
    This avoids DeepChem/RDKit scaffold splitting before invalid molecules are
    removed.
    """

    dataset_out = output_root / spec.name
    dataset_out.mkdir(parents=True, exist_ok=True)

    print(f"[{spec.name}] loading unsplit via DeepChem loader {spec.loader_name!r}...")
    tasks, datasets, transformers = load_deepchem_molnet_unsplit(
        spec=spec,
        data_dir=deepchem_data_dir,
        save_dir=deepchem_save_dir,
    )

    if len(datasets) != 1:
        raise ValueError(
            f"Expected unsplit DeepChem loader to return one dataset for {spec.name}, "
            f"got {len(datasets)}"
        )

    print(f"[{spec.name}] extracting labels and SMILES...")
    frame = deepchem_dataset_to_frame(dataset=datasets[0], tasks=tasks)

    print(f"[{spec.name}] sanitizing SMILES and converting to SELFIES...")
    frame = sanitize_frame(frame)

    n_total = len(frame)
    n_valid = int(frame["is_valid"].sum())
    n_invalid = n_total - n_valid

    if keep_invalid:
        split_frame = frame.copy()
    else:
        split_frame = frame.loc[frame["is_valid"]].reset_index(drop=True)

    print(
        f"[{spec.name}] rows={n_total}, valid={n_valid}, invalid={n_invalid}, "
        f"kept={len(split_frame)}"
    )

    print(f"[{spec.name}] splitting locally with split={split!r}...")
    splits = split_sanitized_frame(
        split_frame,
        split=split,
        seed=13,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
    )

    split_counts: dict[str, dict[str, Any]] = {}

    for split_name, split_df in splits.items():
        out_path = dataset_out / f"{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)

        split_counts[split_name] = {
            "rows": int(len(split_df)),
            "valid_rows": int(split_df["is_valid"].sum()),
            "invalid_rows": int((~split_df["is_valid"]).sum()),
            "path": str(out_path),
        }

    example_path = dataset_out / "example.tsv"

    write_example_tsv(
        splits=splits,
        output_path=example_path,
        n=100,
    )

    metadata = {
        "name": spec.name,
        "deepchem_loader": spec.loader_name,
        "task_type": spec.task_type,
        "preferred_metric": spec.preferred_metric,
        "tasks": list(tasks),
        "split": split,
        "split_source": "modernmolbert_local_after_sanitization",
        "keep_invalid": keep_invalid,
        "n_transformers": len(transformers),
        "example_tsv": str(example_path),
        "row_counts": {
            "raw_total": int(n_total),
            "valid_after_sanitization": int(n_valid),
            "invalid_after_sanitization": int(n_invalid),
            "kept_for_splitting": int(len(split_frame)),
        },
        "columns": {
            "smiles_raw": "raw SMILES-like value extracted from DeepChem",
            "smiles_canonical": "RDKit canonical isomeric SMILES",
            "selfies": "SELFIES encoded from smiles_canonical",
            "is_valid": "True if RDKit parse and SELFIES conversion succeeded",
            "sanitize_error": "error label for invalid rows",
        },
        "splits": split_counts,
    }

    metadata_path = dataset_out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[{spec.name}] wrote {dataset_out}")


def load_deepchem_molnet_unsplit(
    *,
    spec: MolNetSpec,
    data_dir: Path,
    save_dir: Path,
):
    """Load a MoleculeNet dataset from DeepChem without splitting.

    We intentionally do not let DeepChem split the dataset. Some MoleculeNet

    datasets contain invalid or awkward SMILES, and DeepChem's scaffold splitter

    calls RDKit before we have a chance to sanitize/drop invalid rows.

    """

    import deepchem as dc

    loader: Callable[..., Any] = getattr(dc.molnet, spec.loader_name)  # type: ignore

    tasks, datasets, transformers = loader(
        featurizer="Raw",
        splitter=None,
        reload=True,
        data_dir=str(data_dir),
        save_dir=str(save_dir),
    )

    return list(tasks), datasets, transformers


def infer_split_names(n_splits: int) -> list[str]:
    """Infer conventional split names from DeepChem loader output length."""

    if n_splits == 3:
        return ["train", "valid", "test"]

    if n_splits == 1:
        return ["all"]

    return [f"split_{i}" for i in range(n_splits)]


def deepchem_dataset_to_frame(dataset: Any, tasks: Sequence[str]) -> pd.DataFrame:
    """Convert a DeepChem Dataset to a pandas DataFrame.

    This keeps labels as task columns. Missing multitask labels are converted to
    NaN using dataset.w when available.
    """

    smiles_raw = extract_smiles(dataset)

    y = np.asarray(dataset.y)
    w = np.asarray(dataset.w) if getattr(dataset, "w", None) is not None else None

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if len(tasks) != y.shape[1]:
        if y.shape[1] == 1 and len(tasks) == 0:
            tasks = ["label"]
        else:
            raise ValueError(
                f"Task count mismatch: len(tasks)={len(tasks)}, y.shape={y.shape}"
            )

    rows: list[dict[str, Any]] = []

    for i, smi in enumerate(smiles_raw):
        row: dict[str, Any] = {"smiles_raw": smi}

        for j, task in enumerate(tasks):
            value = y[i, j]

            if _label_is_missing(w=w, row_idx=i, task_idx=j):
                row[task] = np.nan
            else:
                row[task] = value

        rows.append(row)

    return pd.DataFrame(rows)


def extract_smiles(dataset: Any) -> list[str]:
    """Best-effort SMILES extraction from a DeepChem Dataset.

    With featurizer='Raw', DeepChem usually places raw SMILES or molecule
    objects in dataset.X. In many MoleculeNet datasets, dataset.ids are also
    SMILES strings. This function tries X first, then ids, then RDKit Mol
    conversion, then str(x) as a last resort.
    """

    smiles_values: list[str] = []

    for x, dataset_id in zip(dataset.X, dataset.ids, strict=False):
        smiles_values.append(extract_one_smiles(x=x, dataset_id=dataset_id))

    return smiles_values


def extract_one_smiles(*, x: Any, dataset_id: Any) -> str:
    """Extract one SMILES string from DeepChem feature/id values."""

    if isinstance(x, str) and x:
        return x

    if isinstance(dataset_id, str) and dataset_id:
        return dataset_id

    if hasattr(x, "GetNumAtoms"):
        from rdkit import Chem

        return Chem.MolToSmiles(x, canonical=False)

    return str(x)


def _label_is_missing(*, w: np.ndarray | None, row_idx: int, task_idx: int) -> bool:
    """Return True when DeepChem's weight matrix marks a label as missing."""

    if w is None:
        return False

    if w.ndim == 1:
        return bool(w[row_idx] == 0)

    if w.ndim == 2:
        return bool(w[row_idx, task_idx] == 0)

    return False


def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical SMILES and SELFIES columns to a frame with smiles_raw."""

    if "smiles_raw" not in df.columns:
        raise ValueError("Expected column 'smiles_raw'")

    canonical_values: list[str | None] = []
    selfies_values: list[str | None] = []
    valid_values: list[bool] = []
    error_values: list[str | None] = []

    for smi in tqdm(df["smiles_raw"].tolist(), desc="Sanitizing SMILES"):
        canonical, selfies, error = canonicalize_and_selfies(str(smi))
        canonical_values.append(canonical)
        selfies_values.append(selfies)
        valid_values.append(error is None)
        error_values.append(error)

    out = df.copy()
    out.insert(1, "smiles_canonical", canonical_values)
    out.insert(2, "selfies", selfies_values)
    out.insert(3, "is_valid", valid_values)
    out.insert(4, "sanitize_error", error_values)

    return out


def canonicalize_and_selfies(smiles: str) -> tuple[str | None, str | None, str | None]:
    """Canonicalize a SMILES string and convert it to SELFIES.

    Returns:
        (canonical_smiles, selfies, error)

    If error is None, both canonical_smiles and selfies should be non-None.
    """

    from rdkit import Chem
    import selfies as sf

    text = smiles.strip()
    if not text:
        return None, None, "empty_smiles"

    try:
        mol = Chem.MolFromSmiles(text)
    except Exception as exc:
        return None, None, f"rdkit_exception:{type(exc).__name__}"

    if mol is None:
        return None, None, "rdkit_parse_failed"

    try:
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        return None, None, f"canonicalization_failed:{type(exc).__name__}"

    try:
        selfies = sf.encoder(canonical)
    except Exception as exc:
        return canonical, None, f"selfies_failed:{type(exc).__name__}"

    if not selfies:
        return canonical, None, "selfies_empty"

    return canonical, selfies, None


def iter_prepared_datasets(root: Path) -> Iterable[Path]:
    """Yield prepared dataset directories under a sanitized MoleculeNet root."""

    if not root.exists():
        return

    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "metadata.json").exists():
            yield path


def split_sanitized_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    seed: int = 13,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Split a sanitized valid-only frame into train/valid/test."""

    if not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    if split == "random":
        return random_split_frame(
            frame,
            seed=seed,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
        )

    if split == "scaffold":
        return scaffold_split_frame(
            frame,
            seed=seed,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
        )

    if split == "index":
        return index_split_frame(
            frame,
            frac_train=frac_train,
            frac_valid=frac_valid,
        )

    raise ValueError(
        f"Unsupported local split {split!r}. Use 'scaffold', 'random', or 'index'."
    )


def random_split_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
) -> dict[str, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return index_split_frame(
        shuffled,
        frac_train=frac_train,
        frac_valid=frac_valid,
    )


def index_split_frame(
    frame: pd.DataFrame,
    *,
    frac_train: float,
    frac_valid: float,
) -> dict[str, pd.DataFrame]:
    n = len(frame)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)

    train = frame.iloc[:n_train].reset_index(drop=True)
    valid = frame.iloc[n_train : n_train + n_valid].reset_index(drop=True)
    test = frame.iloc[n_train + n_valid :].reset_index(drop=True)

    return {
        "train": train,
        "valid": valid,
        "test": test,
    }


def scaffold_split_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
) -> dict[str, pd.DataFrame]:
    """Scaffold split after sanitization.

    This uses smiles_canonical, so invalid molecules have already been removed.
    """

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold_to_indices: dict[str, list[int]] = {}

    for idx, smiles in enumerate(frame["smiles_canonical"].tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Should not happen after sanitization, but keep a fallback.
            scaffold = f"invalid_{idx}"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )

        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    rng = np.random.default_rng(seed)

    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=len, reverse=True)

    # Shuffle groups of equal-ish size lightly while preserving the basic
    # large-scaffold-first behavior.
    # This avoids completely deterministic dataset-order artifacts.
    grouped_by_size: dict[int, list[list[int]]] = {}
    for group in scaffold_groups:
        grouped_by_size.setdefault(len(group), []).append(group)

    shuffled_groups: list[list[int]] = []
    for size in sorted(grouped_by_size, reverse=True):
        groups = grouped_by_size[size]
        rng.shuffle(groups)
        shuffled_groups.extend(groups)

    n_total = len(frame)
    n_train_target = int(frac_train * n_total)
    n_valid_target = int(frac_valid * n_total)

    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []

    for group in shuffled_groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid_target:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return {
        "train": frame.iloc[train_idx].reset_index(drop=True),
        "valid": frame.iloc[valid_idx].reset_index(drop=True),
        "test": frame.iloc[test_idx].reset_index(drop=True),
    }


def write_example_tsv(
    *,
    splits: dict[str, pd.DataFrame],
    output_path: Path,
    n: int = 100,
) -> None:
    """Write a small uncompressed TSV preview of a prepared dataset.

    The preview is for human inspection only. It concatenates splits in
    train/valid/test order when present and adds a `split` column.
    """

    frames: list[pd.DataFrame] = []

    preferred_order = ["train", "valid", "test"]
    ordered_names = [name for name in preferred_order if name in splits] + [
        name for name in sorted(splits) if name not in preferred_order
    ]

    for split_name in ordered_names:
        frame = splits[split_name].copy()
        frame.insert(0, "split", split_name)
        frames.append(frame)

    if frames:
        example = pd.concat(frames, ignore_index=True).head(n)
    else:
        example = pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(output_path, sep="\t", index=False)
