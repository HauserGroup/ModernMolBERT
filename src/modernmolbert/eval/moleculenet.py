from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from modernmolbert.rdkit_safety import looks_like_smiles


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
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    keep_invalid: bool = False,
    seed: int = 42,
) -> None:
    """Prepare several MoleculeNet datasets as local sanitized Parquet files."""

    output_root.mkdir(parents=True, exist_ok=True)
    deepchem_data_dir.mkdir(parents=True, exist_ok=True)
    deepchem_save_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []

    for dataset_name in dataset_names:
        if dataset_name not in ALL_SPECS:
            valid = ", ".join(sorted(ALL_SPECS))
            raise ValueError(f"Unknown dataset {dataset_name!r}. Valid choices: {valid}")

        prepare_dataset(
            spec=ALL_SPECS[dataset_name],
            output_root=output_root,
            deepchem_data_dir=deepchem_data_dir,
            deepchem_save_dir=deepchem_save_dir,
            split=split,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            keep_invalid=keep_invalid,
            seed=seed,
        )

        prepared.append(
            {
                "name": dataset_name,
                "path": str(deepchem_save_dir / dataset_name / "example.tsv"),
            }
        )

    manifest = {
        "datasets": prepared,
        "split": split,
        "split_seed": seed,
        "split_fractions": {
            "train": frac_train,
            "valid": frac_valid,
            "test": frac_test,
        },
        "keep_invalid": keep_invalid,
        "versions": collect_preparation_versions(),
    }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(
    *,
    spec: MolNetSpec,
    output_root: Path,
    deepchem_data_dir: Path,
    deepchem_save_dir: Path,
    split: str = "scaffold",
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
    keep_invalid: bool = False,
) -> Path:
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

    tasks = [str(task) for task in tasks]

    if len(datasets) != 1:
        raise ValueError(
            f"Expected unsplit DeepChem loader to return one dataset for {spec.name}, "
            f"got {len(datasets)}"
        )

    if keep_invalid and split == "scaffold":
        raise ValueError(
            "keep_invalid=True is not supported with scaffold split. "
            "Use keep_invalid=False or split='random'."
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
        seed=seed,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
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
        "split_fractions": {
            "train": frac_train,
            "valid": frac_valid,
            "test": frac_test,
        },
        "scaffold_stats": compute_scaffold_stats(split_frame) if split == "scaffold" else None,
        "split_scaffold_stats": {
            split_name: compute_scaffold_stats(split_df) for split_name, split_df in splits.items()
        }
        if split == "scaffold"
        else None,
        "label_stats": compute_task_label_stats(
            splits=splits,
            tasks=tasks,
            task_type=spec.task_type,
        ),
        "duplicate_stats": compute_duplicate_stats(split_frame),
        "split_overlap_stats": compute_split_overlap_stats(
            splits,
            key="smiles_canonical",
        ),
        "versions": collect_preparation_versions(),
        "split_seed": seed,
    }

    metadata_path = dataset_out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[{spec.name}] wrote {dataset_out}")
    return dataset_out


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
            raise ValueError(f"Task count mismatch: len(tasks)={len(tasks)}, y.shape={y.shape}")

    rows: list[dict[str, Any]] = []

    dataset_ids = list(getattr(dataset, "ids", [None] * len(smiles_raw)))

    for i, smi in enumerate(smiles_raw):
        row: dict[str, Any] = {
            "molnet_row_id": i,
            "deepchem_id": str(dataset_ids[i]) if dataset_ids[i] is not None else None,
            "smiles_raw": smi,
        }

        for j, task in enumerate(tasks):
            value = y[i, j]

            if _label_is_missing(w=w, row_idx=i, task_idx=j):
                row[task] = np.nan
            else:
                row[task] = float(value)

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
    if not looks_like_smiles(text):
        return None, None, "rdkit_parse_failed"

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
    group_duplicates: bool = True,
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
            group_duplicates=group_duplicates,
        )

    if split == "scaffold":
        splits = scaffold_split_frame(
            frame,
            seed=seed,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
        )
        if len(splits["valid"]) == 0 or len(splits["test"]) == 0:
            raise RuntimeError(
                "Scaffold split produced an empty valid or test split. "
                "Use random split or adjust fractions."
            )
        return splits

    if split == "index":
        return index_split_frame(
            frame,
            frac_train=frac_train,
            frac_valid=frac_valid,
        )

    raise ValueError(f"Unsupported local split {split!r}. Use 'scaffold', 'random', or 'index'.")


def random_split_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
    group_duplicates: bool = True,
) -> dict[str, pd.DataFrame]:
    if group_duplicates and "smiles_canonical" in frame.columns:
        return grouped_random_split_frame(
            frame,
            group_column="smiles_canonical",
            seed=seed,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
        )

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

    Splits molecules by Bemis-Murcko scaffold using canonical SMILES.
    Scaffold groups are assigned greedily to the currently most underfilled
    split relative to the requested target sizes.
    """

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if "smiles_canonical" not in frame.columns:
        raise ValueError("Expected column 'smiles_canonical' for scaffold split")

    scaffold_to_indices: dict[str, list[int]] = {}

    for idx, smiles in enumerate(frame["smiles_canonical"].tolist()):
        text = str(smiles).strip()
        if not looks_like_smiles(text):
            scaffold = f"invalid_{idx}"
        else:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                scaffold = f"invalid_{idx}"
                scaffold_to_indices.setdefault(scaffold, []).append(idx)
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )

        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    rng = np.random.default_rng(seed)

    groups = list(scaffold_to_indices.values())

    # Shuffle before sorting so equal-sized groups are randomized reproducibly.
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_total = len(frame)
    targets = {
        "train": frac_train * n_total,
        "valid": frac_valid * n_total,
        "test": frac_test * n_total,
    }

    assigned: dict[str, list[int]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for group in groups:
        # Choose the split with the largest relative remaining capacity.
        def remaining_fraction(split_name: str) -> float:
            target = targets[split_name]
            if target <= 0:
                return -np.inf
            return (target - len(assigned[split_name])) / target

        split_name = max(("train", "valid", "test"), key=remaining_fraction)
        assigned[split_name].extend(group)

    return {
        split_name: frame.iloc[indices].reset_index(drop=True)
        for split_name, indices in assigned.items()
    }


def compute_task_label_stats(
    *,
    splits: dict[str, pd.DataFrame],
    tasks: Sequence[str],
    task_type: TaskType,
) -> dict[str, Any]:
    """Compute per-task label availability and class/regression summaries."""

    stats: dict[str, Any] = {}

    for split_name, split_df in splits.items():
        split_stats: dict[str, Any] = {}

        for task in tasks:
            if task not in split_df.columns:
                split_stats[task] = {"error": "missing_task_column"}
                continue

            values = split_df[task]
            observed = values.dropna()

            task_stats: dict[str, Any] = {
                "n_rows": int(len(values)),
                "n_observed": int(observed.shape[0]),
                "n_missing": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()) if len(values) else 0.0,
            }

            if task_type == "classification":
                counts = observed.value_counts(dropna=True).sort_index()
                task_stats["class_counts"] = {str(k): int(v) for k, v in counts.items()}
                task_stats["n_classes_observed"] = int(counts.shape[0])
            else:
                if len(observed):
                    task_stats["mean"] = float(observed.mean())
                    task_stats["std"] = float(observed.std(ddof=0))
                    task_stats["min"] = float(observed.min())
                    task_stats["max"] = float(observed.max())
                else:
                    task_stats["mean"] = None
                    task_stats["std"] = None
                    task_stats["min"] = None
                    task_stats["max"] = None

            split_stats[task] = task_stats

        stats[split_name] = split_stats

    return stats


def compute_scaffold_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute scaffold-size summaries for a sanitized frame."""

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if "smiles_canonical" not in frame.columns:
        return {"error": "missing_smiles_canonical"}

    scaffold_counts: dict[str, int] = {}

    for smiles in frame["smiles_canonical"].dropna().tolist():
        text = str(smiles).strip()
        if not looks_like_smiles(text):
            scaffold = "__invalid__"
        else:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                scaffold = "__invalid__"
            else:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol,
                    includeChirality=False,
                )
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1

    sizes = np.asarray(list(scaffold_counts.values()), dtype=int)

    if sizes.size == 0:
        return {
            "n_scaffolds": 0,
            "largest_scaffold_size": 0,
            "largest_scaffold_fraction": 0.0,
        }

    return {
        "n_scaffolds": int(sizes.size),
        "largest_scaffold_size": int(sizes.max()),
        "largest_scaffold_fraction": float(sizes.max() / len(frame)),
        "mean_scaffold_size": float(sizes.mean()),
        "median_scaffold_size": float(np.median(sizes)),
        "top_10_scaffold_sizes": [int(x) for x in sorted(sizes, reverse=True)[:10]],
    }


def compute_duplicate_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute duplicate statistics based on canonical SMILES."""

    if "smiles_canonical" not in frame.columns:
        return {"error": "missing_smiles_canonical"}

    valid = frame.loc[frame["smiles_canonical"].notna(), "smiles_canonical"]

    n_valid = int(valid.shape[0])
    n_unique = int(valid.nunique())
    n_duplicate_rows = int(n_valid - n_unique)

    duplicated_values = valid[valid.duplicated(keep=False)]
    duplicate_group_sizes = duplicated_values.value_counts().sort_values(ascending=False).tolist()

    return {
        "n_valid_rows": n_valid,
        "n_unique_canonical_smiles": n_unique,
        "n_duplicate_rows": n_duplicate_rows,
        "duplicate_fraction": float(n_duplicate_rows / n_valid) if n_valid else 0.0,
        "top_10_duplicate_group_sizes": [int(x) for x in duplicate_group_sizes[:10]],
    }


def grouped_random_split_frame(
    frame: pd.DataFrame,
    *,
    group_column: str,
    seed: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    groups = [
        indices.to_list() for _, indices in frame.groupby(group_column, sort=False).groups.items()
    ]

    rng.shuffle(groups)

    n_total = len(frame)
    n_train_target = int(frac_train * n_total)
    n_valid_target = int(frac_valid * n_total)

    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []

    for group in groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid_target:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return {
        "train": frame.loc[train_idx].reset_index(drop=True),
        "valid": frame.loc[valid_idx].reset_index(drop=True),
        "test": frame.loc[test_idx].reset_index(drop=True),
    }


def compute_split_overlap_stats(
    splits: dict[str, pd.DataFrame],
    *,
    key: str = "smiles_canonical",
) -> dict[str, Any]:
    """Compute overlap in molecule keys across train/valid/test."""

    key_sets: dict[str, set[str]] = {}

    for split_name, split_df in splits.items():
        if key not in split_df.columns:
            key_sets[split_name] = set()
            continue

        key_sets[split_name] = {str(x) for x in split_df[key].dropna().tolist()}

    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]

    out: dict[str, Any] = {}

    for left, right in pairs:
        overlap = key_sets.get(left, set()) & key_sets.get(right, set())
        out[f"{left}_{right}"] = {
            "n_overlap": int(len(overlap)),
            "examples": sorted(overlap)[:20],
        }

    return out


def collect_preparation_versions() -> dict[str, str | None]:
    """Collect relevant package versions for prepared dataset provenance."""

    versions: dict[str, str | None] = {}

    for package_name in ["deepchem", "rdkit", "selfies", "pandas", "numpy"]:
        try:
            module = __import__(package_name)

            versions[package_name] = getattr(module, "__version__", None)

        except Exception:
            versions[package_name] = None

    return versions


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
