import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()


@dataclass(frozen=True)
class ChemBL36SelfiesPrepConfig:
    dataset_name: str = "lukaskim/ChEMBL-36"
    dataset_config: str = "molecules"
    split: str = "train"
    smiles_column: str = "canonical_smiles"
    output_dir: Path = Path("data/pretrain/chembl36_selfies")
    seed: int = 13
    valid_fraction: float = 0.01
    test_fraction: float = 0.0
    max_rows: int | None = None
    dedupe_column: str = "standard_inchi_key"
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 100
    max_mw: float = 1000.0
    chunk_size: int = 100_000


def prepare_chembl36_selfies(config: ChemBL36SelfiesPrepConfig) -> None:
    """Load ChEMBL36, prepare valid SELFIES rows, and write split Parquet files."""

    from datasets import load_dataset

    config.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.split,
    )

    if config.max_rows is not None:
        if config.max_rows < 1:
            raise ValueError("max_rows must be positive when provided")
        ds = ds.select(range(min(config.max_rows, len(ds))))

    source_rows = int(len(ds))
    raw = ds.to_pandas()
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from to_pandas(), got {type(raw).__name__}")
    frame = raw
    prepared, stats = prepare_chembl36_frame(frame, config=config, return_stats=True)

    train, valid, test = split_by_hash(
        prepared,
        key_column="split_key",
        valid_fraction=config.valid_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )

    splits: dict[str, pd.DataFrame] = {
        "train": train,
        "valid": valid,
    }

    if test is not None:
        splits["test"] = test

    for split_name, split_frame in splits.items():
        split_frame.to_parquet(config.output_dir / f"{split_name}.parquet", index=False)

    write_example_tsv(splits=splits, output_path=config.output_dir / "example.tsv")

    metadata = {
        "name": "chembl36_selfies",
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "source_split": config.split,
        "representation": "SELFIES",
        "smiles_column": config.smiles_column,
        "selfies_column": "selfies",
        "canonical_smiles_column": "smiles_canonical_clean",
        "config": _jsonable_config(config),
        "source_row_count": source_rows,
        "preparation_stats": stats,
        "row_counts": {
            "prepared_total": int(len(prepared)),
            "train": int(len(train)),
            "valid": int(len(valid)),
            **({"test": int(len(test))} if test is not None else {}),
        },
        "columns": list(prepared.columns),
        "split_overlap": compute_split_overlap_stats(
            splits,
            key="smiles_canonical_clean",
        ),
        "split_policy": {
            "method": "deterministic_hash",
            "key_column": "split_key",
            "seed": config.seed,
            "valid_fraction": config.valid_fraction,
            "test_fraction": config.test_fraction,
            "has_test": test is not None,
        },
        "versions": collect_preparation_versions(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "creation_command": "python -m modernmolbert.data.prepare_chembl36_selfies",
    }

    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


@overload
def prepare_chembl36_frame(
    frame: pd.DataFrame,
    *,
    config: ChemBL36SelfiesPrepConfig,
    return_stats: Literal[True],
) -> tuple[pd.DataFrame, dict[str, Any]]: ...


@overload
def prepare_chembl36_frame(
    frame: pd.DataFrame,
    *,
    config: ChemBL36SelfiesPrepConfig,
    return_stats: Literal[False] = ...,
) -> pd.DataFrame: ...


def prepare_chembl36_frame(
    frame: pd.DataFrame,
    *,
    config: ChemBL36SelfiesPrepConfig,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare a ChEMBL-like frame for SELFIES pretraining.

    This function is intentionally pure apart from progress reporting so it can
    be unit tested without downloading the source dataset.
    """

    required = {config.smiles_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    stats: dict[str, Any] = {
        "input_rows": int(len(frame)),
        "dedupe_column": None,
        "rows_after_dedupe": 0,
        "rows_valid_after_conversion": 0,
        "rows_after_filters": 0,
        "dropped_invalid_or_filtered": 0,
        "sanitize_error_counts": {},
    }

    frame = frame.copy()
    dedupe_column = config.dedupe_column
    if dedupe_column in frame.columns:
        frame = frame.drop_duplicates(dedupe_column).reset_index(drop=True)
        stats["dedupe_column"] = dedupe_column
    else:
        frame = frame.drop_duplicates(config.smiles_column).reset_index(drop=True)
        stats["dedupe_column"] = config.smiles_column

    stats["rows_after_dedupe"] = int(len(frame))

    checkpoint_dir = config.output_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n = len(frame)
    chunk_size = config.chunk_size
    chunk_paths: list[Path] = []
    conversion_valid_rows = 0

    with tqdm(total=n, desc="Preparing ChEMBL36 SELFIES") as pbar:
        for chunk_idx, chunk_start in enumerate(range(0, n, chunk_size)):
            chunk = frame.iloc[chunk_start : chunk_start + chunk_size]
            checkpoint_path = checkpoint_dir / f"chunk_{chunk_idx:05d}.parquet"
            chunk_paths.append(checkpoint_path)

            if checkpoint_path.exists():
                summary = pd.read_parquet(checkpoint_path, columns=["is_valid"])
                conversion_valid_rows += int(summary["is_valid"].sum())
                pbar.update(len(chunk))
                continue

            rows: list[dict[str, Any]] = []
            for _, row in chunk.iterrows():
                out = dict(row)
                try:
                    converted = canonicalize_and_selfies(row.get(config.smiles_column))
                except BaseException:
                    continue
                out.update(converted)
                if converted["is_valid"]:
                    conversion_valid_rows += 1
                if out["is_valid"] and not passes_basic_filters(out, config=config):
                    out["is_valid"] = False
                    out["sanitize_error"] = "failed_basic_filters"
                rows.append(out)

            chunk_df = pd.DataFrame(rows)
            tmp_path = checkpoint_path.with_suffix(".tmp")
            chunk_df.to_parquet(tmp_path, index=False)
            tmp_path.rename(checkpoint_path)
            del rows, chunk_df
            pbar.update(len(chunk))

    # Collect valid rows and compute stats from checkpoints
    error_counts: dict[str, int] = {}
    valid_frames: list[pd.DataFrame] = []
    for path in chunk_paths:
        df = pd.read_parquet(path)
        for err, cnt in df["sanitize_error"].fillna("valid").value_counts().items():
            error_counts[str(err)] = error_counts.get(str(err), 0) + int(cnt)
        valid_frames.append(df.loc[df["is_valid"]].copy())
        del df

    stats["rows_valid_after_conversion"] = int(conversion_valid_rows)
    stats["sanitize_error_counts"] = dict(sorted(error_counts.items()))

    if not valid_frames or all(f.empty for f in valid_frames):
        raise ValueError("No valid ChEMBL36 rows remained after conversion and filtering")

    prepared = pd.concat(valid_frames, ignore_index=True)
    del valid_frames

    if prepared.empty:
        raise ValueError("No valid ChEMBL36 rows remained after conversion and filtering")

    prepared["split_key"] = prepared.apply(make_split_key, axis=1)
    prepared = prepared.drop_duplicates("split_key").reset_index(drop=True)

    stats["rows_after_filters"] = int(len(prepared))
    stats["dropped_invalid_or_filtered"] = int(stats["rows_after_dedupe"] - len(prepared))

    if return_stats:
        return prepared, stats
    return prepared


def canonicalize_and_selfies(smiles: Any) -> dict[str, Any]:
    """Canonicalize a SMILES value with RDKit and encode it as SELFIES."""

    from rdkit import Chem
    import selfies as sf

    if smiles is None or pd.isna(smiles):
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": "missing_smiles",
        }

    text = str(smiles).strip()
    if not text:
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": "empty_smiles",
        }

    try:
        mol = Chem.MolFromSmiles(text, sanitize=False)
    except BaseException as exc:
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": f"rdkit_exception:{type(exc).__name__}",
        }

    if mol is None:
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": "rdkit_parse_failed",
        }

    # SANITIZE_KEKULIZE and SANITIZE_PROPERTIES trigger C-level segfaults in this
    # RDKit build for exotic molecules (non-kekulizable aromatics, unusual valences).
    # Use only the flags MolToSmiles needs: ring perception + aromaticity perception.
    # Molecules that are chemically invalid get dropped by the SELFIES encoder.
    _san_ops = Chem.SanitizeFlags.SANITIZE_SYMMRINGS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    try:
        san_result = Chem.SanitizeMol(mol, _san_ops, catchErrors=True)
    except BaseException as exc:
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": f"sanitize_exception:{type(exc).__name__}",
        }

    if san_result:
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": f"sanitize_failed:{san_result.name}",
        }

    try:
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except BaseException as exc:
        return {
            "smiles_canonical_clean": None,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": f"canonicalization_failed:{type(exc).__name__}",
        }

    try:
        selfies = sf.encoder(canonical)
    except BaseException as exc:
        return {
            "smiles_canonical_clean": canonical,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": f"selfies_failed:{type(exc).__name__}",
        }

    if not selfies:
        return {
            "smiles_canonical_clean": canonical,
            "selfies": None,
            "is_valid": False,
            "sanitize_error": "selfies_empty",
        }

    return {
        "smiles_canonical_clean": canonical,
        "selfies": selfies,
        "is_valid": True,
        "sanitize_error": None,
    }


def passes_basic_filters(
    row: dict[str, Any],
    *,
    config: ChemBL36SelfiesPrepConfig,
) -> bool:
    """Apply light ChEMBL pretraining filters to already valid molecules."""

    heavy_atoms = row.get("heavy_atoms")
    if pd.notna(heavy_atoms):
        heavy_atoms = float(heavy_atoms)
        if heavy_atoms < config.min_heavy_atoms:
            return False
        if heavy_atoms > config.max_heavy_atoms:
            return False

    mw = row.get("mw_freebase")
    if pd.notna(mw) and float(mw) > config.max_mw:
        return False

    molecule_type = row.get("molecule_type")
    return not (pd.notna(molecule_type) and str(molecule_type).strip().lower() != "small molecule")


def make_split_key(row: pd.Series) -> str:
    for col in ["standard_inchi_key", "chembl_id", "smiles_canonical_clean"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value)

    return str(row["selfies"])


def hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest, 16) / float(16 ** len(digest))


def split_by_hash(
    frame: pd.DataFrame,
    *,
    key_column: str,
    valid_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split rows deterministically from a stable key, independent of row order.

    By default this returns train/valid only. A test split is created only when
    test_fraction > 0. For MLM pretraining, the validation split is used for
    monitoring and checkpoint selection; downstream benchmark test splits are
    used for final model evaluation.
    """

    if key_column not in frame.columns:
        raise ValueError(f"Input frame is missing split key column {key_column!r}")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1, inclusive of 0")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid_fraction + test_fraction must be < 1")

    values = frame[key_column].map(lambda x: hash_to_unit_interval(f"{seed}:{x}"))

    if test_fraction > 0.0:
        test_mask = values < test_fraction
        valid_mask = (values >= test_fraction) & (values < test_fraction + valid_fraction)
        train_mask = ~(test_mask | valid_mask)

        train = frame.loc[train_mask].reset_index(drop=True)
        valid = frame.loc[valid_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)

        if len(train) == 0 or len(valid) == 0 or len(test) == 0:
            raise RuntimeError(
                "Hash split produced an empty split. Increase dataset size or fractions."
            )

        return train, valid, test

    valid_mask = values < valid_fraction
    train_mask = ~valid_mask

    train = frame.loc[train_mask].reset_index(drop=True)
    valid = frame.loc[valid_mask].reset_index(drop=True)

    if len(train) == 0 or len(valid) == 0:
        raise RuntimeError(
            "Hash split produced an empty train or valid split. "
            "Increase dataset size or valid_fraction."
        )

    return train, valid, None


def write_example_tsv(
    *,
    splits: dict[str, pd.DataFrame],
    output_path: Path,
    n: int = 100,
) -> None:
    frames: list[pd.DataFrame] = []

    for split_name in ["train", "valid", "test"]:
        if split_name not in splits:
            continue
        frame = splits[split_name].head(n).copy()
        frame.insert(0, "split", split_name)
        frames.append(frame)

    example = pd.concat(frames, ignore_index=True).head(n)
    example.to_csv(output_path, sep="\t", index=False)


def compute_split_overlap_stats(
    splits: dict[str, pd.DataFrame],
    *,
    key: str,
) -> dict[str, Any]:
    key_sets: dict[str, set[str]] = {}
    for split_name, frame in splits.items():
        if key not in frame.columns:
            key_sets[split_name] = set()
            continue
        key_sets[split_name] = {str(value) for value in frame[key].dropna().tolist()}

    out: dict[str, Any] = {}
    split_names = list(splits)

    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = key_sets.get(left, set()) & key_sets.get(right, set())
            out[f"{left}_{right}"] = {
                "n_overlap": int(len(overlap)),
                "examples": sorted(overlap)[:20],
            }

    return out


def collect_preparation_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in ["datasets", "rdkit", "selfies", "pandas", "pyarrow"]:
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", None)
        except Exception:
            versions[package_name] = None
    return versions


def _jsonable_config(config: ChemBL36SelfiesPrepConfig) -> dict[str, Any]:
    out = asdict(config)
    out["output_dir"] = str(config.output_dir)
    return out
