"""
Shared helpers for molecular tokenizers and dataset loading.
"""

import json
import shutil
import statistics
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Re-exported so existing callers can keep importing it from modernmolbert.utils.
from modernmolbert.hf_upload import file_sha256 as file_sha256

# transformers, datasets and tqdm are slow to import, so they are imported inside the
# functions that use them; the names below are only needed for type checking.
if TYPE_CHECKING:
    from datasets import IterableDataset
    from transformers import PreTrainedTokenizerBase


SPECIAL_TOKENS: dict[str, str] = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

# Special-token IDs the model config and inference pipeline depend on.
# resolve_special_ids returns the same keys, so the two compare directly.
EXPECTED_SPECIAL_IDS: dict[str, int] = {
    "bos_token": 0,
    "pad_token": 1,
    "eos_token": 2,
    "unk_token": 3,
    "mask_token": 4,
}


def assert_special_ids(special_ids: dict[str, int]) -> None:
    """Raise if special-token IDs are not the layout the pipeline expects."""
    if special_ids != EXPECTED_SPECIAL_IDS:
        raise ValueError(
            f"Unexpected special token IDs: {special_ids}; expected {EXPECTED_SPECIAL_IDS}. "
            "Model config and inference depend on these positions."
        )


SELFIES_REPRESENTATION = "SELFIES"
SMILES_REPRESENTATION = "SMILES"
TOKENIZER_METADATA_FILENAMES = (
    "tokenizer_metadata.json",
    "ape_tokenizer_metadata.json",
)
PUBCHEM10M_DATASET = "mikemayuare/PubChem10M_SMILES_SELFIES"
ZINC20_DATASET = "haydn-jones/ZINC20"
ZINC20_CHEMBL36_DATASET = "alessandronascimento/zinc20_chembl36"
# ZINC20_CHEMBL36_DATASET notes:
#   - SELFIES column is lowercase: "selfies"
#   - "id" column contains strings prefixed with "ZINC" or "CHEMBL"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_run_args(run_dir: Path) -> dict[str, Any]:
    """Read new run identity arguments, falling back to historical run_args.json."""
    identity_path = run_dir / "run_identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("schema") == 2:
            return dict(identity["args"])
    legacy_path = run_dir / "run_args.json"
    if legacy_path.is_file():
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    return {}


def _local_dataset_metadata(dataset_name: str) -> dict[str, Any]:
    """The metadata.json of a local dataset directory, or {} if there is none."""
    if not _looks_like_path(dataset_name):
        return {}
    local_path = _resolve_dataset_name_as_local_path(dataset_name)
    if local_path is None:
        return {}
    try:
        metadata = json.loads((local_path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def infer_selfies_column(dataset_name: str, selfies_column: str | None = None) -> str:
    if selfies_column is not None:
        return selfies_column

    metadata = _local_dataset_metadata(dataset_name)
    if "selfies_column" in metadata:
        return str(metadata["selfies_column"])
    if dataset_name == ZINC20_CHEMBL36_DATASET:
        return "selfies"  # lowercase in this dataset
    if dataset_name == ZINC20_DATASET:
        return "SELFIES"
    return SELFIES_REPRESENTATION


def infer_validation_split(dataset_name: str, validation_split: str | None = None) -> str | None:
    if validation_split is not None:
        return validation_split
    if dataset_name == ZINC20_DATASET:
        return "validation"
    return None


def _is_smiles(representation: str) -> bool:
    return representation.upper() == SMILES_REPRESENTATION


def infer_molecule_column(
    dataset_name: str, representation: str, molecule_column: str | None = None
) -> str:
    """Resolve the dataset column holding molecule strings for the representation.

    A local dataset names its columns in metadata.json. For SMILES this is the
    RDKit canonical column that its SELFIES were encoded from, so both
    representations describe the same molecules.
    """
    if molecule_column is not None:
        return molecule_column
    if _is_smiles(representation):
        metadata = _local_dataset_metadata(dataset_name)
        return str(metadata.get("canonical_smiles_column", "smiles"))
    return infer_selfies_column(dataset_name)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _local_dataset_matches_request(local_dir: Path, dataset_name: str) -> bool:
    info_path = local_dir / "dataset_info.json"
    if not info_path.exists():
        return False

    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return False

    if not isinstance(info, dict):
        return False

    requested = _normalized_name(dataset_name.split("/")[-1])
    if not requested:
        return False

    candidates: set[str] = {_normalized_name(local_dir.name)}
    for key in ["dataset_name", "config_name", "builder_name"]:
        value = info.get(key)
        if value:
            candidates.add(_normalized_name(str(value)))

    return any(requested in c or c in requested for c in candidates if c)


def _looks_like_path(value: str) -> bool:
    candidate = Path(value)
    return candidate.is_absolute() or "/" in value or "\\" in value or value.startswith(".")


def _is_local_dataset_dir(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    if (directory / "dataset_info.json").exists():
        return True
    if any(directory.glob("*.parquet")):
        return True
    return bool(any(directory.glob("**/*.parquet")))


def _resolve_dataset_name_as_local_path(dataset_name: str) -> Path | None:
    candidate = Path(dataset_name).expanduser()
    candidates: list[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((Path.cwd() / candidate).resolve())
        candidates.append((repo_root() / candidate).resolve())

    seen: set[Path] = set()
    for resolved in candidates:
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir() and _is_local_dataset_dir(resolved):
            return resolved
    return None


# Accepted on-disk filename stems for each requested split.
_SPLIT_ALIASES: dict[str, list[str]] = {
    "train": ["train"],
    "valid": ["valid", "validation", "val"],
    "validation": ["validation", "valid", "val"],
    "val": ["val", "validation", "valid"],
    "test": ["test"],
}


def _available_local_parquet_splits(directory: Path) -> set[str]:
    available: set[str] = set()
    for split_name, names in _SPLIT_ALIASES.items():
        if any((directory / f"{name}.parquet").exists() for name in names):
            available.add(split_name)
            continue
        if any(directory.glob(f"**/{split_name}.parquet")):
            available.add(split_name)
            continue
        if any(directory.glob(f"**/{split_name}-*.parquet")):
            available.add(split_name)
            continue
    return available


def _split_parquet_files(directory: Path, split: str) -> list[Path]:
    files: list[Path] = []
    for name in _SPLIT_ALIASES.get(split, [split]):
        files.extend(directory.glob(f"{name}.parquet"))
        files.extend(directory.glob(f"{name}-*.parquet"))

    if not files:
        for name in _SPLIT_ALIASES.get(split, [split]):
            files.extend(directory.glob(f"**/{name}.parquet"))
            files.extend(directory.glob(f"**/{name}-*.parquet"))

    return sorted(set(files))


def find_local_dataset(
    data_dir: Path | None = None,
    dataset_name: str | None = None,
) -> Path | None:
    """Return local Arrow dataset directory, or None to stream from HF.

    If *data_dir* is given, use it if it contains dataset_info.json.
    If omitted, scan repo_root()/data and return the first directory whose
    dataset metadata looks compatible with *dataset_name*.
    """
    if data_dir is not None:
        if not (data_dir / "dataset_info.json").exists():
            raise FileNotFoundError(f"Invalid data_dir: {data_dir}. Missing dataset_info.json.")
        return data_dir

    search_root = repo_root() / "data"
    if not search_root.exists():
        return None

    for candidate in sorted(search_root.iterdir()):
        if not candidate.is_dir() or not (candidate / "dataset_info.json").exists():
            continue
        if dataset_name is None or _local_dataset_matches_request(candidate, dataset_name):
            return candidate

    return None


def metadata_path_for_vocab(vocab_path: Path) -> Path:
    return vocab_path.with_suffix(".metadata.json")


def write_tokenizer_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_tokenizer_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Tokenizer metadata must be a JSON object: {metadata_path}")
    return data


def copy_tokenizer_artifacts(
    vocab_path: Path,
    metadata_path: Path,
    output_dir: Path,
    final_model_dir: Path,
    model_max_length: int | None = None,
) -> None:
    """Save the training tokenizer and its metadata with a trained model.

    A BPE tokenizer is a standard ``tokenizer.json`` at the model root. An APE
    tokenizer needs custom code; it is saved at the root and in
    ``ape_tokenizer/``, because Transformers only runs remote tokenizer code
    from a directory without the ModernBERT model config.
    """
    from modernmolbert.tokenization.load import BPE, SMIRK, load_tokenizer, tokenizer_algorithm

    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_tokenizer_metadata(metadata_path)
    tokenizer = load_tokenizer(vocab_path, metadata, model_max_length=model_max_length)
    if tokenizer_algorithm(metadata) in {BPE, SMIRK}:
        tokenizer.save_pretrained(str(final_model_dir))
        for target_dir in [output_dir, final_model_dir]:
            shutil.copy2(metadata_path, target_dir / "tokenizer_metadata.json")
        return

    representation = str(metadata["representation"]).upper()
    tokenizer.save_vocabulary(str(output_dir))
    tokenizer.save_vocabulary(str(final_model_dir))
    tokenizer.save_pretrained(str(output_dir / "ape_tokenizer"))
    tokenizer.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir / "ape_tokenizer"))

    alias_name = f"{representation.lower()}_vocab.json"
    for tokenizer_dir in [final_model_dir, final_model_dir / "ape_tokenizer"]:
        active_vocab = tokenizer_dir / "vocab.json"
        if active_vocab.exists():
            shutil.copy2(active_vocab, tokenizer_dir / alias_name)

    # Preserve legacy metadata path while standardizing metadata aliases.
    if metadata_path.exists():
        shutil.copy2(metadata_path, output_dir / "tokenizer_metadata.json")
        shutil.copy2(metadata_path, final_model_dir / "tokenizer_metadata.json")

    source_dirs = [
        metadata_path.parent,
        output_dir,
        output_dir / "ape_tokenizer",
        final_model_dir,
        final_model_dir / "ape_tokenizer",
    ]

    for target_dir in [
        final_model_dir,
        final_model_dir / "ape_tokenizer",
        output_dir,
        output_dir / "ape_tokenizer",
    ]:
        copy_tokenizer_metadata_from_anywhere(source_dirs=source_dirs, target_dir=target_dir)


def copy_tokenizer_metadata_from_anywhere(source_dirs: list[Path], target_dir: Path) -> None:
    """Copy tokenizer metadata aliases into target_dir if available in source_dirs."""
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False

    for filename in TOKENIZER_METADATA_FILENAMES:
        for source_dir in source_dirs:
            source = source_dir / filename
            if source.exists():
                target = target_dir / filename
                if source.resolve() == target.resolve():
                    copied_any = True
                    break
                shutil.copy2(source, target)
                copied_any = True
                break

    if not copied_any:
        print(
            "WARNING: no tokenizer metadata file found in any source directory "
            f"for target {target_dir}"
        )


def assert_metadata_representation(metadata: dict[str, Any], expected_representation: str) -> None:
    representation = str(metadata.get("representation", "")).upper()
    if representation != expected_representation:
        raise ValueError(
            "Tokenizer metadata representation mismatch: "
            f"expected {expected_representation}, found {representation or '<missing>'}."
        )


def sample_jsonl_sequences(file_path: Path, column: str, n: int) -> list[str]:
    rows: list[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            seq = normalize_sequence(record, column)
            if seq is None:
                continue
            rows.append(seq)
            if len(rows) >= n:
                break
    return rows


def validate_selfies_sample_shape(sequences: list[str]) -> None:
    if not sequences:
        raise ValueError("No sequences available for SELFIES validation.")

    bracketed = 0
    for seq in sequences:
        # Heuristic SELFIES guard: bracketed tokens should dominate.
        if "[" in seq and "]" in seq:
            bracketed += 1

    if bracketed / len(sequences) < 0.95:
        raise ValueError(
            "Sampled values do not look like SELFIES strings (insufficient bracketed tokens)."
        )


def validate_smiles_sample_shape(sequences: list[str]) -> None:
    if not sequences:
        raise ValueError("SMILES corpus is empty.")
    empty = sum(1 for s in sequences if not s)
    if empty / len(sequences) > 0.05:
        raise ValueError("Sampled values do not look like SMILES strings (too many empty).")


def validate_sample_shape(sequences: list[str], representation: str) -> None:
    """Sanity-check that sampled sequences look like the expected representation."""
    if _is_smiles(representation):
        validate_smiles_sample_shape(sequences)
    else:
        validate_selfies_sample_shape(sequences)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def normalize_sequence(example: dict[str, Any], column: str) -> str | None:
    seq = example.get(column)
    if seq is None:
        return None
    seq = str(seq).strip()
    return seq if seq else None


def get_streaming_dataset(
    dataset_name: str,
    seed: int,
    buffer_size: int,
    split: str = "train",
    data_dir: Path | None = None,
    data_files: str | None = None,
) -> "IterableDataset":
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    if data_files is not None:
        print(
            f"[data] Streaming parquet files directly for split '{split}': {data_files}",
            flush=True,
        )
        hf_ds = load_dataset(
            "parquet",
            data_files={split: data_files},
            split=split,
            streaming=True,
        )
        return hf_ds.shuffle(seed=seed, buffer_size=buffer_size)

    explicit_path = (
        _resolve_dataset_name_as_local_path(dataset_name)
        if _looks_like_path(dataset_name)
        else None
    )
    if data_dir is None and explicit_path is not None and any(explicit_path.glob("**/*.parquet")):
        files = _split_parquet_files(explicit_path, split)
        if not files:
            available = (
                ", ".join(sorted(_available_local_parquet_splits(explicit_path))) or "<none>"
            )
            raise ValueError(
                f"Local parquet dataset at {explicit_path} has no split '{split}'. "
                f"Available splits: {available}"
            )
        print(f"[data] Loading local parquet split '{split}': {explicit_path}", flush=True)
        hf_ds = load_dataset(
            "parquet",
            data_files={split: [str(f) for f in files]},
            split=split,
            streaming=True,
        )
        return hf_ds.shuffle(seed=seed, buffer_size=buffer_size)

    local = (
        find_local_dataset(data_dir=explicit_path)
        if data_dir is None and explicit_path is not None
        else find_local_dataset(data_dir=data_dir, dataset_name=dataset_name)
    )
    if local is not None:
        print(f"[data] Loading dataset from disk: {local}", flush=True)
        raw = load_from_disk(str(local))
        if isinstance(raw, DatasetDict):
            if split not in raw:
                available = ", ".join(sorted(str(k) for k in raw))
                raise ValueError(
                    f"Local dataset at {local} has no split '{split}'. "
                    f"Available splits: {available}"
                )
            return raw[split].shuffle(seed=seed).to_iterable_dataset()
        if isinstance(raw, Dataset):
            if split != "train":
                raise ValueError(
                    f"Requested split '{split}' but local dataset at {local} "
                    "is a single train-only Dataset. "
                    "Either disable --use_validation_split or save a DatasetDict with splits."
                )
            return raw.shuffle(seed=seed).to_iterable_dataset()
        raise ValueError(f"Unsupported local dataset type at {local}: {type(raw).__name__}")

    print(f"[data] Streaming dataset from HF Hub: {dataset_name} [{split}]", flush=True)
    try:
        hf_ds = load_dataset(dataset_name, split=split, streaming=True)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            raise RuntimeError(
                f"Dataset {dataset_name!r} uses a legacy Hugging Face dataset script, "
                "which is not supported by the installed `datasets` version. "
                "Use a script-free Parquet/Arrow mirror, load data files directly with "
                "`load_dataset('parquet', data_files=...)`, or pin `datasets<4` in a "
                "separate data-preparation environment."
            ) from e
        raise
    return hf_ds.shuffle(seed=seed, buffer_size=buffer_size)


def collect_corpus_for_tokenizer(
    dataset_name: str,
    column: str,
    n: int,
    seed: int,
    buffer_size: int,
    data_dir: Path | None = None,
    data_files: str | None = None,
    show_progress: bool = False,
) -> list[str]:
    ds = get_streaming_dataset(
        dataset_name,
        split="train",
        seed=seed,
        buffer_size=buffer_size,
        data_dir=data_dir,
        data_files=data_files,
    )
    corpus: list[str] = []

    from tqdm.auto import tqdm

    print(f"[corpus] Collecting {n:,} sequences from column {column!r}...", flush=True)
    milestones = {int(n * p) for p in (0.25, 0.50, 0.75)}
    pbar = tqdm(
        total=n,
        desc="Collecting tokenizer corpus",
        disable=not show_progress,
    )
    for row in ds:
        if column not in row:
            raise ValueError(f"Tokenizer corpus does not contain column {column!r}")
        seq = normalize_sequence(row, column)
        if seq is None:
            continue
        corpus.append(seq)
        pbar.update(1)
        if len(corpus) in milestones:
            print(
                f"[corpus] {len(corpus):,}/{n:,} sequences collected ({len(corpus) * 100 // n}%)",
                flush=True,
            )
        if len(corpus) >= n:
            break
    pbar.close()
    print(f"[corpus] Done: {len(corpus):,} sequences collected.", flush=True)

    if not corpus:
        raise RuntimeError("Tokenizer corpus is empty. Check dataset column names.")

    return corpus


# ---------------------------------------------------------------------------
# Tokenizer utilities
# ---------------------------------------------------------------------------


def tokenizer_vocab_size(tokenizer: "PreTrainedTokenizerBase") -> int:
    return len(tokenizer.get_vocab())


def resolve_special_ids(tokenizer: "PreTrainedTokenizerBase") -> dict[str, int]:
    """Each special token's ID in the tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    tokens: dict[str, str] = {}
    missing: list[str] = []
    for name in SPECIAL_TOKENS:
        token = getattr(tokenizer, name, None)
        if not isinstance(token, str) or token not in vocab:
            missing.append(name)
        else:
            tokens[name] = token
    if missing:
        raise ValueError(f"Tokenizer vocabulary lacks special tokens: {missing}")
    return {name: int(vocab[token]) for name, token in tokens.items()}


def encode_sequence(
    tokenizer: "PreTrainedTokenizerBase",
    seq: str,
    max_seq_length: int | None,
) -> dict[str, list[int]]:
    encoded = tokenizer(
        seq,
        padding=False,
        truncation=max_seq_length is not None,
        max_length=max_seq_length,
        add_special_tokens=True,
        return_tensors=None,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()

    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]

    return {
        "input_ids": list(map(int, input_ids)),
        "attention_mask": list(map(int, attention_mask)),
    }


def ignored_special_token_ids(special_ids: dict[str, int]) -> set[int]:
    """Special token IDs ignored for tokenization statistics.

    Important: do NOT ignore unk_token. Unknown tokens must remain in the
    denominator when computing unk_rate.
    """
    return {
        special_ids["pad_token"],
        special_ids["bos_token"],
        special_ids["eos_token"],
        special_ids["mask_token"],
    }


def eligible_token_ids(input_ids: list[int], special_ids: dict[str, int]) -> list[int]:
    excluded = ignored_special_token_ids(special_ids)
    return [tok for tok in input_ids if tok not in excluded]


def assert_representation_compatible(
    tokenizer: "PreTrainedTokenizerBase",
    special_ids: dict[str, int],
    representation: str,
    max_seq_length: int | None = 256,
) -> None:
    """Fail fast if the tokenizer cannot represent a trivial molecule (ethanol).

    Catches a wrong or mismatched vocabulary before any expensive work: ethanol
    must tokenize to mostly-known tokens for the given representation.
    """
    ethanol = "CCO" if _is_smiles(representation) else "[C][C][O]"
    encoded = encode_sequence(tokenizer, ethanol, max_seq_length)["input_ids"]
    eligible = eligible_token_ids(encoded, special_ids)
    if not eligible:
        raise ValueError(f"Tokenizer produced no usable tokens for ethanol ({ethanol}).")

    unk_rate = sum(1 for tok in eligible if tok == special_ids["unk_token"]) / len(eligible)
    if unk_rate > 0.05:
        raise ValueError(
            f"Tokenizer is not {representation}-compatible: {ethanol} "
            f"unk_rate={unk_rate:.3f}, ids={encoded}"
        )


def compute_tokenization_stats(
    tokenizer: "PreTrainedTokenizerBase",
    sequences: list[str],
    max_seq_length: int,
    special_ids: dict[str, int],
) -> dict[str, float]:
    """Length, truncation and coverage statistics over ``sequences``.

    ``silent_loss_rate`` is the fraction of sequences whose tokens, joined, do not
    reproduce the input although no token is ``<unk>``: content was dropped or
    altered without trace.
    """
    if not sequences:
        raise ValueError("Cannot compute tokenization stats on an empty sequence list.")

    unk_id = special_ids["unk_token"]
    unk_token = tokenizer.unk_token

    lengths: list[int] = []
    truncations = 0
    unknown_tokens = 0
    eligible_tokens = 0
    empty_sequences = 0
    mostly_unknown = 0
    silent_losses = 0

    for seq in sequences:
        tokens = tokenizer.tokenize(seq)
        if unk_token not in tokens and "".join(tokens) != seq:
            silent_losses += 1

        raw = tokenizer(seq, add_special_tokens=True, return_tensors=None)
        raw_ids = raw["input_ids"]
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        if raw_ids and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0]
        raw_ids = [int(x) for x in raw_ids]

        if not raw_ids:
            empty_sequences += 1
            continue

        if len(raw_ids) > max_seq_length:
            truncations += 1

        lengths.append(len(raw_ids))

        eligible = eligible_token_ids(raw_ids, special_ids)
        if eligible:
            unk_count = sum(1 for tok in eligible if tok == unk_id)
            unknown_tokens += unk_count
            eligible_tokens += len(eligible)
            if unk_count / len(eligible) > 0.8:
                mostly_unknown += 1

    if not lengths:
        raise ValueError("All sampled sequences tokenized to empty outputs.")

    def pct(values: list[int], q: float) -> float:
        ordered = sorted(values)
        idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])

    stats: dict[str, float] = {
        "sample_size": float(len(sequences)),
        "mean_len": float(statistics.fmean(lengths)),
        "p50_len": pct(lengths, 0.50),
        "p95_len": pct(lengths, 0.95),
        "p99_len": pct(lengths, 0.99),
        "max_len": float(max(lengths)),
        "truncation_rate": float(truncations / len(sequences)),
        "unk_rate": float(unknown_tokens / max(1, eligible_tokens)),
        "empty_sequence_rate": float(empty_sequences / len(sequences)),
        "mostly_unknown_rate": float(mostly_unknown / len(sequences)),
        "silent_loss_rate": float(silent_losses / len(sequences)),
    }
    return stats


def get_git_revision() -> dict[str, object]:
    """Return the current Git commit hash and dirty status, or None values if unavailable."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            return {"commit": commit if commit else None, "dirty": bool(status.stdout.strip())}
    except Exception:
        pass
    return {"commit": None, "dirty": None}
