"""
Shared utilities for APETokenizer interaction and dataset loading.
"""

import hashlib
import json
import shutil
import statistics
import re
from pathlib import Path
from typing import Any, Literal


import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from tqdm.auto import tqdm

from modernmolbert.ape_tokenizer import APETokenizer


SPECIAL_TOKENS: dict[str, str] = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

SELFIES_REPRESENTATION = "SELFIES"
SELFIES_TOKENIZER_FILENAME = "selfies_ape_tokenizer.json"
SELFIES_TOKENIZER_METADATA_FILENAME = "selfies_ape_tokenizer.metadata.json"
PUBCHEM10M_DATASET = "mikemayuare/PubChem10M_SMILES_SELFIES"
ZINC20_DATASET = "haydn-jones/ZINC20"
ZINC20_CHEMBL36_DATASET = "alessandronascimento/zinc20_chembl36"
# ZINC20_CHEMBL36_DATASET notes:
#   - SELFIES column is lowercase: "selfies"
#   - "id" column contains strings prefixed with "ZINC" or "CHEMBL"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def infer_selfies_column(dataset_name: str, selfies_column: str | None = None) -> str:
    if selfies_column is not None:
        return selfies_column
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


def filter_zinc20_chembl36_by_source(
    ds: IterableDataset,
    source: Literal["zinc", "chembl", "all"] = "all",
) -> IterableDataset:
    """Filter a ZINC20_CHEMBL36 streaming dataset by molecule source.

    Parameters
    ----------
    ds:
        Streaming dataset loaded from ZINC20_CHEMBL36_DATASET.
    source:
        ``"zinc"``   — keep only rows whose ``id`` starts with ``"ZINC"``.
        ``"chembl"`` — keep only rows whose ``id`` starts with ``"CHEMBL"``.
        ``"all"``    — no filtering; return the dataset unchanged.
    """
    if source == "all":
        return ds
    prefix = "ZINC" if source == "zinc" else "CHEMBL"
    # batched=True: filter is called once per batch (default 1000 rows) rather
    # than once per row, and np.char.startswith runs the string comparison in C.
    return ds.filter(
        lambda batch: [s.startswith(prefix) for s in batch["id"]],
        batched=True,
    )


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


def default_selfies_tokenizer_path() -> Path:
    return repo_root() / "tokenizer" / SELFIES_TOKENIZER_FILENAME


def metadata_path_for_vocab(vocab_path: Path) -> Path:
    return vocab_path.with_suffix(".metadata.json")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Keep stable artifact names regardless of original file names.
    shutil.copy2(vocab_path, output_dir / "tokenizer.json")
    shutil.copy2(metadata_path, output_dir / "tokenizer_metadata.json")
    shutil.copy2(vocab_path, final_model_dir / "tokenizer.json")
    shutil.copy2(metadata_path, final_model_dir / "tokenizer_metadata.json")


def assert_metadata_representation(metadata: dict[str, Any], expected_representation: str) -> None:
    representation = str(metadata.get("representation", "")).upper()
    if representation != expected_representation:
        raise ValueError(
            "Tokenizer metadata representation mismatch: "
            f"expected {expected_representation}, found {representation or '<missing>'}."
        )


def sample_jsonl_sequences(file_path: Path, representation: str, n: int) -> list[str]:
    rows: list[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            seq = normalize_sequence(record, representation)
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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def normalize_sequence(example: dict[str, Any], representation: str) -> str | None:
    seq = example.get(representation)
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
) -> IterableDataset:
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

    local = find_local_dataset(data_dir=data_dir, dataset_name=dataset_name)
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
    representation: str,
    n: int,
    seed: int,
    buffer_size: int,
    data_dir: Path | None = None,
    data_files: str | None = None,
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

    pbar = tqdm(total=n, desc=f"Collecting {representation} corpus for APE tokenizer")
    for row in ds:
        seq = normalize_sequence(row, representation)
        if seq is None:
            continue
        corpus.append(seq)
        pbar.update(1)
        if len(corpus) >= n:
            break
    pbar.close()

    if not corpus:
        raise RuntimeError("Tokenizer corpus is empty. Check dataset column names.")

    return corpus


# ---------------------------------------------------------------------------
# Tokenizer utilities
# ---------------------------------------------------------------------------


def tokenizer_vocab_size(tokenizer: APETokenizer) -> int:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict):
            return len(vocab)

    for attr in ["vocab", "vocabulary", "token_to_id", "token2id"]:
        if hasattr(tokenizer, attr):
            value = getattr(tokenizer, attr)
            if isinstance(value, dict):
                return len(value)

    raise AttributeError(
        "Could not infer APETokenizer vocabulary size. "
        "Inspect the tokenizer object and adjust tokenizer_vocab_size()."
    )


def token_id(tokenizer: APETokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        out = tokenizer.convert_tokens_to_ids([token])
        return int(out[0] if isinstance(out, list) else out)

    encoded = tokenizer(token, add_special_tokens=False)
    ids = encoded["input_ids"]

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} resolved to {ids}, expected one ID.")

    return int(ids[0])


def resolve_special_ids(tokenizer: APETokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for name, token in SPECIAL_TOKENS.items():
        try:
            ids[name] = token_id(tokenizer, token)
        except Exception as err:
            attr_name = name + "_id"
            if hasattr(tokenizer, attr_name):
                ids[name] = int(getattr(tokenizer, attr_name))
            else:
                raise RuntimeError(
                    f"Could not resolve ID for special token {token!r}. "
                    "Check APETokenizer special-token names."
                ) from err
    return ids


def encode_sequence(tokenizer: APETokenizer, seq: str, max_seq_length: int) -> dict[str, list[int]]:
    encoded = tokenizer(
        seq,
        padding=False,
        max_length=max_seq_length,
        add_special_tokens=True,
        return_tensors=None,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(attention_mask, torch.Tensor):
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


def compute_tokenization_stats(
    tokenizer: APETokenizer,
    sequences: list[str],
    max_seq_length: int,
    special_ids: dict[str, int],
) -> dict[str, float]:
    if not sequences:
        raise ValueError("Cannot compute tokenization stats on an empty sequence list.")

    unk_id = special_ids["unk_token"]

    lengths: list[int] = []
    truncations = 0
    unknown_tokens = 0
    eligible_tokens = 0
    empty_sequences = 0
    mostly_unknown = 0

    for seq in sequences:
        raw = tokenizer(seq, add_special_tokens=True, return_tensors=None)
        raw_ids = raw["input_ids"]
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.tolist()
        if raw_ids and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0]
        raw_ids = [int(x) for x in raw_ids]

        if not raw_ids:
            empty_sequences += 1
            continue

        if len(raw_ids) > max_seq_length:
            truncations += 1

        encoded = encode_sequence(tokenizer, seq, max_seq_length)["input_ids"]
        lengths.append(len(encoded))

        eligible = eligible_token_ids(encoded, special_ids)
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
    }
    return stats
