#!/usr/bin/env python3
"""Train a ModernBERT masked-language model on SELFIES or SMILES molecular strings.

Model training requires an existing, vetted tokenizer file and metadata. The
metadata fixes the tokenizer algorithm (APE or BPE) and the representation the
model reads. Tokenizer training is intentionally a separate command:

    python -m modernmolbert.train_tokenizer
"""

import argparse
import hashlib
import time
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from modernmolbert.hf_upload import resolve_hf_token

import numpy as np
import torch
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from modernmolbert.collator import MolecularMLMCollator
from modernmolbert.tokenization.load import (
    APE,
    load_verified_tokenizer,
    tokenizer_algorithm,
    tokenizer_representation,
)
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    _resolve_dataset_name_as_local_path,
    assert_representation_compatible,
    assert_special_ids,
    compute_tokenization_stats,
    copy_tokenizer_artifacts,
    eligible_token_ids,
    encode_sequence,
    file_sha256,
    find_local_dataset,
    get_streaming_dataset,
    get_git_revision,
    infer_molecule_column,
    infer_validation_split,
    load_tokenizer_metadata,
    normalize_sequence,
    resolve_special_ids,
    tokenizer_vocab_size,
    validate_sample_shape,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DATASET_NAME = PUBCHEM10M_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a molecular ModernBERT MLM with a vetted APE or BPE tokenizer.",
    )

    # Paths
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer_vocab_path",
        type=str,
        required=True,
        help="Tokenizer file: an APE vocabulary JSON or a BPE tokenizer.json.",
    )
    parser.add_argument(
        "--tokenizer_metadata_path",
        type=str,
        default=None,
        help="Tokenizer metadata JSON. Defaults to <file>.metadata.json.",
    )
    parser.add_argument(
        "--require_corpus_only_vocab",
        action="store_true",
        help="Require a tokenizer scanned over the training corpus without injected symbols.",
    )

    # Dataset
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--molecule_column",
        "--selfies_column",
        dest="molecule_column",
        type=str,
        default=None,
        help=(
            "Column containing molecule strings in the tokenizer's representation. "
            "Defaults by dataset. --selfies_column is the former name."
        ),
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split used for training.",
    )
    parser.add_argument(
        "--validation_split",
        type=str,
        default=None,
        help="Dataset split used for validation when --use_validation_split is set.",
    )
    parser.add_argument(
        "--use_validation_split",
        action="store_true",
        help="Use dataset validation split for eval instead of hash-bucketing train split.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "Local Arrow dataset directory (e.g. data/pubchem10m_selfies). "
            "If omitted, auto-detect a matching dataset under data/."
        ),
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help=(
            "Optional parquet file path or glob to stream directly. "
            "When set, this takes precedence over --dataset_name/--data_dir."
        ),
    )
    parser.add_argument("--eval_size", type=int, default=100_000)
    parser.add_argument(
        "--validation_row_ids_path",
        type=Path,
        default=None,
        help="Persisted source-row IDs for the common finite validation cohort.",
    )
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument(
        "--global_train_shuffle",
        action="store_true",
        help=(
            "Load the exact local train Parquet as an Arrow dataset, shuffle all rows "
            "before streaming. Avoids source-order bias."
        ),
    )
    parser.add_argument(
        "--train_order_path",
        type=Path,
        default=None,
        help="Persisted source-row permutation shared by every factorial run.",
    )
    parser.add_argument("--seed", type=int, default=13)

    # Deterministic non-overlapping split by molecule identity.
    parser.add_argument("--val_split_mod", type=int, default=100)
    parser.add_argument("--val_split_bucket", type=int, default=0)

    # Tokenization gates
    parser.add_argument("--tokenizer_validation_samples", type=int, default=1000)
    parser.add_argument("--unk_rate_threshold", type=float, default=0.001)
    parser.add_argument("--truncation_warn_threshold", type=float, default=0.05)

    # Model
    parser.add_argument(
        "--model_size",
        choices=["small", "base"],
        default="small",
        help="ModernBERT architecture preset. 'small' ~30M/512-dim; 'base' ~90M/768-dim.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override max sequence length (default: use official model context length).",
    )

    # MLM
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.30,
        help=(
            "Fraction of eligible tokens to mask. For span/hetero_span strategies the "
            "budget is round(n_eligible × mlm_probability); short sequences may exceed "
            "this rate when a single span covers the full budget in one draw."
        ),
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        choices=["standard", "span", "hetero_span"],
        default="standard",
        help=(
            "MLM masking strategy. "
            "'standard': independent Bernoulli per token (original). "
            "'span': budget-based contiguous APE-token span masking. "
            "'hetero_span': span masking with span-start positions weighted toward "
            "APE tokens that contain heteroatoms (N, O, S, P, F, Cl, Br, I, Se, Si); "
            "APE SELFIES tokenizers only."
        ),
    )
    parser.add_argument(
        "--span_p_geom",
        type=float,
        default=0.4,
        help=(
            "Success probability for the geometric distribution used to sample span lengths. "
            "The unclamped mean span length is approximately 1/span_p_geom. "
            "With the default p=0.4 and span_max_length=6, the realized mean is about "
            "2.4 APE tokens after clamping. Only used when --masking_strategy is "
            "'span' or 'hetero_span'."
        ),
    )
    parser.add_argument(
        "--span_max_length",
        type=int,
        default=6,
        help=(
            "Maximum span length in APE tokens. Individual sampled lengths are clamped to "
            "this value. Adjacent independent spans can form longer contiguous masked runs — "
            "this parameter bounds individual draws, not total run length. "
            "Only used when --masking_strategy is 'span' or 'hetero_span'."
        ),
    )
    parser.add_argument(
        "--heteroatom_start_weight",
        type=float,
        default=2.0,
        help=(
            "Sampling weight multiplier for span-start positions whose APE token contains "
            "a heteroatom bracket (N, O, S, P, F, Cl, Br, I, Se, Si). "
            "Non-heteroatom-containing positions receive weight 1.0. "
            "Only used when --masking_strategy is 'hetero_span'."
        ),
    )

    # Training
    parser.add_argument("--max_steps", type=int, default=150_000)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Resume this run from one of its complete checkpoint-* directories.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the checkpoint with the best eval metric at the end of training.",
    )

    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help="Metric used to choose the best checkpoint.",
    )

    parser.add_argument(
        "--greater_is_better",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether a larger best-model metric is better.",
    )

    # Runtime
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--device_backend", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 mixed precision when supported.",
    )
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help=(
            "Maximum number of eval batches to materialize. "
            "Use 0 for no cap, so --eval_size controls validation size."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        choices=["none", "tensorboard"],
        default="none",
    )
    parser.add_argument(
        "--compute_masked_accuracy",
        action="store_true",
        help="Compute masked-token accuracy during eval.",
    )
    parser.add_argument("--debug", action="store_true", help="Run a tiny smoke test.")
    parser.add_argument(
        "--campaign_manifest",
        type=Path,
        default=None,
        help="Shared staged campaign manifest; replaces repeated per-run input hashes.",
    )
    parser.add_argument(
        "--require_clean_git",
        action="store_true",
        help="Reject a dirty or unresolved Git revision and pin it for resumption.",
    )
    parser.add_argument(
        "--hf_login",
        action="store_true",
        help="Log in to Hugging Face Hub using HF_TOKEN_ORG or HF_TOKEN before loading datasets/models.",
    )

    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "resume_from_checkpoint"
    }


def _run_input_hashes(
    args: argparse.Namespace, vocab_path: Path, metadata_path: Path
) -> dict[str, str]:
    paths = {
        "tokenizer": vocab_path,
        "tokenizer_metadata": metadata_path,
        "modernbert_base_config": _MODERNBERT_BASE_CONFIG,
        "uv_lock": Path(__file__).resolve().parents[2] / "uv.lock",
    }
    root = Path(args.dataset_name)
    if args.data_files is not None:
        paths["train_parquet"] = Path(args.data_files)
    elif root.is_dir():
        paths["train_parquet"] = root / f"{args.train_split}.parquet"
    if args.use_validation_split and (args.data_files is not None or root.is_dir()):
        paths["validation_parquet"] = root / f"{args.validation_split}.parquet"
    train_order_path = getattr(args, "train_order_path", None)
    if train_order_path is not None:
        if not train_order_path.is_file():
            raise FileNotFoundError(f"Frozen training order is missing: {train_order_path}")
        paths["train_order"] = train_order_path
    validation_row_ids_path = getattr(args, "validation_row_ids_path", None)
    if validation_row_ids_path is not None:
        if not validation_row_ids_path.is_file():
            raise FileNotFoundError(
                f"Frozen validation row IDs are missing: {validation_row_ids_path}"
            )
        paths["validation_row_ids"] = validation_row_ids_path
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Run input {key} is missing: {path}")
    return {key: file_sha256(path) for key, path in paths.items()}


def _identity_inputs(
    args: argparse.Namespace, vocab_path: Path, metadata_path: Path, git_commit: object
) -> dict[str, str]:
    campaign_path = getattr(args, "campaign_manifest", None)
    if campaign_path is None:
        return _run_input_hashes(args, vocab_path, metadata_path)
    campaign_path = Path(campaign_path)
    if not campaign_path.is_file():
        raise FileNotFoundError(f"Campaign manifest is missing: {campaign_path}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema") != 1 or campaign.get("code_commit") != git_commit:
        raise ValueError("Campaign manifest does not match the current code commit")
    relative_tokenizer = str(vocab_path)
    repo = Path(__file__).resolve().parents[2]
    if vocab_path.is_absolute() and vocab_path.is_relative_to(repo):
        relative_tokenizer = str(vocab_path.relative_to(repo))
    expected = campaign.get("frozen_files_sha256", {}).get(relative_tokenizer)
    if expected is None or file_sha256(vocab_path) != expected:
        raise ValueError("Tokenizer does not match the staged campaign")
    return {"campaign_manifest_sha256": file_sha256(campaign_path)}


def prepare_run_directory(
    args: argparse.Namespace, vocab_path: Path, metadata_path: Path
) -> Path | None:
    """Pin a fresh run's inputs, or fail before altering an incompatible run."""
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "run_identity.json"
    git_revision = get_git_revision()
    if getattr(args, "require_clean_git", False) and (
        git_revision["commit"] is None or git_revision["dirty"] is not False
    ):
        raise ValueError("This run requires a clean Git checkout with a resolved commit")
    identity = {
        "schema": 2,
        "args": _serializable_args(args),
        "inputs": _identity_inputs(args, vocab_path, metadata_path, git_revision["commit"]),
        "git": git_revision,
    }
    checkpoint = args.resume_from_checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if checkpoint.parent != output_dir.resolve() or not checkpoint.name.startswith(
            "checkpoint-"
        ):
            raise ValueError("Resume checkpoint must be a checkpoint-* directory in output_dir")
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (checkpoint / filename).is_file():
                raise ValueError(f"Incomplete resume checkpoint: missing {filename}")
        if not manifest_path.is_file():
            raise ValueError("Cannot resume: original run_identity.json is missing")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("schema") == 2:
            comparable = {key: saved.get(key) for key in identity}
        else:
            # Pre-migration runs retain their original input-hash contract.
            comparable = saved
            identity = {
                "args": identity["args"],
                "input_sha256": _run_input_hashes(args, vocab_path, metadata_path),
                "git": git_revision,
            }
        if comparable != identity:
            raise ValueError("Cannot resume: run arguments, inputs, or code revision differ")
        return checkpoint

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Fresh run destination is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    return None


def sequence_bucket(seq: str, mod: int) -> int:
    digest = hashlib.sha1(seq.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def detect_backend(args: argparse.Namespace) -> str:
    if args.device_backend != "auto":
        return args.device_backend
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_args(args: argparse.Namespace, backend: str) -> None:
    if getattr(args, "train_order_path", None) is not None and not getattr(
        args, "global_train_shuffle", False
    ):
        raise ValueError("--train_order_path requires --global_train_shuffle")
    if getattr(args, "validation_row_ids_path", None) is not None and not getattr(
        args, "use_validation_split", False
    ):
        raise ValueError("--validation_row_ids_path requires --use_validation_split")
    if args.max_seq_length is not None and args.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    if not 0.0 <= args.mlm_probability <= 1.0:
        raise ValueError("mlm_probability must be between 0 and 1")
    if args.bf16 and args.fp16:
        raise ValueError("bf16 and fp16 are mutually exclusive")
    if args.eval_size <= 0:
        raise ValueError("eval_size must be positive")
    if args.max_eval_batches < 0:
        raise ValueError("max_eval_batches must be >= 0")
    if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.val_split_mod < 2:
        raise ValueError("val_split_mod must be >= 2")
    if not 0 <= args.val_split_bucket < args.val_split_mod:
        raise ValueError("val_split_bucket must satisfy 0 <= bucket < val_split_mod")
    if backend == "cuda" and not torch.cuda.is_available():
        raise ValueError("device_backend=cuda requested but CUDA is not available")
    if backend == "cuda" and args.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "bf16 was requested, but the current CUDA device does not support bf16. Use --no-bf16."
        )
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device_backend=mps requested but MPS is not available")
    if args.load_best_model_at_end and args.save_steps != args.eval_steps:
        raise ValueError(
            "--load_best_model_at_end requires --save_steps to equal --eval_steps "
            "so every evaluated checkpoint can be selected as best."
        )
    if args.masking_strategy not in {"standard", "span", "hetero_span"}:
        raise ValueError(f"Unknown masking_strategy: {args.masking_strategy!r}")
    if args.masking_strategy in {"span", "hetero_span"}:
        if not (0.0 < args.span_p_geom < 1.0):
            raise ValueError("span_p_geom must be in (0, 1)")
        if args.span_max_length < 1:
            raise ValueError("span_max_length must be >= 1")
    if args.masking_strategy == "hetero_span":
        if args.heteroatom_start_weight <= 0.0:
            raise ValueError("heteroatom_start_weight must be > 0")
        if args.tokenizer_algorithm != APE or args.representation != SELFIES_REPRESENTATION:
            raise ValueError(
                "hetero_span finds heteroatoms in whole SELFIES bracket symbols, "
                "so it needs an APE SELFIES tokenizer"
            )


def adjust_args_for_backend(args: argparse.Namespace, backend: str) -> argparse.Namespace:
    # MPS and CPU are safest in full precision for this workflow.
    if backend in {"mps", "cpu"}:
        args.bf16 = False
        args.fp16 = False
        args.num_workers = 0

    if args.debug:
        args.eval_size = min(args.eval_size, 500)
        args.max_steps = min(args.max_steps, 200)
        args.logging_steps = min(args.logging_steps, 10)
        args.eval_steps = min(args.eval_steps, 50)
        args.save_steps = args.eval_steps
        args.tokenizer_validation_samples = min(args.tokenizer_validation_samples, 200)

    return args


def resolve_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    args.molecule_column = infer_molecule_column(
        args.dataset_name, args.representation, args.molecule_column
    )
    args.validation_split = infer_validation_split(
        args.dataset_name,
        args.validation_split,
    )
    if args.use_validation_split and not args.validation_split:
        raise ValueError(
            "--use_validation_split requested but no validation split was resolved. "
            "Pass --validation_split explicitly for this dataset."
        )
    return args


def preview_dataset_and_tokenizer(
    args: argparse.Namespace,
    tokenizer: "PreTrainedTokenizerBase",
    special_ids: dict[str, int],
    n_examples: int = 3,
) -> None:
    log("Previewing dataset and tokenization...")

    ds = get_streaming_dataset(
        args.dataset_name,
        split=args.train_split,
        seed=args.seed + 999,
        buffer_size=min(args.shuffle_buffer_size, 10_000),
        data_dir=args.data_dir,
        data_files=args.data_files,
    )

    examples: list[str] = []
    for row in ds:
        seq = normalize_sequence(row, args.molecule_column)
        if seq is None:
            continue
        examples.append(seq)
        if len(examples) >= n_examples:
            break

    local = _resolve_dataset_name_as_local_path(args.dataset_name)
    if local is None:
        local = find_local_dataset(args.data_dir, dataset_name=args.dataset_name)
    log(f"Dataset: {args.dataset_name}")
    log(f"Molecule column: {args.molecule_column}")
    log(f"Train split: {args.train_split}")
    log(f"Validation split: {args.validation_split}")
    log(f"Use validation split: {args.use_validation_split}")
    if args.data_files is not None:
        log(f"Dataset mode: parquet data_files={args.data_files}")
    elif local is not None:
        log(f"Dataset mode: local dataset at {local}")
    else:
        log("Dataset mode: streaming from HuggingFace Hub")
    log(f"Representation: {args.representation} ({args.tokenizer_algorithm} tokenizer)")

    for i, seq in enumerate(examples, start=1):
        encoded = encode_sequence(tokenizer, seq, args.max_seq_length)
        input_ids = encoded["input_ids"]

        eligible = eligible_token_ids(input_ids, special_ids)
        unk_count = sum(1 for x in eligible if x == special_ids["unk_token"])
        unk_rate = unk_count / max(1, len(eligible))

        tokens = tokenizer.convert_ids_to_tokens(input_ids[:30])

        log(f"Example {i}:")
        print(f"  raw input:   {seq[:300]}{'...' if len(seq) > 300 else ''}", flush=True)
        print(
            f"  token ids:   {input_ids[:30]}{' ...' if len(input_ids) > 30 else ''}",
            flush=True,
        )
        print(f"  tokens:      {tokens}", flush=True)
        print(f"  length:      {len(input_ids)}", flush=True)
        print(f"  unk count:   {unk_count}", flush=True)
        print(f"  unk rate:    {unk_rate:.3f}", flush=True)


def _sample_train_partition_sequences(args: argparse.Namespace, n: int) -> list[str]:
    ds = get_streaming_dataset(
        args.dataset_name,
        split=args.train_split,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
        data_files=args.data_files,
    )

    rows: list[str] = []
    for row in ds:
        seq = normalize_sequence(row, args.molecule_column)
        if seq is None:
            continue
        if (
            not args.use_validation_split
            and sequence_bucket(seq, args.val_split_mod) == args.val_split_bucket
        ):
            continue
        rows.append(seq)
        if len(rows) >= n:
            break

    return rows


def _pretokenized_example(row: dict[str, Any], max_seq_length: int | None) -> dict[str, list[int]]:
    ids = [int(token_id) for token_id in row["input_ids"]]
    if max_seq_length is not None and len(ids) > max_seq_length:
        raise ValueError(
            f"Pretokenized molecule has {len(ids)} tokens, exceeding context {max_seq_length}"
        )
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _encode_without_truncation(
    tokenizer: "PreTrainedTokenizerBase", seq: str, max_seq_length: int | None
) -> dict[str, list[int]]:
    encoded = encode_sequence(tokenizer, seq, None)
    if max_seq_length is not None and len(encoded["input_ids"]) > max_seq_length:
        raise ValueError(
            f"Molecule has {len(encoded['input_ids'])} tokens, exceeding context {max_seq_length}"
        )
    return encoded


def _split_key(row: dict[str, Any], molecule_column: str) -> str | None:
    seq = normalize_sequence(row, molecule_column)
    if seq is not None:
        return seq
    if "input_ids" in row:
        return ",".join(str(int(token_id)) for token_id in row["input_ids"])
    return None


def _is_validation_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    key = _split_key(row, args.molecule_column)
    return key is not None and sequence_bucket(key, args.val_split_mod) == args.val_split_bucket


def corpus_only_training_parquet(args: argparse.Namespace) -> Path:
    """Identify the exact local Parquet that the revision training will stream."""
    if args.data_dir is not None:
        raise ValueError("Corpus-only training requires a local Parquet, not --data_dir")
    if args.data_files is None and find_local_dataset(dataset_name=args.dataset_name) is not None:
        # get_streaming_dataset prefers a name-matched Arrow dataset under data/ over the
        # Parquet hashed below, so the gate would check a file that training never reads.
        raise ValueError(
            "Corpus-only training found a matching local Arrow dataset under data/; "
            "pass --data_files to stream the scanned training Parquet explicitly"
        )
    if args.data_files is not None:
        source = Path(args.data_files)
    else:
        source = Path(args.dataset_name) / f"{args.train_split}.parquet"
    if not source.is_file():
        raise ValueError(
            f"Corpus-only training requires one local Parquet file for the training split: {source}"
        )
    return source


def assert_corpus_only_vocab(metadata: dict[str, Any], training_parquet: Path) -> None:
    """Require uninjected primitives scanned from this exact training file."""
    scan = metadata.get("corpus_primitive_scan")
    if not isinstance(scan, dict):
        raise ValueError("Corpus-only tokenizer requires corpus_primitive_scan metadata")
    if not scan.get("sha256") or int(scan.get("n_rows", 0)) <= 0:
        raise ValueError("Corpus-only tokenizer scan lacks a source hash or positive row count")
    if scan["sha256"] != file_sha256(training_parquet):
        raise ValueError("Corpus-only tokenizer was scanned from a different training Parquet")
    if int(metadata.get("extra_vocab_symbols_requested", 0)) != 0:
        raise ValueError("Corpus-only tokenizer includes requested extra vocabulary symbols")
    if int(metadata.get("extra_vocab_symbols_added", 0)) != 0:
        raise ValueError("Corpus-only tokenizer includes added extra vocabulary symbols")


def read_training_tokenizer(
    args: argparse.Namespace,
) -> tuple["PreTrainedTokenizerBase", dict[str, Any], Path, Path]:
    """Load the tokenizer file, checked against its metadata hash.

    Sets ``args.tokenizer_algorithm`` and ``args.representation`` from the metadata.
    """
    tokenizer, metadata, vocab_path, metadata_path = load_verified_tokenizer(
        args.tokenizer_vocab_path, args.tokenizer_metadata_path, log=log
    )
    if "tokenizer_sha256" not in metadata:
        raise ValueError(
            f"Training tokenizer metadata {metadata_path} must record 'tokenizer_sha256'"
        )
    args.tokenizer_algorithm = tokenizer_algorithm(metadata)
    args.representation = tokenizer_representation(metadata)
    return tokenizer, metadata, vocab_path, metadata_path


def validate_tokenizer_for_training(
    args: argparse.Namespace,
    tokenizer: "PreTrainedTokenizerBase",
    metadata: dict[str, Any],
) -> tuple[int, dict[str, int], dict[str, float]]:
    """Gate the tokenizer on a training sample; return vocabulary size, special IDs and stats."""
    if args.require_corpus_only_vocab:
        assert_corpus_only_vocab(metadata, corpus_only_training_parquet(args))

    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size < 100:
        raise ValueError(f"Suspiciously small tokenizer vocabulary: {vocab_size}")

    special_ids = resolve_special_ids(tokenizer)
    assert_special_ids(special_ids)

    validation_sequences = _sample_train_partition_sequences(
        args, n=args.tokenizer_validation_samples
    )
    validate_sample_shape(validation_sequences, args.representation)

    assert_representation_compatible(
        tokenizer, special_ids, args.representation, args.max_seq_length
    )

    stats = compute_tokenization_stats(
        tokenizer=tokenizer,
        sequences=validation_sequences,
        max_seq_length=args.max_seq_length,
        special_ids=special_ids,
    )

    if stats["unk_rate"] > args.unk_rate_threshold:
        raise ValueError(
            f"Unknown-token rate too high: {stats['unk_rate']:.6f} "
            f"(threshold {args.unk_rate_threshold:.6f})"
        )
    if stats["silent_loss_rate"] > 0:
        raise ValueError(
            "Tokenization silently changed training inputs: "
            f"silent_loss_rate={stats['silent_loss_rate']:.6f}"
        )
    if stats["empty_sequence_rate"] > 0:
        raise ValueError("Tokenizer produced empty tokenized outputs.")
    if stats["mostly_unknown_rate"] > 0.01:
        raise ValueError(
            f"Too many sequences are mostly unknown tokens: {stats['mostly_unknown_rate']:.4f}"
        )

    return vocab_size, special_ids, stats


def make_train_iterable_dataset(
    args: argparse.Namespace, tokenizer: "PreTrainedTokenizerBase"
) -> IterableDataset:
    if getattr(args, "global_train_shuffle", False):
        source = corpus_only_training_parquet(args)
        import pyarrow.parquet as pq

        available_columns = pq.ParquetFile(source).schema_arrow.names
        if args.molecule_column in available_columns:
            input_columns = [args.molecule_column]
        elif "input_ids" in available_columns:
            input_columns = ["input_ids"]
        else:
            raise ValueError(f"Missing {args.molecule_column!r} and input_ids columns in {source}")
        cache_dir = Path(args.output_dir) / "dataset_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        indexed = Dataset.from_parquet(str(source), columns=input_columns, cache_dir=str(cache_dir))
        if not len(indexed):
            raise ValueError(f"No training rows in {source}")
        order_path = getattr(args, "train_order_path", None)
        if order_path is not None:
            order = np.load(order_path, allow_pickle=False)
            if order.ndim != 1 or len(order) != len(indexed):
                raise ValueError("Training order length differs from source Parquet")
            if not np.issubdtype(order.dtype, np.integer) or not np.array_equal(
                np.sort(order), np.arange(len(indexed))
            ):
                raise ValueError("Training order must be a permutation of source row IDs")
            indexed = indexed.select(order.tolist())
            log(f"Using frozen source-row order {order_path}: {file_sha256(order_path)}")
        else:
            # Compatibility path for previous runs. Factorial runs supply an
            # explicit order file through the shared campaign manifest.
            indexed = indexed.shuffle(seed=args.seed + 100)
        ds = indexed.to_iterable_dataset(num_shards=min(64, len(indexed)))
        if order_path is None:
            ds = ds.shuffle(seed=args.seed + 101, buffer_size=args.shuffle_buffer_size)
        log(f"Training rows globally ordered from {source}: {len(indexed):,}")
    else:
        ds = get_streaming_dataset(
            args.dataset_name,
            split=args.train_split,
            seed=args.seed + 100,
            buffer_size=args.shuffle_buffer_size,
            data_dir=args.data_dir,
            data_files=args.data_files,
        )

    def keep_train(row: dict[str, Any]) -> bool:
        has_content = (
            normalize_sequence(row, args.molecule_column) is not None or "input_ids" in row
        )
        if not has_content:
            return False
        if args.use_validation_split:
            return True
        return not _is_validation_row(row, args)

    ds = ds.filter(keep_train)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        if "input_ids" in row:
            return _pretokenized_example(row, args.max_seq_length)
        seq = normalize_sequence(row, args.molecule_column)
        if seq is None:
            raise ValueError(f"Training row is missing {args.molecule_column!r} and input_ids.")
        return _encode_without_truncation(tokenizer, seq, args.max_seq_length)

    return ds.map(preprocess)


def make_eval_dataset(args: argparse.Namespace, tokenizer: "PreTrainedTokenizerBase") -> Dataset:
    n_eval = args.eval_size
    if args.max_eval_batches > 0:
        n_eval = min(n_eval, args.max_eval_batches * args.per_device_eval_batch_size)

    log(
        "Building finite validation set: "
        f"requested_eval_size={args.eval_size}, "
        f"max_eval_batches={args.max_eval_batches or 'none'}, "
        f"actual_eval_size={n_eval}"
    )

    validation_row_ids_path = getattr(args, "validation_row_ids_path", None)
    if validation_row_ids_path is not None:
        if not args.use_validation_split:
            raise ValueError("Frozen validation row IDs require --use_validation_split")
        source = Path(args.dataset_name) / f"{args.validation_split}.parquet"
        if not source.is_file():
            raise FileNotFoundError(f"Frozen validation Parquet is missing: {source}")
        indices = np.load(validation_row_ids_path, allow_pickle=False)
        if indices.ndim != 1 or len(indices) != n_eval:
            raise ValueError("Frozen validation row ID count differs from eval_size")
        frame = Dataset.from_parquet(str(source), columns=[args.molecule_column])
        if (
            not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(frame))
        ):
            raise ValueError("Frozen validation row IDs must be unique in-range integers")
        selected = frame.select(indices.tolist())
        rows = [
            _encode_without_truncation(tokenizer, row[args.molecule_column], args.max_seq_length)
            for row in selected
        ]
        log(
            f"Using frozen validation rows {validation_row_ids_path}: {file_sha256(validation_row_ids_path)}"
        )
        return Dataset.from_list(rows)

    eval_split = args.validation_split if args.use_validation_split else args.train_split
    eval_data_files = args.data_files
    if args.use_validation_split and args.data_files is not None:
        # --data_files identifies the training Parquet for the global shuffle.
        # Passing it through to the validation loader would silently read the
        # training population even when --validation_split is set.
        validation_source = Path(args.dataset_name) / f"{eval_split}.parquet"
        if not validation_source.is_file():
            raise FileNotFoundError(
                f"Explicit training Parquet requires a separate validation file: "
                f"{validation_source}"
            )
        eval_data_files = str(validation_source)
    ds = get_streaming_dataset(
        args.dataset_name,
        split=eval_split,
        seed=args.seed + 200,
        buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
        data_files=eval_data_files,
    )

    rows: list[dict[str, list[int]]] = []
    pbar = tqdm(total=n_eval, desc="Building finite validation set")

    for row in ds:
        seq = normalize_sequence(row, args.molecule_column)
        pretokenized = "input_ids" in row
        if seq is None and not pretokenized:
            continue
        if not args.use_validation_split and not _is_validation_row(row, args):
            continue

        if pretokenized:
            rows.append(_pretokenized_example(row, args.max_seq_length))
        else:
            if seq is None:
                continue
            rows.append(_encode_without_truncation(tokenizer, seq, args.max_seq_length))
        pbar.update(1)
        if len(rows) >= n_eval:
            break

    pbar.close()

    if not rows:
        if args.use_validation_split:
            raise RuntimeError(
                "Validation set is empty after sampling from validation split. "
                "Check --validation_split and dataset contents."
            )
        raise RuntimeError(
            "Validation set is empty after deterministic split. "
            "Try a larger eval_size or adjust val_split_mod/val_split_bucket."
        )

    return Dataset.from_list(rows)


# Pinned upstream configuration only; no pretrained weights are loaded.
_MODERNBERT_BASE_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "modernbert_base_config.json"
)

# Both presets use a 128-token context; --max_seq_length overrides it.
DEFAULT_MAX_SEQ_LENGTH = 128

LOCAL_MODERNBERT_PRESETS = {
    # ~28–30M params, 512-dim. Sub-MoLFormer-XL (46.8M) efficiency variant.
    # Comparable to Chemformer (45M, 512-dim) and Uni-Mol (47M, 512-dim).
    # To check parameter count before a full run:
    #   uv run python -c "
    #   from transformers import AutoConfig, AutoModelForMaskedLM
    #   from modernmolbert.train_selfies_ape_modernbert import build_modernbert_config, LOCAL_MODERNBERT_PRESETS
    #   import types
    #   args = types.SimpleNamespace(model_size='small', max_seq_length=256)
    #   config = build_modernbert_config(args, vocab_size=5000, special_ids={'bos_token':0,'pad_token':1,'eos_token':2,'unk_token':3,'mask_token':4})
    #   model = AutoModelForMaskedLM.from_config(config)
    #   print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')
    #   "
    "small": {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "global_attn_every_n_layers": 3,
        "local_attention": 128,
    },
    # ~85–95M params, 768-dim. Direct SELFormer (86.7M, 768-dim) comparator.
    # Also matches MolBERT (85M, 768-dim) and Uni-Mol2 (84M, 768-dim).
    # Trained from scratch with our vocabulary (not fine-tuned from official weights).
    # Effective batch size 256: --per_device_train_batch_size 64 --gradient_accumulation_steps 4
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "global_attn_every_n_layers": 3,
        "local_attention": 128,
    },
}


def build_modernbert_config(
    args: argparse.Namespace,
    vocab_size: int,
    special_ids: dict[str, int],
):
    # Start from the official base config to preserve ModernBERT-specific fields,
    # then override only the scale-related fields for the chosen preset.
    provenance = json.loads(
        _MODERNBERT_BASE_CONFIG.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    if file_sha256(_MODERNBERT_BASE_CONFIG) != provenance["sha256"]:
        raise ValueError("Pinned ModernBERT base configuration hash mismatch")
    config = AutoConfig.from_pretrained(_MODERNBERT_BASE_CONFIG)
    for key, value in LOCAL_MODERNBERT_PRESETS[args.model_size].items():
        setattr(config, key, value)
    # Regenerate layer_types to match the new num_hidden_layers and global_attn_every_n_layers.
    # The base config carries a fixed 22-element list; overriding num_hidden_layers alone leaves
    # them out of sync and triggers a save-time validation error.
    every_n = getattr(config, "global_attn_every_n_layers", 3)
    config.layer_types = [
        "sliding_attention" if bool(i % every_n) else "full_attention"
        for i in range(config.num_hidden_layers)
    ]

    try:
        import flash_attn  # type: ignore # noqa

        config._attn_implementation = "flash_attention_2"
    except ImportError:
        pass

    # Molecular tokenizer-specific fields.
    config.vocab_size = vocab_size
    config.pad_token_id = special_ids["pad_token"]
    config.bos_token_id = special_ids["bos_token"]
    config.eos_token_id = special_ids["eos_token"]
    # ModernBERT also keeps CLS/SEP IDs from its base vocabulary. Align them
    # with the molecular tokenizer so saved configs never reference IDs above
    # the new vocabulary size.
    config.cls_token_id = special_ids["bos_token"]
    config.sep_token_id = special_ids["eos_token"]
    # Optional context-length override.
    if args.max_seq_length is not None:
        config.max_position_embeddings = args.max_seq_length
    return config


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = logits if logits.ndim == labels.ndim else np.argmax(logits, axis=-1)
    mask = labels != -100

    if mask.sum() == 0:
        return {"masked_accuracy": 0.0}

    return {"masked_accuracy": float((preds[mask] == labels[mask]).mean())}


def preprocess_logits_for_metrics(logits: Any, _labels: Any) -> torch.Tensor:
    """Keep token predictions rather than a full [batch, sequence, vocabulary] tensor."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def log_training_plan(
    args: argparse.Namespace,
    backend: str,
    n_params: int | None = None,
    world_size: int = 1,
) -> None:
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    log("Training plan:")
    print(f"  backend:                    {backend}", flush=True)
    print(f"  model_size:                 {args.model_size}", flush=True)
    if n_params is not None:
        print(f"  parameters:                 {n_params / 1e6:.2f}M", flush=True)
    print(f"  max_steps:                  {args.max_steps}", flush=True)
    print(f"  max_seq_length:             {args.max_seq_length}", flush=True)
    print(f"  mlm_probability:            {args.mlm_probability}", flush=True)
    print(f"  masking_strategy:           {args.masking_strategy}", flush=True)
    if args.masking_strategy in {"span", "hetero_span"}:
        print(f"  span_p_geom:                {args.span_p_geom}", flush=True)
        print(f"  span_max_length:            {args.span_max_length}", flush=True)
    if args.masking_strategy == "hetero_span":
        print(f"  heteroatom_start_weight:    {args.heteroatom_start_weight}", flush=True)
    print(f"  train batch/device:         {args.per_device_train_batch_size}", flush=True)
    print(f"  gradient_accumulation:      {args.gradient_accumulation_steps}", flush=True)
    print(f"  effective batch size:       {effective_batch_size}", flush=True)
    print(f"  eval batch/device:          {args.per_device_eval_batch_size}", flush=True)
    print(
        f"  load_best_model_at_end:     {args.load_best_model_at_end}",
        flush=True,
    )
    print(f"  metric_for_best_model:      {args.metric_for_best_model}", flush=True)
    print(f"  greater_is_better:          {args.greater_is_better}", flush=True)
    print(f"  eval every steps:           {args.eval_steps}", flush=True)
    print(f"  save every steps:           {args.save_steps}", flush=True)
    print(f"  save_total_limit:           {args.save_total_limit}", flush=True)
    print(f"  logging every steps:        {args.logging_steps}", flush=True)
    print(f"  report_to:                  {args.report_to}", flush=True)
    print(f"  world_size:                 {world_size}", flush=True)
    print(f"  bf16/fp16:                  {args.bf16}/{args.fp16}", flush=True)


def finalize_run_identity(
    args: argparse.Namespace,
    backend: str,
    vocab_size: int,
    special_ids: dict[str, int],
    n_params: int,
    tokenizer_stats: dict[str, float],
    tokenizer_metadata_path: Path,
    final_model_dir: Path,
    selected_step: int | None,
    final_eval_metrics: dict[str, float],
    trainer_state: dict[str, Any],
) -> None:
    """Append final outcome to the run's one machine-readable record."""
    path = Path(args.output_dir) / "run_identity.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("schema") != 2:
        raise ValueError("New training runs require a schema-2 run identity")
    model_files = [
        candidate
        for name in ("model.safetensors", "pytorch_model.bin")
        if (candidate := final_model_dir / name).is_file()
    ]
    if len(model_files) != 1:
        raise ValueError("Expected exactly one final model weight file")
    metadata = load_tokenizer_metadata(tokenizer_metadata_path)
    identity["result"] = {
        "terminal_step": int(trainer_state["global_step"]),
        "selected_step": int(selected_step) if selected_step is not None else None,
        "selection_rule": "best_validation" if args.load_best_model_at_end else "terminal_step",
        "final_model_file": model_files[0].name,
        "final_model_sha256": file_sha256(model_files[0]),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "backend": backend,
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "num_parameters": n_params,
        "tokenizer_stats": tokenizer_stats,
        "final_eval_metrics": final_eval_metrics,
        "trainer_state_summary": trainer_state,
    }
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.assume_static_by_default = False
    args = parse_args()
    load_dotenv()

    if args.hf_login:
        token = resolve_hf_token(hf_login=False)
        if not token:
            raise ValueError(
                "--hf_login was set but neither HF_TOKEN_ORG nor HF_TOKEN is available."
            )
        resolve_hf_token(hf_login=True)

    tokenizer, tokenizer_metadata, tokenizer_vocab_path, tokenizer_metadata_path = (
        read_training_tokenizer(args)
    )
    args = resolve_dataset_args(args)
    backend = detect_backend(args)
    args = adjust_args_for_backend(args, backend)
    validate_args(args, backend)

    if args.max_seq_length is None:
        args.max_seq_length = DEFAULT_MAX_SEQ_LENGTH

    output_dir = Path(args.output_dir)
    resume_checkpoint = prepare_run_directory(args, tokenizer_vocab_path, tokenizer_metadata_path)

    set_seed(args.seed)

    log(f"Backend: {backend}")
    log(f"bf16={args.bf16}, fp16={args.fp16}")
    log(f"Dataset: {args.dataset_name}")
    log(f"Molecule column: {args.molecule_column}")
    log(f"Train split: {args.train_split}")
    log(f"Validation split: {args.validation_split}")
    log(f"Use validation split: {args.use_validation_split}")
    log(f"Tokenizer: {args.tokenizer_algorithm} {args.representation} ({tokenizer_vocab_path})")

    log("Validating tokenizer...")
    vocab_size, special_ids, tokenizer_stats = validate_tokenizer_for_training(
        args, tokenizer, tokenizer_metadata
    )

    log(f"Vocabulary size: {vocab_size}")
    log(f"Special token IDs: {special_ids}")
    log("Tokenizer validation stats:")
    for key in sorted(tokenizer_stats):
        value = tokenizer_stats[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}", flush=True)
        else:
            print(f"  {key}: {value}", flush=True)
    if tokenizer_stats["truncation_rate"] > args.truncation_warn_threshold:
        log(
            f"Warning: truncation rate is high "
            f"({tokenizer_stats['truncation_rate']:.4f} > {args.truncation_warn_threshold:.4f})"
        )

    log("Building datasets...")
    train_dataset = make_train_iterable_dataset(args, tokenizer)
    eval_dataset = make_eval_dataset(args, tokenizer)

    preview_dataset_and_tokenizer(
        args=args,
        tokenizer=tokenizer,
        special_ids=special_ids,
        n_examples=3,
    )

    log("Building ModernBERT model (this can take a while on MPS/CPU)...")

    config = build_modernbert_config(args, vocab_size, special_ids)
    model = AutoModelForMaskedLM.from_config(config)
    n_params = sum(p.numel() for p in model.parameters())

    log(
        f"Config: ModernBERT-{args.model_size}, "
        f"vocab_size={config.vocab_size}, "
        f"hidden_size={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, "
        f"max_position_embeddings={config.max_position_embeddings}"
    )
    log(f"Model parameters: {n_params / 1e6:.2f}M")

    collator = MolecularMLMCollator(
        pad_token_id=special_ids["pad_token"],
        mask_token_id=special_ids["mask_token"],
        vocab_size=vocab_size,
        mlm_probability=args.mlm_probability,
        special_token_ids=list(special_ids.values()),
        masking_strategy=args.masking_strategy,
        span_p_geom=args.span_p_geom,
        span_max_length=args.span_max_length,
        heteroatom_start_weight=args.heteroatom_start_weight,
        ids_to_tokens={index: token for token, index in tokenizer.get_vocab().items()},
    )

    report_to = [] if args.report_to == "none" else [args.report_to]

    if args.report_to == "tensorboard":
        log("TensorBoard enabled.")
        log(f"Follow training with: tensorboard --logdir {output_dir}")
    else:
        log("TensorBoard disabled. Use --report_to tensorboard to enable it.")

    log("Testing one training batch before Trainer...")

    one = []

    it = iter(train_dataset)

    for _ in range(args.per_device_train_batch_size):
        try:
            one.append(next(it))
        except StopIteration:
            break

    if not one:
        raise RuntimeError(
            "No training examples available after filtering. "
            "Check dataset, split, and molecule column."
        )

    batch = collator(one)

    print({k: tuple(v.shape) for k, v in batch.items()}, flush=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(backend == "cuda"),
        remove_unused_columns=False,
        prediction_loss_only=not args.compute_masked_accuracy,
        include_num_input_tokens_seen="non_padding",
        report_to=report_to,
        seed=args.seed,
        data_seed=args.seed,
    )

    world_size = training_args.world_size if hasattr(training_args, "world_size") else 1

    if args.masking_strategy in {"span", "hetero_span"} and args.num_workers < 2:
        log(
            "Warning: masking_strategy='span'/'hetero_span' runs in Python on the "
            "data-loader path. Consider --num_workers >= 4 to overlap collation with "
            "GPU compute and avoid becoming a training bottleneck."
        )

    log_training_plan(args, backend, n_params=n_params, world_size=world_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if args.compute_masked_accuracy else None,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if args.compute_masked_accuracy else None
        ),
    )
    log("Starting training...")
    log(f"Training logs will print every {args.logging_steps} steps.")
    log(f"Evaluation will run every {args.eval_steps} steps.")
    log(f"Checkpoints will be saved every {args.save_steps} steps.")
    log(f"Only the most recent {args.save_total_limit} checkpoints will be kept.")
    log(f"Intermediate checkpoints: {output_dir}/checkpoint-*")
    log(f"Final model will be saved to: {output_dir}/final_model")
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )

    print("Saving final model...")

    terminal_step = int(trainer.state.global_step)
    if terminal_step != args.max_steps:
        raise RuntimeError(
            f"Training stopped at step {terminal_step}, expected terminal step {args.max_steps}"
        )
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    # global_step stays terminal after load_best_model_at_end restores an earlier checkpoint.
    selected_step = (
        getattr(trainer.state, "best_global_step", None)
        if args.load_best_model_at_end
        else terminal_step
    )
    (final_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_step": selected_step,
                "selection_rule": (
                    "best_validation" if args.load_best_model_at_end else "terminal_step"
                ),
                "loaded_best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None)
                if args.load_best_model_at_end
                else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    copy_tokenizer_artifacts(
        vocab_path=tokenizer_vocab_path,
        metadata_path=tokenizer_metadata_path,
        output_dir=output_dir,
        final_model_dir=final_dir,
        model_max_length=args.max_seq_length,
    )

    metrics = train_result.metrics

    estimated_train_samples = (
        args.max_steps
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * world_size
    )
    metrics["train_samples_streaming"] = float(estimated_train_samples)
    metrics["num_parameters"] = float(n_params)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        try:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["eval_perplexity"] = float("inf")

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    trainer.save_state()

    trainer_state_summary = {
        "best_global_step": getattr(trainer.state, "best_global_step", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "global_step": getattr(trainer.state, "global_step", None),
    }

    finalize_run_identity(
        args=args,
        backend=backend,
        vocab_size=vocab_size,
        special_ids=special_ids,
        n_params=n_params,
        tokenizer_stats=tokenizer_stats,
        tokenizer_metadata_path=tokenizer_metadata_path,
        final_model_dir=final_dir,
        selected_step=selected_step,
        final_eval_metrics={k: float(v) for k, v in eval_metrics.items()},
        trainer_state=trainer_state_summary,
    )

    print("Done.")
    print(f"Final model: {final_dir}")
    print(f"Hub-ready folder: {final_dir}")
    if args.tokenizer_algorithm == APE:
        print("Load tokenizer from final_model/ape_tokenizer with trust_remote_code=True")
        print(f"Tokenizer vocabulary: {final_dir / 'vocab.json'}")
    else:
        print(f"Load tokenizer from {final_dir} with AutoTokenizer")


if __name__ == "__main__":
    main()
