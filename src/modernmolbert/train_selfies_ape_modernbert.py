#!/usr/bin/env python3
"""Train a ModernBERT masked-language model for SELFIES molecular strings.

Model training requires an existing, vetted tokenizer vocabulary and metadata.
Tokenizer training is intentionally a separate command:

    python -m modernmolbert.train_ape_tokenizer
"""

import argparse
import hashlib
import re
import time
import json
import math
import platform
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
import os

from dotenv import load_dotenv
from huggingface_hub import login

import numpy as np
import torch
import transformers
from datasets import Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    PUBCHEM10M_DATASET,
    SELFIES_REPRESENTATION,
    assert_metadata_representation,
    compute_tokenization_stats,
    copy_tokenizer_artifacts,
    default_selfies_tokenizer_path,
    eligible_token_ids,
    encode_sequence,
    file_sha256,
    find_local_dataset,
    get_streaming_dataset,
    infer_selfies_column,
    infer_validation_split,
    load_tokenizer_metadata,
    metadata_path_for_vocab,
    normalize_sequence,
    resolve_special_ids,
    tokenizer_vocab_size,
    validate_selfies_sample_shape,
)

DATASET_NAME = PUBCHEM10M_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SELFIES ModernBERT MLM with a vetted APE tokenizer.",
    )

    # Paths
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer_vocab_path",
        type=str,
        default=str(default_selfies_tokenizer_path()),
        help="SELFIES tokenizer vocabulary JSON.",
    )
    parser.add_argument(
        "--tokenizer_metadata_path",
        type=str,
        default=None,
        help="Tokenizer metadata JSON. Defaults to <vocab>.metadata.json.",
    )

    # Dataset
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--selfies_column",
        type=str,
        default=None,
        help=("Column containing SELFIES strings. Defaults by dataset."),
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
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
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
        choices=["small", "medium", "base", "large"],
        default="small",
        help="ModernBERT architecture preset.",
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
            "APE tokens that contain heteroatoms (N, O, S, P, F, Cl, Br, I, Se, Si)."
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
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
        "--hf_login",
        action="store_true",
        help="Call huggingface_hub.login using HF_TOKEN before loading datasets/models.",
    )

    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
    if args.masking_strategy == "hetero_span" and args.heteroatom_start_weight <= 0.0:
        raise ValueError("heteroatom_start_weight must be > 0")


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
    args.selfies_column = infer_selfies_column(args.dataset_name, args.selfies_column)
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
    tokenizer: APEPreTrainedTokenizer,
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
        seq = normalize_sequence(row, args.selfies_column)
        if seq is None:
            continue
        examples.append(seq)
        if len(examples) >= n_examples:
            break

    local = find_local_dataset(args.data_dir, dataset_name=args.dataset_name)
    log(f"Dataset: {args.dataset_name}")
    log(f"SELFIES column: {args.selfies_column}")
    log(f"Train split: {args.train_split}")
    log(f"Validation split: {args.validation_split}")
    log(f"Use validation split: {args.use_validation_split}")
    if args.data_files is not None:
        log(f"Dataset mode: parquet data_files={args.data_files}")
    else:
        log(
            f"Dataset mode: {'local (from disk): ' + str(local) if local else 'streaming (HF Hub)'}"
        )
    log(f"Representation: {SELFIES_REPRESENTATION}")

    for i, seq in enumerate(examples, start=1):
        encoded = encode_sequence(tokenizer, seq, args.max_seq_length)
        input_ids = encoded["input_ids"]

        eligible = eligible_token_ids(input_ids, special_ids)
        unk_count = sum(1 for x in eligible if x == special_ids["unk_token"])
        unk_rate = unk_count / max(1, len(eligible))

        # Best effort token display. Adjust if your APE tokenizer has a different method.
        tokens = None
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                tokens = tokenizer.convert_ids_to_tokens(input_ids[:30])
            except Exception:
                tokens = None

        log(f"Example {i}:")
        print(f"  raw SELFIES: {seq[:300]}{'...' if len(seq) > 300 else ''}", flush=True)
        print(
            f"  token ids:   {input_ids[:30]}{' ...' if len(input_ids) > 30 else ''}",
            flush=True,
        )
        if tokens is not None:
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
        seq = normalize_sequence(row, args.selfies_column)
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


def load_and_validate_tokenizer(
    args: argparse.Namespace,
) -> tuple[
    APEPreTrainedTokenizer,
    dict[str, Any],
    Path,
    Path,
    int,
    dict[str, int],
    dict[str, float],
]:
    vocab_path = Path(args.tokenizer_vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Tokenizer vocabulary not found: {vocab_path}\n"
            "Train a tokenizer first with:\n"
            "  python -m modernmolbert.train_ape_tokenizer"
        )

    metadata_path = (
        Path(args.tokenizer_metadata_path)
        if args.tokenizer_metadata_path is not None
        else metadata_path_for_vocab(vocab_path)
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Tokenizer metadata not found: {metadata_path}\n"
            "Training requires tokenizer metadata with representation and hash details."
        )

    metadata = load_tokenizer_metadata(metadata_path)
    assert_metadata_representation(metadata, expected_representation=SELFIES_REPRESENTATION)

    recorded_sha = str(metadata.get("tokenizer_sha256", ""))
    actual_sha = file_sha256(vocab_path)
    if recorded_sha and recorded_sha != actual_sha:
        raise ValueError(
            "Tokenizer hash mismatch between file and metadata. "
            f"metadata={recorded_sha}, file={actual_sha}"
        )

    tokenizer = APEPreTrainedTokenizer(representation=SELFIES_REPRESENTATION)
    tokenizer.load_vocabulary_file(vocab_path)

    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size < 100:
        raise ValueError(f"Suspiciously small tokenizer vocabulary: {vocab_size}")

    special_ids = resolve_special_ids(tokenizer)

    validation_sequences = _sample_train_partition_sequences(
        args, n=args.tokenizer_validation_samples
    )
    validate_selfies_sample_shape(validation_sequences)

    ethanol_encoded = encode_sequence(tokenizer, "[C][C][O]", args.max_seq_length)["input_ids"]
    eligible_ethanol = eligible_token_ids(ethanol_encoded, special_ids)
    if not eligible_ethanol:
        raise ValueError("Tokenizer produced no usable SELFIES tokens for [C][C][O]")
    unk_ethanol = sum(1 for x in eligible_ethanol if x == special_ids["unk_token"])
    unk_ethanol_rate = unk_ethanol / len(eligible_ethanol)
    if unk_ethanol_rate > 0.05:
        tokens = (
            tokenizer.convert_ids_to_tokens(ethanol_encoded)
            if hasattr(tokenizer, "convert_ids_to_tokens")
            else None
        )
        raise ValueError(
            "Tokenizer is not SELFIES-compatible: "
            f"[C][C][O] unk_rate={unk_ethanol_rate:.3f}, "
            f"ids={ethanol_encoded}, tokens={tokens}"
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
    if stats["empty_sequence_rate"] > 0:
        raise ValueError("Tokenizer produced empty tokenized outputs.")
    if stats["mostly_unknown_rate"] > 0.01:
        raise ValueError(
            f"Too many sequences are mostly unknown tokens: {stats['mostly_unknown_rate']:.4f}"
        )

    return (
        tokenizer,
        metadata,
        vocab_path,
        metadata_path,
        vocab_size,
        special_ids,
        stats,
    )


def make_train_iterable_dataset(
    args: argparse.Namespace, tokenizer: APEPreTrainedTokenizer
) -> IterableDataset:
    ds = get_streaming_dataset(
        args.dataset_name,
        split=args.train_split,
        seed=args.seed + 100,
        buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
        data_files=args.data_files,
    )

    def keep_train(row: dict[str, Any]) -> bool:
        seq = normalize_sequence(row, args.selfies_column)
        if seq is None:
            return False
        if args.use_validation_split:
            return True
        return sequence_bucket(seq, args.val_split_mod) != args.val_split_bucket

    ds = ds.filter(keep_train)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_sequence(row, args.selfies_column)
        assert seq is not None
        return encode_sequence(tokenizer, seq, args.max_seq_length)

    return ds.map(preprocess)


def make_eval_dataset(args: argparse.Namespace, tokenizer: APEPreTrainedTokenizer) -> Dataset:
    requested_eval_size = args.eval_size
    n_eval = requested_eval_size

    if args.max_eval_batches > 0:
        batch_capped_eval_size = args.max_eval_batches * args.per_device_eval_batch_size

        n_eval = min(requested_eval_size, batch_capped_eval_size)

        log(
            "Building finite validation set: "
            f"requested_eval_size={requested_eval_size}, "
            f"max_eval_batches={args.max_eval_batches}, "
            f"per_device_eval_batch_size={args.per_device_eval_batch_size}, "
            f"actual_eval_size={n_eval}"
        )

    else:
        log(
            "Building finite validation set: "
            f"requested_eval_size={requested_eval_size}, "
            f"max_eval_batches=none, "
            f"actual_eval_size={n_eval}"
        )

    eval_split = args.validation_split if args.use_validation_split else args.train_split
    ds = get_streaming_dataset(
        args.dataset_name,
        split=eval_split,
        seed=args.seed + 200,
        buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
        data_files=args.data_files,
    )

    rows: list[dict[str, list[int]]] = []
    pbar = tqdm(total=n_eval, desc="Building finite validation set")

    for row in ds:
        seq = normalize_sequence(row, args.selfies_column)
        if seq is None:
            continue
        if (
            not args.use_validation_split
            and sequence_bucket(seq, args.val_split_mod) != args.val_split_bucket
        ):
            continue

        rows.append(encode_sequence(tokenizer, seq, args.max_seq_length))
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


@dataclass
class MolecularMLMCollator:
    pad_token_id: int
    mask_token_id: int
    vocab_size: int
    mlm_probability: float
    special_token_ids: list[int]
    masking_strategy: str = "standard"
    span_p_geom: float = 0.4
    span_max_length: int = 6
    heteroatom_start_weight: float = 2.0
    ids_to_tokens: dict[int, str] = field(default_factory=dict)

    # ClassVar: excluded from __init__ by dataclass machinery.
    # Ordered longest-first so alternation matches Cl before C, Br before B, Se before S.
    _HETEROATOM_IN_BRACKET: ClassVar[re.Pattern] = re.compile(
        r"\["
        r"[=#/\\@+\-]*"
        r"(?:Cl|Br|Se|Si|[NOSPFI])"
        r"[^\]]*"
        r"\]"
    )

    def __post_init__(self) -> None:
        special_ids = {int(token_id) for token_id in self.special_token_ids}
        eligible = [token_id for token_id in range(self.vocab_size) if token_id not in special_ids]
        self._eligible_replacement_ids = torch.tensor(eligible, dtype=torch.long)

        if self.masking_strategy in {"span", "hetero_span"}:
            if not (0.0 < self.span_p_geom < 1.0):
                raise ValueError("span_p_geom must be in (0, 1)")
            if self.span_max_length < 1:
                raise ValueError("span_max_length must be >= 1")
            self._geom_dist: torch.distributions.Geometric | None = torch.distributions.Geometric(
                torch.tensor(self.span_p_geom)
            )
        else:
            self._geom_dist = None

        if self.masking_strategy == "hetero_span":
            if self.heteroatom_start_weight <= 0.0:
                raise ValueError("heteroatom_start_weight must be > 0")
            self._token_start_weights = self._build_token_start_weights()
        else:
            self._token_start_weights = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        ids = [
            torch.tensor(ex["input_ids"], dtype=torch.long)
            for ex in examples
            if len(ex["input_ids"]) > 0
        ]

        if not ids:
            raise ValueError("Received an empty batch after tokenization.")

        input_ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()

        special_mask = torch.zeros_like(labels, dtype=torch.bool)
        for sid in self.special_token_ids:
            special_mask |= labels.eq(sid)

        if self.masking_strategy == "standard":
            masked_indices = self._sample_standard_mask(labels, attention_mask, special_mask)
        elif self.masking_strategy in {"span", "hetero_span"}:
            masked_indices = self._sample_batch_span_mask(input_ids, attention_mask, special_mask)
        else:
            raise ValueError(f"Unknown masking_strategy: {self.masking_strategy!r}")

        labels[~masked_indices] = -100

        # 80% of selected tokens become mask tokens.
        replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[replace_with_mask] = self.mask_token_id

        # 10% become random tokens.
        replace_with_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~replace_with_mask
        )
        if replace_with_random.any():
            eligible_random_ids = self.eligible_random_token_ids(device=input_ids.device)
            random_indices = torch.randint(
                low=0,
                high=len(eligible_random_ids),
                size=labels.shape,
                device=input_ids.device,
            )
            random_words = eligible_random_ids[random_indices]
            input_ids[replace_with_random] = random_words[replace_with_random]

        # Remaining 10% stay unchanged.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _sample_standard_mask(
        self,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        special_mask: torch.Tensor,
    ) -> torch.Tensor:
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_mask, 0.0)
        probability_matrix.masked_fill_(attention_mask.eq(0), 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        if self.mlm_probability > 0.0 and not masked_indices.any():
            eligible_positions = (~special_mask & attention_mask.bool()).nonzero(as_tuple=False)
            if len(eligible_positions) > 0:
                idx = int(torch.randint(len(eligible_positions), (1,)).item())
                row_pos = eligible_positions[idx]
                row = int(row_pos[0].item())
                col = int(row_pos[1].item())
                masked_indices[row, col] = True

        return masked_indices

    def _sample_batch_span_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(batch_size):
            row_mask = self._sample_span_mask(
                input_ids_row=input_ids[i],
                attention_mask_row=attention_mask[i],
                special_mask_row=special_mask[i],
            )
            if not row_mask.any():
                eligible = (~special_mask[i] & attention_mask[i].bool()).nonzero(as_tuple=False)
                if len(eligible) > 0:
                    rand_idx = int(torch.randint(len(eligible), (1,)).item())
                    col = int(eligible[rand_idx].item())
                    row_mask[col] = True
            masked_indices[i] = row_mask
        return masked_indices

    def eligible_random_token_ids(
        self,
        device: torch.device | None = None,
    ) -> torch.Tensor:

        if len(self._eligible_replacement_ids) == 0:
            raise ValueError(
                "No eligible non-special token IDs available for MLM random replacement."
            )

        if device is None:
            return self._eligible_replacement_ids

        return self._eligible_replacement_ids.to(device)

    def _build_token_start_weights(self) -> torch.Tensor:
        """Return a (vocab_size,) float weight tensor for heteroatom-biased span starts.

        Covered heteroatom set: N, O, S, P, F, Cl, Br, I, Se, Si.
        Elements not in this set (e.g. B, Sn, As, Ge) receive weight 1.0.
        If intentional coverage of additional elements is needed, extend
        _HETEROATOM_IN_BRACKET accordingly.

        Token IDs in special_token_ids receive weight 0.0 as a defensive guard;
        the eligible-position filter in _sample_span_mask is the primary barrier.
        Tokens matching the heteroatom pattern receive weight heteroatom_start_weight.
        All other tokens receive weight 1.0.
        """
        weights = torch.ones(self.vocab_size, dtype=torch.float32)
        special_ids = set(self.special_token_ids)
        for tok_id, tok_str in self.ids_to_tokens.items():
            if tok_id in special_ids:
                weights[tok_id] = 0.0
            elif self._HETEROATOM_IN_BRACKET.search(tok_str):
                weights[tok_id] = float(self.heteroatom_start_weight)
        return weights

    def _sample_span_mask(
        self,
        input_ids_row: torch.Tensor,
        attention_mask_row: torch.Tensor,
        special_mask_row: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a span-based boolean mask for one sequence.

        Contiguous spans of APE tokens are sampled until the number of newly
        masked positions reaches round(n_eligible × mlm_probability).
        Span lengths are drawn from a Geometric(span_p_geom) distribution and
        clamped to span_max_length. For hetero_span, span-start positions are
        sampled with weights proportional to heteroatom content.

        Adjacent independent spans may produce contiguous masked runs longer than
        span_max_length — the parameter bounds individual draws, not total runs.
        On very short sequences the actual masked fraction may exceed mlm_probability
        because a single span can cover the entire budget in one draw.
        """
        seq_len = input_ids_row.size(0)
        masked = torch.zeros(seq_len, dtype=torch.bool)

        eligible_mask = (~special_mask_row) & attention_mask_row.bool()
        eligible_pos = eligible_mask.nonzero(as_tuple=False).squeeze(1)

        if len(eligible_pos) == 0:
            return masked

        n_eligible = len(eligible_pos)
        budget = max(1, round(n_eligible * self.mlm_probability))

        # Pre-sample all geometric span lengths in one vectorised call.
        max_draws = budget * 5
        assert self._geom_dist is not None
        span_lengths = (
            self._geom_dist.sample((max_draws,)).long() + 1  # shift k≥0 → k≥1
        ).clamp(max=self.span_max_length)

        if self.masking_strategy == "hetero_span" and self._token_start_weights is not None:
            tok_ids_at_eligible = input_ids_row[eligible_pos]
            pos_weights = self._token_start_weights[tok_ids_at_eligible].clone()
        else:
            pos_weights = torch.ones(n_eligible, dtype=torch.float32)

        masked_count = 0

        for draw_idx in range(max_draws):
            if masked_count >= budget:
                break
            if pos_weights.sum().item() == 0.0:
                break

            start_local = int(torch.multinomial(pos_weights, num_samples=1).item())
            start = int(eligible_pos[start_local].item())
            span_len = int(span_lengths[draw_idx].item())
            end = min(start + span_len, seq_len)

            for pos in range(start, end):
                if not attention_mask_row[pos].item() or special_mask_row[pos].item():
                    end = pos
                    break

            if end <= start:
                pos_weights[start_local] = 0.0
                continue

            new_count = int((~masked[start:end]).sum().item())
            masked[start:end] = True
            masked_count += new_count

            # Zero weights for covered eligible positions so subsequent draws
            # explore unmasked territory.
            covered = (eligible_pos >= start) & (eligible_pos < end)
            pos_weights[covered] = 0.0

        return masked


MODERNBERT_CONFIGS = {
    "base": "answerdotai/ModernBERT-base",
    "large": "answerdotai/ModernBERT-large",
}
LOCAL_MODERNBERT_PRESETS = {
    # Designed to sit below MoLFormer-XL's ~46M params for small molecular vocabularies.
    # To check parameter count before a full run:
    #   uv run python -c "
    #   from transformers import AutoConfig, AutoModelForMaskedLM
    #   from modernmolbert.train_selfies_ape_modernbert import build_modernbert_config, LOCAL_MODERNBERT_PRESETS
    #   import types
    #   args = types.SimpleNamespace(model_size='small', max_seq_length=256)
    #   config = build_modernbert_config(args, vocab_size=5000, special_ids={'pad_token':0,'bos_token':1,'eos_token':2,'unk_token':3,'mask_token':4})
    #   model = AutoModelForMaskedLM.from_config(config)
    #   print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')
    #   "
    "small": {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 768,
        "global_attn_every_n_layers": 3,
        "local_attention": 128,
    },
    # Still much smaller than official ModernBERT-base, but closer to a strong encoder.
    "medium": {
        "hidden_size": 512,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "intermediate_size": 768,
        "global_attn_every_n_layers": 3,
        "local_attention": 128,
    },
}


def build_modernbert_config(
    args: argparse.Namespace,
    vocab_size: int,
    special_ids: dict[str, int],
):
    if args.model_size in LOCAL_MODERNBERT_PRESETS:
        # Start from official base config so we preserve ModernBERT-specific fields,
        # then override only the scale-related fields.
        config = AutoConfig.from_pretrained(MODERNBERT_CONFIGS["base"])
        for key, value in LOCAL_MODERNBERT_PRESETS[args.model_size].items():
            setattr(config, key, value)
    else:
        config = AutoConfig.from_pretrained(MODERNBERT_CONFIGS[args.model_size])
    # Molecular tokenizer-specific fields.
    config.vocab_size = vocab_size
    config.pad_token_id = special_ids["pad_token"]
    config.bos_token_id = special_ids["bos_token"]
    config.eos_token_id = special_ids["eos_token"]
    # Optional context-length override.
    if args.max_seq_length is not None:
        config.max_position_embeddings = args.max_seq_length
    return config


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100

    if mask.sum() == 0:
        return {"masked_accuracy": 0.0}

    return {"masked_accuracy": float((preds[mask] == labels[mask]).mean())}


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


def write_run_metadata(
    args: argparse.Namespace,
    backend: str,
    vocab_size: int,
    special_ids: dict[str, int],
    n_params: int,
    tokenizer_stats: dict[str, float],
    tokenizer_vocab_path: Path,
    tokenizer_metadata_path: Path,
    final_eval_metrics: dict[str, float] | None = None,
    trainer_state: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(args.output_dir)

    metadata = {
        "dataset_name": args.dataset_name,
        "selfies_column": args.selfies_column,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "use_validation_split": args.use_validation_split,
        "representation": SELFIES_REPRESENTATION,
        "expected_input": (
            "SELFIES strings only. Convert SMILES before inference using a helper such "
            "as smiles_to_selfies()."
        ),
        "tokenizer_vocab_path": str(tokenizer_vocab_path),
        "tokenizer_metadata_path": str(tokenizer_metadata_path),
        "backend": backend,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "num_parameters": n_params,
        "tokenizer_stats": tokenizer_stats,
        "final_eval_metrics": final_eval_metrics,
        "trainer_state_summary": trainer_state,
        "args": vars(args),
    }

    best_checkpoint_text = ""

    if trainer_state:
        best_checkpoint_text = f"""

    ## Best checkpoint

    - Best checkpoint: `{trainer_state.get("best_model_checkpoint")}`

    - Best metric: `{trainer_state.get("best_metric")}`

    - Best global step: `{trainer_state.get("best_global_step")}`

    """

    with (output_dir / "ape_tokenizer_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""# APE ModernBERT Molecular MLM Checkpoint

This checkpoint was trained from scratch with ModernBERT for SELFIES masked language modeling.

## Representation

`{SELFIES_REPRESENTATION}`

This checkpoint expects SELFIES strings. Convert SMILES before tokenization.

## Tokenizer

This model uses `APEPreTrainedTokenizer`, loadable through `AutoTokenizer`.

Keep these files with the checkpoint:

- `tokenizer.json`
- `tokenizer_metadata.json`

## Dataset

`{args.dataset_name}`

## Model

- Parameters: {n_params / 1e6:.2f}M
- Vocabulary size: {vocab_size}
- Max sequence length: {args.max_seq_length}
- MLM probability: {args.mlm_probability}

{best_checkpoint_text}
## Loading sketch

```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("final_model")

tokenizer = AutoTokenizer.from_pretrained(
    "final_model/ape_tokenizer",
    trust_remote_code=True,
)
```
"""
    with (output_dir / "README.checkpoint.md").open("w", encoding="utf-8") as f:
        f.write(readme)


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.hf_login:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("--hf_login was set but HF_TOKEN is not available.")
        login(token=hf_token)

    args = resolve_dataset_args(args)
    backend = detect_backend(args)
    args = adjust_args_for_backend(args, backend)
    validate_args(args, backend)

    # Resolve max_seq_length from the official model config when not explicitly set.
    if args.max_seq_length is None:
        if args.model_size in LOCAL_MODERNBERT_PRESETS:
            args.max_seq_length = 256
        else:
            _tmp = AutoConfig.from_pretrained(MODERNBERT_CONFIGS[args.model_size])
            args.max_seq_length = _tmp.max_position_embeddings

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    log(f"Backend: {backend}")
    log(f"bf16={args.bf16}, fp16={args.fp16}")
    log(f"Dataset: {args.dataset_name}")
    log(f"SELFIES column: {args.selfies_column}")
    log(f"Train split: {args.train_split}")
    log(f"Validation split: {args.validation_split}")
    log(f"Use validation split: {args.use_validation_split}")

    log("Loading and validating tokenizer...")
    (
        tokenizer,
        _tokenizer_metadata,
        tokenizer_vocab_path,
        tokenizer_metadata_path,
        vocab_size,
        special_ids,
        tokenizer_stats,
    ) = load_and_validate_tokenizer(args)

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
        ids_to_tokens=dict(tokenizer.ids_to_tokens),
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
            "Check dataset, split, and SELFIES column."
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
        report_to=report_to,
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
    )

    log("Starting training...")
    log(f"Training logs will print every {args.logging_steps} steps.")
    log(f"Evaluation will run every {args.eval_steps} steps.")
    log(f"Checkpoints will be saved every {args.save_steps} steps.")
    log(f"Only the most recent {args.save_total_limit} checkpoints will be kept.")
    log(f"Intermediate checkpoints: {output_dir}/checkpoint-*")
    log(f"Final model will be saved to: {output_dir}/final_model")
    train_result = trainer.train()

    print("Saving final model...")

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))

    copy_tokenizer_artifacts(
        vocab_path=tokenizer_vocab_path,
        metadata_path=tokenizer_metadata_path,
        output_dir=output_dir,
        final_model_dir=final_dir,
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

    write_run_metadata(
        args=args,
        backend=backend,
        vocab_size=vocab_size,
        special_ids=special_ids,
        n_params=n_params,
        tokenizer_stats=tokenizer_stats,
        tokenizer_vocab_path=tokenizer_vocab_path,
        tokenizer_metadata_path=tokenizer_metadata_path,
        final_eval_metrics={k: float(v) for k, v in eval_metrics.items()},
        trainer_state=trainer_state_summary,
    )

    print("Done.")
    print(f"Final model: {final_dir}")
    print(f"Tokenizer (copied): {final_dir / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
