#!/usr/bin/env python3
"""Train a ModernBERT masked-language model for SELFIES molecular strings.

Model training requires an existing, vetted tokenizer vocabulary and metadata.
Tokenizer training is intentionally a separate command:

    python -m modernmolbert.train_ape_tokenizer
"""

import argparse
import hashlib
import time
import json
import math
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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

from modernmolbert.ape_tokenizer import APETokenizer
from modernmolbert.utils import (
    SELFIES_REPRESENTATION,
    assert_metadata_representation,
    compute_tokenization_stats,
    copy_tokenizer_artifacts,
    default_selfies_tokenizer_path,
    encode_sequence,
    file_sha256,
    get_streaming_dataset,
    load_tokenizer_metadata,
    metadata_path_for_vocab,
    normalize_sequence,
    resolve_special_ids,
    tokenizer_vocab_size,
    validate_selfies_sample_shape,
)

DATASET_NAME = "mikemayuare/PubChem10M_SMILES_SELFIES"


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
        choices=["base", "large"],
        default="base",
        help="Use the official ModernBERT-base or ModernBERT-large architecture config.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override max sequence length (default: use official model context length).",
    )

    # MLM
    parser.add_argument("--mlm_probability", type=float, default=0.30)

    # Training
    parser.add_argument("--max_steps", type=int, default=150_000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Runtime
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--device_backend", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_eval_batches", type=int, default=20)
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

    return parser.parse_args()


def log_step(message: str) -> None:

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
    if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.val_split_mod < 2:
        raise ValueError("val_split_mod must be >= 2")
    if not 0 <= args.val_split_bucket < args.val_split_mod:
        raise ValueError("val_split_bucket must satisfy 0 <= bucket < val_split_mod")
    if backend == "cuda" and not torch.cuda.is_available():
        raise ValueError("device_backend=cuda requested but CUDA is not available")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device_backend=mps requested but MPS is not available")


def adjust_args_for_backend(
    args: argparse.Namespace, backend: str
) -> argparse.Namespace:
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
        args.save_steps = min(args.save_steps, 100)
        args.tokenizer_validation_samples = min(args.tokenizer_validation_samples, 200)

    return args


def _sample_train_partition_sequences(args: argparse.Namespace, n: int) -> list[str]:
    ds = get_streaming_dataset(
        args.dataset_name,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
    )

    rows: list[str] = []
    for row in ds:
        seq = normalize_sequence(row, SELFIES_REPRESENTATION)
        if seq is None:
            continue
        if sequence_bucket(seq, args.val_split_mod) == args.val_split_bucket:
            continue
        rows.append(seq)
        if len(rows) >= n:
            break

    return rows


def load_and_validate_tokenizer(
    args: argparse.Namespace,
) -> tuple[
    APETokenizer, dict[str, Any], Path, Path, int, dict[str, int], dict[str, float]
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
    assert_metadata_representation(
        metadata, expected_representation=SELFIES_REPRESENTATION
    )

    recorded_sha = str(metadata.get("tokenizer_sha256", ""))
    actual_sha = file_sha256(vocab_path)
    if recorded_sha and recorded_sha != actual_sha:
        raise ValueError(
            "Tokenizer hash mismatch between file and metadata. "
            f"metadata={recorded_sha}, file={actual_sha}"
        )

    tokenizer = APETokenizer()
    tokenizer.load_vocabulary(str(vocab_path))

    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size < 100:
        raise ValueError(f"Suspiciously small tokenizer vocabulary: {vocab_size}")

    special_ids = resolve_special_ids(tokenizer)

    validation_sequences = _sample_train_partition_sequences(
        args, n=args.tokenizer_validation_samples
    )
    validate_selfies_sample_shape(validation_sequences)

    ethanol_encoded = encode_sequence(tokenizer, "[C][C][O]", args.max_seq_length)[
        "input_ids"
    ]
    non_special_ethanol = [
        x for x in ethanol_encoded if x not in set(special_ids.values())
    ]
    if not non_special_ethanol:
        raise ValueError("Tokenizer produced no usable SELFIES tokens for [C][C][O]")
    unk_ethanol = sum(1 for x in non_special_ethanol if x == special_ids["unk_token"])
    if unk_ethanol / len(non_special_ethanol) > 0.05:
        raise ValueError("Tokenizer encodes [C][C][O] mostly as <unk>")

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
            "Too many sequences are mostly unknown tokens: "
            f"{stats['mostly_unknown_rate']:.4f}"
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
    args: argparse.Namespace, tokenizer: APETokenizer
) -> IterableDataset:
    ds = get_streaming_dataset(
        args.dataset_name,
        seed=args.seed + 100,
        buffer_size=args.shuffle_buffer_size,
    )

    def keep_train(row: dict[str, Any]) -> bool:
        seq = normalize_sequence(row, SELFIES_REPRESENTATION)
        if seq is None:
            return False
        return sequence_bucket(seq, args.val_split_mod) != args.val_split_bucket

    ds = ds.filter(keep_train)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_sequence(row, SELFIES_REPRESENTATION)
        assert seq is not None
        return encode_sequence(tokenizer, seq, args.max_seq_length)

    return ds.map(preprocess)


def make_eval_dataset(args: argparse.Namespace, tokenizer: APETokenizer) -> Dataset:
    n_eval = args.eval_size
    if args.max_eval_batches > 0:
        n_eval = min(
            n_eval,
            args.max_eval_batches * args.per_device_eval_batch_size,
        )

    ds = get_streaming_dataset(
        args.dataset_name,
        seed=args.seed + 200,
        buffer_size=args.shuffle_buffer_size,
    )

    rows: list[dict[str, list[int]]] = []
    pbar = tqdm(total=n_eval, desc="Building finite validation set")

    for row in ds:
        seq = normalize_sequence(row, SELFIES_REPRESENTATION)
        if seq is None:
            continue
        if sequence_bucket(seq, args.val_split_mod) != args.val_split_bucket:
            continue

        rows.append(encode_sequence(tokenizer, seq, args.max_seq_length))
        pbar.update(1)
        if len(rows) >= n_eval:
            break

    pbar.close()

    if not rows:
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

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_mask = torch.zeros_like(labels, dtype=torch.bool)
        for sid in self.special_token_ids:
            special_mask |= labels.eq(sid)

        probability_matrix.masked_fill_(special_mask, 0.0)
        probability_matrix.masked_fill_(attention_mask.eq(0), 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% of selected tokens become mask tokens.
        replace_with_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[replace_with_mask] = self.mask_token_id

        # 10% become random tokens.
        replace_with_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~replace_with_mask
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[replace_with_random] = random_words[replace_with_random]

        # Remaining 10% stay unchanged.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


MODERNBERT_CONFIGS = {
    "base": "answerdotai/ModernBERT-base",
    "large": "answerdotai/ModernBERT-large",
}


def build_modernbert_config(
    args: argparse.Namespace,
    vocab_size: int,
    special_ids: dict[str, int],
):
    config = AutoConfig.from_pretrained(MODERNBERT_CONFIGS[args.model_size])
    # Molecular tokenizer-specific fields only.
    config.vocab_size = vocab_size
    config.pad_token_id = special_ids["pad_token"]
    config.bos_token_id = special_ids["bos_token"]
    config.eos_token_id = special_ids["eos_token"]
    # Optional context-length override (useful for MPS/debug runs).
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


def write_run_metadata(
    args: argparse.Namespace,
    backend: str,
    vocab_size: int,
    special_ids: dict[str, int],
    n_params: int,
    tokenizer_stats: dict[str, float],
    tokenizer_vocab_path: Path,
    tokenizer_metadata_path: Path,
) -> None:
    output_dir = Path(args.output_dir)

    metadata = {
        "dataset_name": args.dataset_name,
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
        "args": vars(args),
    }

    with (output_dir / "ape_tokenizer_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""# APE ModernBERT Molecular MLM Checkpoint

This checkpoint was trained from scratch with ModernBERT for SELFIES masked language modeling.

## Representation

`{SELFIES_REPRESENTATION}`

This checkpoint expects SELFIES strings. Convert SMILES before tokenization.

## Tokenizer

This model uses `APETokenizer` from `modernmolbert.ape_tokenizer`.

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

## Loading sketch

```python
from transformers import AutoModelForMaskedLM
from modernmolbert.ape_tokenizer import APETokenizer

model = AutoModelForMaskedLM.from_pretrained("final_model")

tokenizer = APETokenizer()
tokenizer.load_vocabulary("final_model/tokenizer.json")
```
"""
    with (output_dir / "README.checkpoint.md").open("w", encoding="utf-8") as f:
        f.write(readme)


def main() -> None:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    args = parse_args()
    backend = detect_backend(args)
    validate_args(args, backend)
    args = adjust_args_for_backend(args, backend)

    # Resolve max_seq_length from the official model config when not explicitly set.
    if args.max_seq_length is None:
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

    print(f"Backend: {backend}")
    print(f"bf16={args.bf16}, fp16={args.fp16}")

    print("Loading and validating tokenizer...")
    (
        tokenizer,
        _tokenizer_metadata,
        tokenizer_vocab_path,
        tokenizer_metadata_path,
        vocab_size,
        special_ids,
        tokenizer_stats,
    ) = load_and_validate_tokenizer(args)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Special token IDs: {special_ids}")
    print(
        f"Tokenizer stats: unk_rate={tokenizer_stats['unk_rate']:.6f}, "
        f"truncation_rate={tokenizer_stats['truncation_rate']:.6f}"
    )
    if tokenizer_stats["truncation_rate"] > args.truncation_warn_threshold:
        print(
            "Warning: truncation rate is high "
            f"({tokenizer_stats['truncation_rate']:.4f} > {args.truncation_warn_threshold:.4f})"
        )

    print("Building datasets...")
    train_dataset = make_train_iterable_dataset(args, tokenizer)
    eval_dataset = make_eval_dataset(args, tokenizer)

    log_step("Building ModernBERT config...")

    config = build_modernbert_config(args, vocab_size, special_ids)

    log_step("ModernBERT config built.")

    log_step(
        f"Config: ModernBERT-{args.model_size}, "
        f"vocab_size={config.vocab_size}, "
        f"hidden_size={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, "
        f"max_position_embeddings={config.max_position_embeddings}"
    )

    log_step(
        "Initializing ModernBERT model weights. This can take a while on MPS/CPU..."
    )

    model = AutoModelForMaskedLM.from_config(config)

    log_step("Model object created. Counting parameters...")

    n_params = sum(p.numel() for p in model.parameters())

    log_step(f"Model parameters: {n_params / 1e6:.2f}M")

    collator = MolecularMLMCollator(
        pad_token_id=special_ids["pad_token"],
        mask_token_id=special_ids["mask_token"],
        vocab_size=vocab_size,
        mlm_probability=args.mlm_probability,
        special_token_ids=list(special_ids.values()),
    )

    report_to = [] if args.report_to == "none" else [args.report_to]

    print("Testing one training batch before Trainer...", flush=True)

    one = []

    it = iter(train_dataset)

    for _ in range(args.per_device_train_batch_size):
        one.append(next(it))

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
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        prediction_loss_only=not args.compute_masked_accuracy,
        report_to=report_to,
        load_best_model_at_end=False,
    )
    setattr(training_args, "overwrite_output_dir", True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if args.compute_masked_accuracy else None,
    )

    print("Starting training...")
    train_result = trainer.train()

    print("Saving final model...")
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    model.config.save_pretrained(str(final_dir))

    copy_tokenizer_artifacts(
        vocab_path=tokenizer_vocab_path,
        metadata_path=tokenizer_metadata_path,
        output_dir=output_dir,
        final_model_dir=final_dir,
    )

    metrics = train_result.metrics
    metrics["train_samples_streaming"] = 1.0
    metrics["num_parameters"] = float(n_params)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    write_run_metadata(
        args=args,
        backend=backend,
        vocab_size=vocab_size,
        special_ids=special_ids,
        n_params=n_params,
        tokenizer_stats=tokenizer_stats,
        tokenizer_vocab_path=tokenizer_vocab_path,
        tokenizer_metadata_path=tokenizer_metadata_path,
    )

    print("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        try:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["eval_perplexity"] = float("inf")

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("Done.")
    print(f"Final model: {final_dir}")
    print(f"Tokenizer (copied): {final_dir / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
