#!/usr/bin/env python3
"""
Train a ModernBERT masked-language model on PubChem10M SMILES or SELFIES
using the APE tokenizer.

This script is designed for two stages:

1. Mac MPS smoke tests:
   - Validate dataset loading, APE tokenization, ModernBERT forward/backward,
     checkpointing, and reloadability.
   - Use small model and small batch sizes.

2. CUDA training:
   - Use the same script with larger settings.
   - Enable bf16 on supported NVIDIA GPUs.

Dataset:
    mikemayuare/PubChem10M_SMILES_SELFIES

Important:
    If --representation SELFIES is used, the trained model expects SELFIES at
    inference time. Convert SMILES to SELFIES before tokenization, for example
    with selfies.encoder(smiles).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    ModernBertConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from ape_tokenizer import APETokenizer
except ImportError as exc:
    raise ImportError(
        "Could not import APETokenizer. Install it with:\n"
        "  pip install git+https://github.com/mikemayuare/apetokenizer.git"
    ) from exc


DATASET_NAME = "mikemayuare/PubChem10M_SMILES_SELFIES"

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ModernBERT MLM on PubChem10M SMILES/SELFIES with APE tokenization."
    )

    # Paths
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer_vocab_path",
        type=str,
        default=None,
        help="Existing APE vocab JSON. If omitted, defaults to output_dir/ape_<representation>_vocab.json.",
    )

    # Dataset
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--representation", type=str, choices=["SELFIES", "SMILES"], default="SELFIES"
    )
    parser.add_argument("--tokenizer_train_size", type=int, default=2_000_000)
    parser.add_argument("--eval_size", type=int, default=100_000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)

    # Tokenizer
    parser.add_argument("--max_vocab_size", type=int, default=5000)
    parser.add_argument("--min_freq_for_merge", type=int, default=2000)
    parser.add_argument("--train_tokenizer", action="store_true", default=True)
    parser.add_argument(
        "--no_train_tokenizer", action="store_false", dest="train_tokenizer"
    )

    # Model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--global_attn_every_n_layers", type=int, default=3)
    parser.add_argument("--local_attention", type=int, default=64)

    # MLM
    parser.add_argument("--mlm_probability", type=float, default=0.30)

    # Training
    parser.add_argument("--max_steps", type=int, default=150_000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
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
    parser.add_argument(
        "--compute_masked_accuracy",
        action="store_true",
        help="Compute masked-token accuracy during eval. Off by default to avoid large MLM logits in memory.",
    )
    parser.add_argument("--debug", action="store_true", help="Run a tiny smoke test.")

    return parser.parse_args()


def detect_backend(args: argparse.Namespace) -> str:
    if args.device_backend != "auto":
        return args.device_backend
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def adjust_args_for_backend(
    args: argparse.Namespace, backend: str
) -> argparse.Namespace:
    # MPS and CPU are safest in full precision for this workflow.
    if backend in {"mps", "cpu"}:
        args.bf16 = False
        args.fp16 = False
        args.num_workers = min(args.num_workers, 2)

    if args.debug:
        args.tokenizer_train_size = min(args.tokenizer_train_size, 10_000)
        args.eval_size = min(args.eval_size, 1_000)
        args.max_steps = min(args.max_steps, 200)
        args.logging_steps = min(args.logging_steps, 10)
        args.eval_steps = min(args.eval_steps, 50)
        args.save_steps = min(args.save_steps, 100)

    return args


def normalize_sequence(example: dict[str, Any], representation: str) -> str | None:
    seq = example.get(representation)
    if seq is None:
        return None
    seq = str(seq).strip()
    return seq if seq else None


def get_streaming_dataset(
    dataset_name: str, seed: int, buffer_size: int
) -> IterableDataset:
    ds = load_dataset(dataset_name, split="train", streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer_size)


def collect_corpus_for_tokenizer(
    dataset_name: str,
    representation: str,
    n: int,
    seed: int,
    buffer_size: int,
) -> list[str]:
    ds = get_streaming_dataset(dataset_name, seed=seed, buffer_size=buffer_size)
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


def train_or_load_ape_tokenizer(args: argparse.Namespace) -> APETokenizer:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = (
        Path(args.tokenizer_vocab_path)
        if args.tokenizer_vocab_path is not None
        else output_dir / f"ape_{args.representation.lower()}_vocab.json"
    )

    tokenizer = APETokenizer()

    if args.train_tokenizer or not vocab_path.exists():
        corpus = collect_corpus_for_tokenizer(
            dataset_name=args.dataset_name,
            representation=args.representation,
            n=args.tokenizer_train_size,
            seed=args.seed,
            buffer_size=args.shuffle_buffer_size,
        )
        tokenizer.train(
            corpus,
            max_vocab_size=args.max_vocab_size,
            min_freq_for_merge=args.min_freq_for_merge,
            save_checkpoint=False,
        )
        tokenizer.save_vocabulary(str(vocab_path))
    else:
        tokenizer.load_vocabulary(str(vocab_path))

    return tokenizer


def tokenizer_vocab_size(tokenizer: APETokenizer) -> int:
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
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
        except Exception:
            attr_name = name + "_id"
            if hasattr(tokenizer, attr_name):
                ids[name] = int(getattr(tokenizer, attr_name))
            else:
                raise RuntimeError(
                    f"Could not resolve ID for special token {token!r}. "
                    "Check APETokenizer special-token names."
                )
    return ids


def encode_sequence(
    tokenizer: APETokenizer, seq: str, max_seq_length: int
) -> dict[str, list[int]]:
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
        "input_ids": list(map(int, input_ids[:max_seq_length])),
        "attention_mask": list(map(int, attention_mask[:max_seq_length])),
    }


def make_train_iterable_dataset(
    args: argparse.Namespace, tokenizer: APETokenizer
) -> IterableDataset:
    ds = get_streaming_dataset(
        args.dataset_name,
        seed=args.seed + 100,
        buffer_size=args.shuffle_buffer_size,
    )
    ds = ds.filter(lambda row: normalize_sequence(row, args.representation) is not None)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_sequence(row, args.representation)
        assert seq is not None
        return encode_sequence(tokenizer, seq, args.max_seq_length)

    return ds.map(preprocess)


def make_eval_dataset(args: argparse.Namespace, tokenizer: APETokenizer) -> Dataset:
    ds = get_streaming_dataset(
        args.dataset_name,
        seed=args.seed + 200,
        buffer_size=args.shuffle_buffer_size,
    )

    rows: list[dict[str, list[int]]] = []
    pbar = tqdm(total=args.eval_size, desc="Building finite validation set")

    for row in ds:
        seq = normalize_sequence(row, args.representation)
        if seq is None:
            continue
        rows.append(encode_sequence(tokenizer, seq, args.max_seq_length))
        pbar.update(1)
        if len(rows) >= args.eval_size:
            break

    pbar.close()

    if not rows:
        raise RuntimeError("Validation set is empty. Check dataset column names.")

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


def build_modernbert_config(
    args: argparse.Namespace,
    vocab_size: int,
    special_ids: dict[str, int],
) -> ModernBertConfig:
    return ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_length,
        global_attn_every_n_layers=args.global_attn_every_n_layers,
        local_attention=args.local_attention,
        pad_token_id=special_ids["pad_token"],
        bos_token_id=special_ids["bos_token"],
        eos_token_id=special_ids["eos_token"],
    )


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
) -> None:
    output_dir = Path(args.output_dir)

    metadata = {
        "dataset_name": args.dataset_name,
        "representation": args.representation,
        "expected_input": (
            "SELFIES strings. Convert SMILES with selfies.encoder() before tokenization."
            if args.representation == "SELFIES"
            else "SMILES strings."
        ),
        "ape_tokenizer_vocab": f"ape_{args.representation.lower()}_vocab.json",
        "backend": backend,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "num_parameters": n_params,
        "args": vars(args),
    }

    with (output_dir / "ape_tokenizer_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""# APE ModernBERT Molecular MLM Checkpoint

This checkpoint was trained from scratch with ModernBERT for masked language modeling.

## Representation

`{args.representation}`

{metadata["expected_input"]}

## Tokenizer

This model uses `APETokenizer`, not a standard Hugging Face tokenizer.

Keep this file paired with the checkpoint:

`ape_{args.representation.lower()}_vocab.json`

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
from ape_tokenizer import APETokenizer

model = AutoModelForMaskedLM.from_pretrained("final_model")

tokenizer = APETokenizer()
tokenizer.load_vocabulary("ape_{args.representation.lower()}_vocab.json")
```
"""
    with (output_dir / "README.checkpoint.md").open("w", encoding="utf-8") as f:
        f.write(readme)


def main() -> None:
    args = parse_args()
    backend = detect_backend(args)
    args = adjust_args_for_backend(args, backend)

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

    print("Training/loading APE tokenizer...")
    tokenizer = train_or_load_ape_tokenizer(args)

    vocab_size = tokenizer_vocab_size(tokenizer)
    assert vocab_size > 100, f"Suspiciously small vocab size: {vocab_size}"

    special_ids = resolve_special_ids(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special token IDs: {special_ids}")

    print("Building datasets...")
    train_dataset = make_train_iterable_dataset(args, tokenizer)
    eval_dataset = make_eval_dataset(args, tokenizer)

    print("Building ModernBERT model...")
    config = build_modernbert_config(args, vocab_size, special_ids)
    model = AutoModelForMaskedLM.from_config(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    collator = MolecularMLMCollator(
        pad_token_id=special_ids["pad_token"],
        mask_token_id=special_ids["mask_token"],
        vocab_size=vocab_size,
        mlm_probability=args.mlm_probability,
        special_token_ids=list(special_ids.values()),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
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
        report_to=["tensorboard"],
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
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

    metrics = train_result.metrics
    metrics["train_samples_streaming"] = "streaming"
    metrics["num_parameters"] = n_params
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    write_run_metadata(args, backend, vocab_size, special_ids, n_params)

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
    print(f"APE vocab: {output_dir / f'ape_{args.representation.lower()}_vocab.json'}")


if __name__ == "__main__":
    main()
