"""Fixed-mask head-to-head evaluation of best checkpoints per masking group.

ANALYSIS
--------
The sweep trains models under two masking strategies (standard, span) at three
masking probabilities (0.15, 0.20, 0.25) and three learning rates.  Each run
reports a best_metric (eval loss) measured under its *own* objective, so the
numbers are not directly comparable: a span model trained at 20% masking is
measured at 20% span loss, while a standard model at 15% is measured at 15%
standard loss.

This script makes the comparison apples-to-apples:

  1. For each (masking_strategy, mlm_probability) group it selects the run with
     the lowest training-objective best_metric — i.e. the best lr within that
     group.  This yields six checkpoints (standard@0.15, standard@0.20,
     standard@0.25, span@0.15, span@0.20, span@0.25).

  2. It pre-generates one fixed masked validation dataset with standard masking
     at 15% probability, seeded so every model sees the *exact same* corrupted
     inputs and labels.  The masking is frozen to disk as a .pt tensor cache;
     the tensor fingerprint is recorded in the manifest for reproducibility.

  3. Every checkpoint is evaluated on that frozen dataset.  The resulting
     fixed_eval_loss values are on a common scale and can be compared directly.

The key question answered: does any span configuration match or beat standard
masking at 15% when both are scored on the *same* task?  The span training loss
was lower on its own objective (harder masking), but the fixed eval reveals
whether that transfers.

SELECTION CRITERION
-------------------
Groups are defined by the pair (run_args["masking_strategy"],
run_args["mlm_probability"]).  Within each group the run with the lowest
trainer_state["best_metric"] is selected.  Runs without trainer_state.json
(e.g. incomplete or aborted runs) are skipped automatically.

FIXED EVAL DESIGN
-----------------
- Masking strategy : standard
- MLM probability  : 0.15 (--fixed_mlm_probability)
- RNG seed         : 42   (--fixed_seed)
- Batches frozen before any model is loaded; all six models see identical masks.

OUTPUT FILES (written to --output_dir, defaults to --sweep_root)
----------------------------------------------------------------------
fixed_eval_per_prob.log          — full run log (also printed to stdout)
fixed_eval_valid_full.pt         — frozen masked dataset tensors (all examples)
fixed_eval_valid_full.pt         — frozen masked dataset tensors (4096 examples)
fixed_eval_per_prob_results.csv  — one row per (eval_set, run) with metrics
fixed_eval_per_prob_results.json — same data as JSON
fixed_eval_per_prob_manifest.json — run metadata, SHAs, dataset fingerprints

USAGE
-----
Basic:
    python scripts/fixed_eval_best_models.py

Different sweep root:
    python scripts/fixed_eval_best_models.py \\
        --sweep_root runs/my_sweep \\
        --valid_parquet data/pretrain/chembl36_selfies/valid/valid.parquet

Smoke test (fast, small subset):
    python scripts/fixed_eval_best_models.py --limit 128

Write outputs to a separate directory:
    python scripts/fixed_eval_best_models.py --output_dir results/fixed_eval

REQUIREMENTS
------------
- Each run directory must contain trainer_state.json and run_args.json.
- final_model/ape_tokenizer/vocab.json and final_model/config.json must be
  identical (same SHA-256) across all selected checkpoints — the script
  verifies this before evaluating.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from modernmolbert.collator import MolecularMLMCollator
from modernmolbert.utils import encode_sequence, resolve_special_ids


log = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    log.info("Log file: %s", log_path)


SWEEP_ROOT = Path("runs/chembl36_small_mask_mlm_lr_sweep")
VALID_PARQUET = Path("data/pretrain/chembl36_selfies/valid/valid.parquet")


@dataclass(frozen=True)
class SelectedRun:
    label: str  # e.g. "span@0.20"
    run_name: str  # directory name
    run_dir: Path
    best_checkpoint: Path
    trained_masking_strategy: str
    trained_mlm_probability: float
    learning_rate: float
    training_best_metric: float  # loss on the run's own objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate best-per-(strategy, mlm_probability) checkpoints on a "
            "single fixed masked validation task for apples-to-apples comparison."
        )
    )
    parser.add_argument("--sweep_root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--valid_parquet", type=Path, default=VALID_PARQUET)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fixed_seed", type=int, default=42)
    parser.add_argument(
        "--fixed_masking_strategy",
        choices=["standard"],
        default="standard",
        help="Masking strategy used to build the frozen eval dataset.",
    )
    parser.add_argument(
        "--fixed_mlm_probability",
        type=float,
        default=0.15,
        help="Masking probability used to build the frozen eval dataset.",
    )
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--expected_vocab_size", type=int, default=631)
    parser.add_argument("--expected_max_position_embeddings", type=int, default=128)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke mode: cap each eval set to this many examples.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bf16 autocast when CUDA is available and supported.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_best_per_group(sweep_root: Path) -> list[SelectedRun]:
    """Return the best checkpoint for each (masking_strategy, mlm_probability) group.

    Iterates every subdirectory of sweep_root.  Subdirectories missing
    trainer_state.json or run_args.json are silently skipped.  Within each
    group the run with the lowest best_metric (training-objective eval loss) is
    selected.  Runs are returned sorted by (masking_strategy, mlm_probability).
    """
    # group_key -> (best_metric, SelectedRun)
    best: dict[tuple[str, float], tuple[float, SelectedRun]] = {}

    for subdir in sorted(sweep_root.iterdir()):
        if not subdir.is_dir():
            continue
        state_path = subdir / "trainer_state.json"
        args_path = subdir / "run_args.json"
        if not state_path.exists() or not args_path.exists():
            log.info("  Skipping %s (missing trainer_state.json or run_args.json)", subdir.name)
            continue

        state = load_json(state_path)
        run_args = load_json(args_path)

        best_metric = state.get("best_metric")
        best_ckpt_str = state.get("best_model_checkpoint")
        if best_metric is None or best_ckpt_str is None:
            log.info(
                "  Skipping %s (trainer_state missing best_metric/best_model_checkpoint)",
                subdir.name,
            )
            continue

        strategy = str(run_args["masking_strategy"])
        mlm_prob = float(run_args["mlm_probability"])
        lr = float(run_args["learning_rate"])
        best_checkpoint = Path(best_ckpt_str)
        if not best_checkpoint.exists():
            log.warning("  Skipping %s (checkpoint not found: %s)", subdir.name, best_checkpoint)
            continue

        group_key = (strategy, mlm_prob)
        label = f"{strategy}@{mlm_prob:.2f}"
        run = SelectedRun(
            label=label,
            run_name=subdir.name,
            run_dir=subdir,
            best_checkpoint=best_checkpoint,
            trained_masking_strategy=strategy,
            trained_mlm_probability=mlm_prob,
            learning_rate=lr,
            training_best_metric=float(best_metric),
        )

        if group_key not in best or float(best_metric) < best[group_key][0]:
            best[group_key] = (float(best_metric), run)

    if not best:
        raise RuntimeError(f"No valid runs found under {sweep_root}")

    return [
        run
        for _, run in sorted(
            best.values(),
            key=lambda kv: (kv[1].trained_masking_strategy, kv[1].trained_mlm_probability),
        )
    ]


def assert_compatible_runs(
    runs: list[SelectedRun],
    *,
    expected_vocab_size: int,
    expected_max_position_embeddings: int,
) -> dict[str, str]:
    if not runs:
        raise ValueError("No selected runs to compare.")

    ref_vocab_sha: str | None = None
    ref_tokenizer_metadata_sha: str | None = None
    ref_config_sha: str | None = None

    for run in runs:
        final_model = run.run_dir / "final_model"
        tokenizer_vocab = final_model / "ape_tokenizer" / "vocab.json"
        tokenizer_metadata = final_model / "tokenizer_metadata.json"
        config_path = final_model / "config.json"
        for path in [tokenizer_vocab, tokenizer_metadata, config_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required metadata for {run.run_name}: {path}")

        config = load_json(config_path)
        if int(config.get("vocab_size", -1)) != expected_vocab_size:
            raise ValueError(
                f"{run.run_name} has vocab_size={config.get('vocab_size')}; "
                f"expected {expected_vocab_size}."
            )
        if int(config.get("max_position_embeddings", -1)) != expected_max_position_embeddings:
            raise ValueError(
                f"{run.run_name} has max_position_embeddings="
                f"{config.get('max_position_embeddings')}; "
                f"expected {expected_max_position_embeddings}."
            )

        vocab_sha = sha256_file(tokenizer_vocab)
        metadata_sha = sha256_file(tokenizer_metadata)
        config_sha = sha256_file(config_path)
        ref_vocab_sha = vocab_sha if ref_vocab_sha is None else ref_vocab_sha
        ref_tokenizer_metadata_sha = (
            metadata_sha if ref_tokenizer_metadata_sha is None else ref_tokenizer_metadata_sha
        )
        ref_config_sha = config_sha if ref_config_sha is None else ref_config_sha

        if vocab_sha != ref_vocab_sha:
            raise ValueError(f"Tokenizer vocab SHA mismatch for {run.run_name}.")
        if metadata_sha != ref_tokenizer_metadata_sha:
            raise ValueError(f"Tokenizer metadata SHA mismatch for {run.run_name}.")
        if config_sha != ref_config_sha:
            raise ValueError(f"Config SHA mismatch for {run.run_name}.")

    assert ref_vocab_sha is not None
    assert ref_tokenizer_metadata_sha is not None
    assert ref_config_sha is not None
    return {
        "tokenizer_vocab_sha256": ref_vocab_sha,
        "tokenizer_metadata_sha256": ref_tokenizer_metadata_sha,
        "config_sha256": ref_config_sha,
    }


def load_tokenizer(tokenizer_dir: Path):
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_dir}")
    return AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)


def load_valid_full(valid_parquet: Path, *, limit: int | None) -> list[str]:
    if not valid_parquet.exists():
        raise FileNotFoundError(f"Missing validation parquet: {valid_parquet}")
    frame = pd.read_parquet(valid_parquet, columns=["selfies"])
    seqs = [str(value).strip() for value in frame["selfies"] if str(value).strip()]
    return seqs[:limit] if limit is not None else seqs


def load_valid_train_matched(
    valid_parquet: Path,
    *,
    n_examples: int,
    seed: int,
    shuffle_buffer_size: int,
) -> list[str]:
    ds = load_dataset(
        "parquet",
        data_files={"valid": [str(valid_parquet)]},
        split="valid",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    seqs: list[str] = []
    for row in ds:
        seq = str(row.get("selfies", "")).strip()
        if not seq:
            continue
        seqs.append(seq)
        if len(seqs) >= n_examples:
            break
    if len(seqs) < n_examples:
        raise RuntimeError(f"Only collected {len(seqs)} validation rows; requested {n_examples}.")
    return seqs


def encode_and_pad_sequences(
    sequences: list[str],
    tokenizer: Any,
    *,
    max_seq_length: int,
    pad_token_id: int,
) -> list[dict[str, list[int]]]:
    examples: list[dict[str, list[int]]] = []
    for seq in tqdm(sequences, desc="Tokenizing validation SELFIES", unit="mol"):
        encoded = encode_sequence(tokenizer, seq, max_seq_length)
        input_ids = [int(x) for x in encoded["input_ids"]]
        if len(input_ids) > max_seq_length:
            raise ValueError("Tokenized sequence exceeded max_seq_length after truncation.")
        pad_len = max_seq_length - len(input_ids)
        examples.append({"input_ids": input_ids + [pad_token_id] * pad_len})
    return examples


def tensor_fingerprint(tensors: dict[str, torch.Tensor], keys: list[str]) -> str:
    h = hashlib.sha256()
    for key in keys:
        tensor = tensors[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def build_fixed_masked_dataset(
    examples: list[dict[str, list[int]]],
    *,
    batch_size: int,
    seed: int,
    pad_token_id: int,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: list[int],
    ids_to_tokens: dict[int, str],
    mlm_probability: float,
    masking_strategy: str,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot build a fixed eval dataset from zero examples.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    collator = MolecularMLMCollator(
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        mlm_probability=mlm_probability,
        special_token_ids=special_token_ids,
        masking_strategy=masking_strategy,
        ids_to_tokens=ids_to_tokens,
    )

    input_batches: list[torch.Tensor] = []
    attention_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    original_batches: list[torch.Tensor] = []

    for start in tqdm(range(0, len(examples), batch_size), desc="Freezing masks", unit="batch"):
        features = examples[start : start + batch_size]
        original = torch.tensor([ex["input_ids"] for ex in features], dtype=torch.long)
        batch = collator(features)
        input_batches.append(batch["input_ids"].cpu())
        attention_batches.append(batch["attention_mask"].cpu())
        label_batches.append(batch["labels"].cpu())
        original_batches.append(original)

    tensors = {
        "input_ids": torch.cat(input_batches, dim=0).to(torch.long),
        "attention_mask": torch.cat(attention_batches, dim=0).to(torch.long),
        "labels": torch.cat(label_batches, dim=0).to(torch.long),
        "original_input_ids": torch.cat(original_batches, dim=0).to(torch.long),
    }

    masked = tensors["labels"] != -100
    attention = tensors["attention_mask"].bool()
    original = tensors["original_input_ids"]
    special = torch.zeros_like(masked)
    for token_id in special_token_ids:
        special |= original.eq(int(token_id))
    eligible = attention & ~special

    if (masked & ~eligible).any():
        raise ValueError("Fixed labels include padding or special-token positions.")

    masked_tokens = int(masked.sum().item())
    eligible_tokens = int(eligible.sum().item())
    if masked_tokens == 0:
        raise ValueError("Fixed eval dataset has zero masked tokens.")

    fingerprint = tensor_fingerprint(tensors, ["input_ids", "labels"])
    return {
        "tensors": tensors,
        "num_examples": int(tensors["input_ids"].shape[0]),
        "seq_length": int(tensors["input_ids"].shape[1]),
        "masked_tokens": masked_tokens,
        "eligible_tokens": eligible_tokens,
        "actual_mask_fraction": float(masked_tokens / max(1, eligible_tokens)),
        "fingerprint_sha256": fingerprint,
    }


def save_fixed_dataset(path: Path, fixed: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tensors": fixed["tensors"],
            "metadata": {key: value for key, value in fixed.items() if key != "tensors"},
        },
        path,
    )


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device, *, bf16: bool):
    enabled = bool(bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_checkpoint(
    run: SelectedRun,
    fixed: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> dict[str, float]:
    model = AutoModelForMaskedLM.from_pretrained(str(run.best_checkpoint))
    model.to(device)
    model.eval()

    tensors = fixed["tensors"]
    total_loss = 0.0
    total_masked = 0
    total_correct = 0

    with torch.inference_mode():
        for start in tqdm(
            range(0, fixed["num_examples"], batch_size),
            desc=f"Evaluating {run.label}",
            unit="batch",
        ):
            end = min(start + batch_size, fixed["num_examples"])
            input_ids = tensors["input_ids"][start:end].to(device)
            attention_mask = tensors["attention_mask"][start:end].to(device)
            labels = tensors["labels"][start:end].to(device)
            mask = labels.ne(-100)
            masked_count = int(mask.sum().item())
            if masked_count == 0:
                continue

            with autocast_context(device, bf16=bf16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_loss += float(outputs.loss.detach().float().item()) * masked_count
            preds = outputs.logits.detach().argmax(dim=-1)
            total_correct += int((preds[mask] == labels[mask]).sum().item())
            total_masked += masked_count

    if total_masked == 0:
        raise RuntimeError("Evaluation saw zero masked tokens.")

    loss = total_loss / total_masked
    return {
        "fixed_eval_loss": float(loss),
        "fixed_eval_perplexity": float(math.exp(loss)),
        "fixed_eval_masked_accuracy": float(total_correct / total_masked),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.sweep_root
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "fixed_eval_per_prob.log")

    log.info("=== fixed_eval_best_models (per masking-probability) ===")
    log.info("sweep_root=%s  valid_parquet=%s", args.sweep_root, args.valid_parquet)
    log.info(
        "fixed eval: strategy=%s  mlm_prob=%.2f  seed=%d",
        args.fixed_masking_strategy,
        args.fixed_mlm_probability,
        args.fixed_seed,
    )
    if args.limit is not None:
        log.info("Smoke mode: limit=%d examples per eval set", args.limit)

    log.info("--- Discovering best run per (strategy, mlm_probability) ---")
    selected_runs = discover_best_per_group(args.sweep_root)
    log.info("Selected %d runs:", len(selected_runs))
    for run in selected_runs:
        log.info(
            "  [%-20s]  run=%-40s  lr=%g  train_loss=%.6f  ckpt=%s",
            run.label,
            run.run_name,
            run.learning_rate,
            run.training_best_metric,
            run.best_checkpoint,
        )

    log.info("--- Checking compatibility (tokenizer + config SHA) ---")
    compatibility = assert_compatible_runs(
        selected_runs,
        expected_vocab_size=args.expected_vocab_size,
        expected_max_position_embeddings=args.expected_max_position_embeddings,
    )
    for key, val in compatibility.items():
        log.info("  %s: %s", key, val)

    tokenizer_dir = selected_runs[0].run_dir / "final_model" / "ape_tokenizer"
    log.info("Loading tokenizer from %s", tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    special_ids = resolve_special_ids(tokenizer)
    pad_token_id = int(special_ids["pad_token"])
    mask_token_id = int(special_ids["mask_token"])
    vocab_size = int(getattr(tokenizer, "vocab_size", args.expected_vocab_size))
    ids_to_tokens = dict(getattr(tokenizer, "ids_to_tokens", {}))
    log.info("  vocab_size=%d  pad=%d  mask=%d", vocab_size, pad_token_id, mask_token_id)

    train_matched_n = min(4096, args.limit) if args.limit is not None else 4096
    eval_sets = {
        "valid_full": load_valid_full(args.valid_parquet, limit=args.limit),
        "valid_4096_train_matched": load_valid_train_matched(
            args.valid_parquet,
            n_examples=train_matched_n,
            seed=242,
            shuffle_buffer_size=100_000,
        ),
    }
    for name, seqs in eval_sets.items():
        log.info("Eval set %-30s  %d sequences", name, len(seqs))

    fixed_sets: dict[str, dict[str, Any]] = {}
    manifest_sets: dict[str, dict[str, Any]] = {}
    for eval_set_name, sequences in eval_sets.items():
        log.info("--- Building fixed masked dataset: %s ---", eval_set_name)
        examples = encode_and_pad_sequences(
            sequences,
            tokenizer,
            max_seq_length=args.max_seq_length,
            pad_token_id=pad_token_id,
        )
        fixed = build_fixed_masked_dataset(
            examples,
            batch_size=args.batch_size,
            seed=args.fixed_seed,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            special_token_ids=list(special_ids.values()),
            ids_to_tokens=ids_to_tokens,
            mlm_probability=args.fixed_mlm_probability,
            masking_strategy=args.fixed_masking_strategy,
        )
        log.info(
            "  examples=%d  seq_len=%d  masked=%d / %d eligible  actual_frac=%.4f",
            fixed["num_examples"],
            fixed["seq_length"],
            fixed["masked_tokens"],
            fixed["eligible_tokens"],
            fixed["actual_mask_fraction"],
        )
        log.info("  fingerprint=%s", fixed["fingerprint_sha256"])
        cache_path = output_dir / f"fixed_eval_{eval_set_name}.pt"
        save_fixed_dataset(cache_path, fixed)
        log.info("  Saved fixed dataset to %s", cache_path)
        fixed_sets[eval_set_name] = fixed
        manifest_sets[eval_set_name] = {
            "cache_path": str(cache_path),
            "source_dataset_path": str(args.valid_parquet),
            "num_examples": fixed["num_examples"],
            "seq_length": fixed["seq_length"],
            "fixed_seed": args.fixed_seed,
            "fixed_masking_strategy": args.fixed_masking_strategy,
            "fixed_mlm_probability": args.fixed_mlm_probability,
            "masked_tokens": fixed["masked_tokens"],
            "eligible_tokens": fixed["eligible_tokens"],
            "actual_mask_fraction": fixed["actual_mask_fraction"],
            "fingerprint_sha256": fixed["fingerprint_sha256"],
        }

    device = select_device(args.device)
    log.info("Device: %s  bf16=%s", device, args.bf16)

    rows: list[dict[str, Any]] = []
    for eval_set_name, fixed in fixed_sets.items():
        log.info("--- Evaluating on %s ---", eval_set_name)
        for run in selected_runs:
            log.info("  Evaluating [%s] %s ...", run.label, run.run_name)
            metrics = evaluate_checkpoint(
                run,
                fixed,
                batch_size=args.batch_size,
                device=device,
                bf16=args.bf16,
            )
            log.info(
                "  [%-20s]  loss=%.6f  ppl=%.4f  acc=%.4f",
                run.label,
                metrics["fixed_eval_loss"],
                metrics["fixed_eval_perplexity"],
                metrics["fixed_eval_masked_accuracy"],
            )
            rows.append(
                {
                    "eval_set": eval_set_name,
                    "label": run.label,
                    "run_name": run.run_name,
                    "best_checkpoint": str(run.best_checkpoint),
                    "trained_masking_strategy": run.trained_masking_strategy,
                    "trained_mlm_probability": run.trained_mlm_probability,
                    "learning_rate": run.learning_rate,
                    "training_best_metric": run.training_best_metric,
                    "fixed_masking_strategy": args.fixed_masking_strategy,
                    "fixed_mlm_probability": args.fixed_mlm_probability,
                    "fixed_seed": args.fixed_seed,
                    "num_examples": fixed["num_examples"],
                    "masked_tokens": fixed["masked_tokens"],
                    "actual_mask_fraction": fixed["actual_mask_fraction"],
                    **metrics,
                }
            )

    results_csv = output_dir / "fixed_eval_per_prob_results.csv"
    results_json = output_dir / "fixed_eval_per_prob_results.json"
    manifest_json = output_dir / "fixed_eval_per_prob_manifest.json"

    write_csv(results_csv, rows)
    results_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    manifest_json.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "sweep_root": str(args.sweep_root),
                "valid_parquet": str(args.valid_parquet),
                "tokenizer_dir": str(tokenizer_dir),
                **compatibility,
                "selected_runs": [
                    {
                        "label": run.label,
                        "run_name": run.run_name,
                        "run_dir": str(run.run_dir),
                        "best_checkpoint": str(run.best_checkpoint),
                        "trained_masking_strategy": run.trained_masking_strategy,
                        "trained_mlm_probability": run.trained_mlm_probability,
                        "learning_rate": run.learning_rate,
                        "training_best_metric": run.training_best_metric,
                    }
                    for run in selected_runs
                ],
                "eval_sets": manifest_sets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log.info("--- Done ---")
    log.info("Wrote %s", results_csv)
    log.info("Wrote %s", results_json)
    log.info("Wrote %s", manifest_json)

    log.info("=== Summary (valid_full, fixed standard@%.2f) ===", args.fixed_mlm_probability)
    log.info("  %-20s  %8s  %8s  %8s  %8s", "label", "train_loss", "fix_loss", "fix_ppl", "fix_acc")
    for row in [r for r in rows if r["eval_set"] == "valid_full"]:
        log.info(
            "  %-20s  %8.4f  %8.4f  %8.4f  %8.4f",
            row["label"],
            row["training_best_metric"],
            row["fixed_eval_loss"],
            row["fixed_eval_perplexity"],
            row["fixed_eval_masked_accuracy"],
        )


if __name__ == "__main__":
    main()
