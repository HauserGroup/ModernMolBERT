#!/usr/bin/env python3
"""Hyperparameter sweep launcher for ModernMolBERT pre-training.

One grid over masking_strategy x mlm_probability x learning_rate, launched
sequentially with `accelerate launch` on a single CUDA device. Replaces the
former per-variant shell scripts. Run from the repo root:

    python scripts/sweeps/run_sweep.py --model-size small
    python scripts/sweeps/run_sweep.py --model-size base --masking standard
    python scripts/sweeps/run_sweep.py --model-size small --dry-run

Batch geometry, warmup, and the default learning-rate grid follow the model
size (see PRESETS); every axis can be overridden on the command line. Each run
writes to runs/<run-root>/<run-name>/ and tees its output to train.log.
Already-populated output directories are skipped, so re-running after a partial
failure resumes the remaining runs.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ─── Fixed across all runs ────────────────────────────────────────────────────
# The prepared chembl36 dataset carries both a `selfies` and a `smiles_canonical_clean`
# column, so both representations reuse the same dataset directory read-only.
DATASET_DIR = "data/pretrain/chembl36_selfies"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "valid"

# Per-representation tokenizer, dataset column, and masking grid. SELFIES keeps the
# historical defaults; SMILES points at the SMILES APE tokenizer and drops hetero_span
# (its heteroatom bias is SELFIES-bracket-specific and degrades to plain span on SMILES).
REPRESENTATION_DEFAULTS = {
    "SELFIES": {
        "tokenizer_path": "tokenizer/chembl36_selfies_2m_ape_max2_min3000.json",
        "tokenizer_metadata_path": "tokenizer/chembl36_selfies_2m_ape_max2_min3000.metadata.json",
        "molecule_column": "selfies",
        "masking": ["standard", "span", "hetero_span"],
        "run_root_tag": "",
    },
    "SMILES": {
        "tokenizer_path": "tokenizer/chembl36_smiles_2m_ape_max6_mf3000.json",
        "tokenizer_metadata_path": "tokenizer/chembl36_smiles_2m_ape_max6_mf3000.metadata.json",
        "molecule_column": "smiles_canonical_clean",
        "masking": ["standard", "span"],
        "run_root_tag": "smiles_",
    },
}

MAX_SEQ_LENGTH = 128
MAX_STEPS = 30000
EVAL_SIZE = 4096
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 5000
EVAL_STEPS = 5000
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2
NUM_WORKERS = 4
SEED = 42

ALL_MASKING = ["standard", "span", "hetero_span"]
# Strings (not floats) so run-directory names match the literal CLI tokens.
DEFAULT_MLM_PROBS = ["0.15", "0.20", "0.25"]

# Per-model-size batch geometry, warmup, and default learning-rate grid.
PRESETS = {
    "small": {
        "per_device_batch": 256,
        "grad_accum": 1,
        "max_eval_batches": 16,
        "warmup_steps": 1500,
        "learning_rates": ["1e-4", "2e-4", "4e-4"],
    },
    "base": {
        "per_device_batch": 64,
        "grad_accum": 4,
        "max_eval_batches": 64,
        "warmup_steps": 2000,
        "learning_rates": ["2e-4", "4e-4", "8e-4"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-size", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--representation",
        choices=sorted(REPRESENTATION_DEFAULTS),
        default="SELFIES",
        help="Molecular string representation (default: SELFIES).",
    )
    parser.add_argument(
        "--masking",
        nargs="+",
        choices=ALL_MASKING,
        default=None,
        help=(
            "Masking strategies to sweep (default: per representation; "
            "SELFIES uses all three, SMILES uses standard+span)."
        ),
    )
    parser.add_argument(
        "--mlm-probs",
        nargs="+",
        default=DEFAULT_MLM_PROBS,
        help="MLM probabilities to sweep.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        default=None,
        help="Learning rates to sweep (default: per model-size preset).",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Output root (default: runs/chembl36_<model-size>_mask_mlm_lr_sweep).",
    )
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument(
        "--molecule-column",
        default=None,
        help="Dataset column (default: per representation).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer vocabulary JSON (default: per representation).",
    )
    parser.add_argument(
        "--tokenizer-metadata-path",
        default=None,
        help="Tokenizer metadata JSON (default: per representation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs and commands without launching.",
    )
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> None:
    if not Path(args.dataset_dir).is_dir():
        sys.exit(
            f"ERROR: missing local dataset directory: {args.dataset_dir}\n"
            "       Run prepare_chembl36_selfies first."
        )
    for label, path in [
        ("tokenizer", args.tokenizer_path),
        ("tokenizer metadata", args.tokenizer_metadata_path),
    ]:
        if not Path(path).is_file():
            sys.exit(f"ERROR: missing {label}: {path}")


def run_name_for(masking: str, mlm: str, lr: str) -> str:
    return f"mask_{masking}__mlm_{mlm}__lr_{lr}".replace(".", "p")


def build_command(
    args: argparse.Namespace, preset: dict, masking: str, mlm: str, lr: str, output_dir: Path
) -> list[str]:
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--dynamo_backend",
        "no",
        "--mixed_precision",
        "bf16",
        "-m",
        "modernmolbert.train_selfies_ape_modernbert",
        "--dataset_name",
        args.dataset_dir,
        "--representation",
        args.representation,
        "--molecule_column",
        args.molecule_column,
        "--train_split",
        TRAIN_SPLIT,
        "--use_validation_split",
        "--validation_split",
        VALIDATION_SPLIT,
        "--output_dir",
        str(output_dir),
        "--device_backend",
        "cuda",
        "--model_size",
        args.model_size,
        "--tokenizer_vocab_path",
        args.tokenizer_path,
        "--tokenizer_metadata_path",
        args.tokenizer_metadata_path,
        "--max_seq_length",
        str(MAX_SEQ_LENGTH),
        "--max_steps",
        str(args.max_steps),
        "--eval_size",
        str(EVAL_SIZE),
        "--max_eval_batches",
        str(preset["max_eval_batches"]),
        "--per_device_train_batch_size",
        str(preset["per_device_batch"]),
        "--per_device_eval_batch_size",
        str(preset["per_device_batch"]),
        "--gradient_accumulation_steps",
        str(preset["grad_accum"]),
        "--mlm_probability",
        mlm,
        "--masking_strategy",
        masking,
        "--learning_rate",
        lr,
        "--weight_decay",
        str(WEIGHT_DECAY),
        "--max_grad_norm",
        str(MAX_GRAD_NORM),
        "--warmup_steps",
        str(preset["warmup_steps"]),
        "--logging_steps",
        str(LOGGING_STEPS),
        "--eval_steps",
        str(EVAL_STEPS),
        "--save_steps",
        str(SAVE_STEPS),
        "--save_total_limit",
        str(SAVE_TOTAL_LIMIT),
        "--num_workers",
        str(NUM_WORKERS),
        "--seed",
        str(SEED),
        "--compute_masked_accuracy",
        "--report_to",
        "tensorboard",
    ]


def launch(cmd: list[str], log_path: Path) -> int:
    """Run cmd, streaming combined output to the console and to log_path."""
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return proc.wait()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.model_size]
    learning_rates = args.learning_rates or preset["learning_rates"]

    # Fill representation-dependent defaults for anything not overridden on the CLI.
    rep = REPRESENTATION_DEFAULTS[args.representation]
    args.tokenizer_path = args.tokenizer_path or rep["tokenizer_path"]
    args.tokenizer_metadata_path = args.tokenizer_metadata_path or rep["tokenizer_metadata_path"]
    args.molecule_column = args.molecule_column or rep["molecule_column"]
    args.masking = args.masking or rep["masking"]

    invalid = [m for m in args.masking if m not in rep["masking"]]
    if invalid:
        sys.exit(
            f"ERROR: masking {invalid} not valid for representation {args.representation}; "
            f"allowed: {rep['masking']}"
        )

    run_root = args.run_root or Path(
        f"runs/chembl36_{rep['run_root_tag']}{args.model_size}_mask_mlm_lr_sweep"
    )

    if not args.dry_run:
        preflight(args)
    run_root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, str, str, Path, str]] = []
    total = skipped = 0
    for masking in args.masking:
        for mlm in args.mlm_probs:
            for lr in learning_rates:
                total += 1
                name = run_name_for(masking, mlm, lr)
                output_dir = run_root / name
                if output_dir.is_dir() and any(output_dir.iterdir()):
                    print(f"SKIP  already populated: {output_dir}")
                    skipped += 1
                    continue
                pending.append((masking, mlm, lr, output_dir, name))

    bar = "─" * 62
    print(bar)
    print(f"Total grid: {total}  Skipped: {skipped}  To run: {len(pending)}")
    print(bar)

    if not pending:
        print("Nothing to do. All runs already present.")
        print(f"TensorBoard: uv run tensorboard --logdir {run_root}")
        return

    for masking, mlm, lr, output_dir, name in pending:
        cmd = build_command(args, preset, masking, mlm, lr, output_dir)
        print(bar)
        print(f"LAUNCH  {name}")
        print(f"  mask={masking}  mlm={mlm}  lr={lr}")
        print(f"  output → {output_dir}")

        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "train.log"
        print(f"  log    → {log_file}")
        code = launch(cmd, log_file)
        if code != 0:
            sys.exit(f"Run failed ({name}) with exit code {code}; aborting sweep.")
        print(f"  Done: {name}")

    print("═" * 60)
    print("Sweep complete.")
    print(f"Results: {run_root}")
    print(f"TensorBoard: uv run tensorboard --logdir {run_root}")
    print("═" * 60)


if __name__ == "__main__":
    main()
