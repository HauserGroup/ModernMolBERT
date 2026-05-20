#!/usr/bin/env python3
"""Upload a trained ModernMolBERT checkpoint to HuggingFace Hub.

# Upload final model
uv run python -m modernmolbert.upload_model \
  --run_dir runs/chembl36_small_mask_mlm_lr_sweep/mask_standard__mlm_0p15__lr_1e-4 \
  --repo_id HauserGroup/ModernMolBERT-small-chembl36 \
  --private

# Upload best checkpoint instead of final
uv run python -m modernmolbert.upload_model \
  --run_dir runs/chembl36_small_mask_mlm_lr_sweep/mask_standard__mlm_0p15__lr_1e-4 \
  --repo_id HauserGroup/ModernMolBERT-small-chembl36 \
  --checkpoint best \
  --private

# Upload a specific step checkpoint
uv run python -m modernmolbert.upload_model \
  --run_dir runs/chembl36_small_mask_mlm_lr_sweep/mask_standard__mlm_0p15__lr_1e-4 \
  --repo_id HauserGroup/ModernMolBERT-small-chembl36 \
  --checkpoint 25000 \
  --private
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from modernmolbert.utils import repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a trained ModernMolBERT checkpoint to HuggingFace Hub.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Training run directory (e.g. runs/my_run).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. HauserGroup/ModernMolBERT-small-chembl36).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final",
        help=(
            "Which checkpoint to upload. "
            "'final' (default) uses final_model/; "
            "'best' reads trainer_state.json to find the best checkpoint; "
            "a number (e.g. '25000') uses checkpoint-25000/."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HuggingFace repo as private.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload trained ModernMolBERT checkpoint",
        help="Commit message for the HuggingFace upload.",
    )
    parser.add_argument(
        "--hf_login",
        action="store_true",
        help="Call huggingface_hub.login() before uploading (reads HF_TOKEN from env / .env).",
    )
    return parser.parse_args()


def resolve_source_dir(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "final":
        source = run_dir / "final_model"
    elif checkpoint == "best":
        state_path = run_dir / "trainer_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.json not found in {run_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if not best:
            raise ValueError("trainer_state.json has no best_model_checkpoint entry")
        source = Path(best)
        if not source.is_absolute():
            source = repo_root() / source
    else:
        source = run_dir / f"checkpoint-{checkpoint}"

    if not (source / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors not found in {source}")
    if not (source / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {source}")

    return source


def _build_readme(source_dir: Path, run_dir: Path, repo_id: str) -> str:
    config: dict = {}
    config_path = source_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    run_args: dict = {}
    run_args_path = run_dir / "run_args.json"
    if run_args_path.exists():
        run_args = json.loads(run_args_path.read_text(encoding="utf-8"))

    state: dict = {}
    state_path = run_dir / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    frontmatter = """\
---
library_name: transformers
tags:
  - chemistry
  - selfies
  - masked-language-modeling
---
"""

    model_details = f"""\
## Model details

| field | value |
|-------|-------|
| model_type | {config.get("model_type", "modernbert")} |
| vocab_size | {config.get("vocab_size", "—")} |
| hidden_size | {config.get("hidden_size", "—")} |
| num_hidden_layers | {config.get("num_hidden_layers", "—")} |
| num_attention_heads | {config.get("num_attention_heads", "—")} |
| intermediate_size | {config.get("intermediate_size", "—")} |
| max_position_embeddings | {config.get("max_position_embeddings", "—")} |
"""

    training_rows = []
    for key in (
        "dataset_name",
        "model_size",
        "mlm_probability",
        "masking_strategy",
        "max_steps",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "weight_decay",
        "max_seq_length",
        "seed",
    ):
        if key in run_args:
            training_rows.append(f"| {key} | {run_args[key]} |")

    best_metric = state.get("best_metric")
    best_step = state.get("best_global_step")
    if best_metric is not None:
        training_rows.append(f"| best_eval_loss | {best_metric:.6f} |")
    if best_step is not None:
        training_rows.append(f"| best_global_step | {best_step} |")

    training_section = (
        "## Training\n\n| field | value |\n|-------|-------|\n" + "\n".join(training_rows) + "\n"
    )

    usage = f"""\
## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
```

This model expects SELFIES strings. Convert SMILES before tokenization.
"""

    return (
        frontmatter
        + f"\n# {repo_id}\n\n"
        + "ModernBERT pre-trained on SELFIES for masked language modeling.\n\n"
        + model_details
        + "\n"
        + training_section
        + "\n"
        + usage
    )


def build_staging_dir(source_dir: Path, run_dir: Path, repo_id: str, tmp: Path) -> None:
    shutil.copy(source_dir / "model.safetensors", tmp / "model.safetensors")
    shutil.copy(source_dir / "config.json", tmp / "config.json")

    # Prefer ape_tokenizer/ subdir; fall back to final_model/ape_tokenizer/ for
    # intermediate checkpoints that don't carry tokenizer artifacts.
    ape_dir = source_dir / "ape_tokenizer"
    if not ape_dir.exists():
        ape_dir = run_dir / "final_model" / "ape_tokenizer"
    if ape_dir.exists():
        shutil.copy(ape_dir / "vocab.json", tmp / "vocab.json")
        shutil.copy(ape_dir / "tokenizer_config.json", tmp / "tokenizer_config.json")
        shutil.copy(ape_dir / "special_tokens_map.json", tmp / "special_tokens_map.json")
        # tokenization_ape.py must be at repo root for trust_remote_code
        shutil.copy(ape_dir / "tokenization_ape.py", tmp / "tokenization_ape.py")
    elif (source_dir / "vocab.json").exists():
        shutil.copy(source_dir / "vocab.json", tmp / "vocab.json")

    for name in ("run_args.json", "trainer_state.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, tmp / name)

    readme = _build_readme(source_dir, run_dir, repo_id)
    (tmp / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Prefer HF_TOKEN_ORG for org-scoped uploads, fall back to HF_TOKEN.
    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None

    if args.hf_login:
        from huggingface_hub import login

        login(token=token)
        token = None  # login() caches it; don't double-pass

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = repo_root() / run_dir

    source_dir = resolve_source_dir(run_dir, args.checkpoint)
    print(f"Source: {source_dir}")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        build_staging_dir(source_dir, run_dir, args.repo_id, tmp)

        staged = sorted(tmp.iterdir())
        print(f"Staged {len(staged)} files: {[f.name for f in staged]}")

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    print(f"Done — https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
