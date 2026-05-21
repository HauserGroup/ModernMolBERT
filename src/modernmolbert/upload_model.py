#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi

from modernmolbert.utils import repo_root


MODEL_MAX_LENGTH = 256
EXPECTED_VOCAB_SIZE = 631
EXPECTED_SPECIAL_IDS = {
    "bos_token_id": 0,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
    "mask_token_id": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a trained ModernMolBERT checkpoint to HuggingFace Hub.",
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Training run directory, for example runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID, for example HauserGroup/ModernMolBERT-small-chembl36.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final",
        help="Checkpoint to upload: final, best, or a numeric step such as 25000.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the HuggingFace repo as private.",
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
        help="Call huggingface_hub.login() before uploading. Reads HF_TOKEN_ORG or HF_TOKEN from env or .env.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Stage and validate upload contents without creating or updating a Hub repo.",
    )
    parser.add_argument(
        "--keep_staging_dir",
        type=Path,
        default=None,
        help="Keep staged upload files in this directory for debugging.",
    )

    return parser.parse_args()


def resolve_source_dir(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "final":
        source = run_dir / "final_model"

    elif checkpoint == "best":
        state_path = run_dir / "trainer_state.json"

        if not state_path.exists():
            fallback = run_dir / "final_model"
            if fallback.exists():
                print(
                    f"trainer_state.json not found in {run_dir}; falling back to {fallback}",
                    flush=True,
                )
                source = fallback
            else:
                raise FileNotFoundError(f"trainer_state.json not found in {run_dir}")

        else:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            best = state.get("best_model_checkpoint")

            if not best:
                fallback = run_dir / "final_model"
                if fallback.exists():
                    print(
                        f"trainer_state.json has no best_model_checkpoint; falling back to {fallback}",
                        flush=True,
                    )
                    source = fallback
                else:
                    raise ValueError("trainer_state.json has no best_model_checkpoint entry")

            else:
                source = Path(best)
                if not source.is_absolute():
                    source = repo_root() / source

                if not source.exists():
                    fallback = run_dir / "final_model"
                    if fallback.exists():
                        print(
                            f"best_model_checkpoint does not exist: {source}; falling back to {fallback}",
                            flush=True,
                        )
                        source = fallback
                    else:
                        raise FileNotFoundError(f"best_model_checkpoint does not exist: {source}")

    else:
        source = run_dir / f"checkpoint-{checkpoint}"

    if not (source / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors not found in {source}")

    if not (source / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {source}")

    return source


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_patch_config(source_dir: Path, run_dir: Path) -> dict[str, Any]:
    config_path = source_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    config.setdefault("model_type", "modernbert")
    config.setdefault("architectures", ["ModernBertForMaskedLM"])

    config.setdefault("vocab_size", EXPECTED_VOCAB_SIZE)
    config.setdefault("bos_token_id", EXPECTED_SPECIAL_IDS["bos_token_id"])
    config.setdefault("pad_token_id", EXPECTED_SPECIAL_IDS["pad_token_id"])
    config.setdefault("eos_token_id", EXPECTED_SPECIAL_IDS["eos_token_id"])
    config.setdefault("unk_token_id", EXPECTED_SPECIAL_IDS["unk_token_id"])
    config.setdefault("mask_token_id", EXPECTED_SPECIAL_IDS["mask_token_id"])

    run_args = load_json_if_exists(run_dir / "run_args.json")
    if "max_seq_length" in run_args:
        config.setdefault("max_position_embeddings", int(run_args["max_seq_length"]))

    return config


def find_tokenizer_vocab(source_dir: Path, run_dir: Path) -> Path:
    candidates = [
        source_dir / "ape_tokenizer" / "vocab.json",
        source_dir / "vocab.json",
        run_dir / "final_model" / "ape_tokenizer" / "vocab.json",
        run_dir / "vocab.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find tokenizer vocab.json. Checked:\n{checked}")


def find_tokenization_code(source_dir: Path, run_dir: Path) -> Path:
    candidates = [
        source_dir / "ape_tokenizer" / "tokenization_ape.py",
        source_dir / "tokenization_ape.py",
        run_dir / "final_model" / "ape_tokenizer" / "tokenization_ape.py",
        run_dir / "final_model" / "tokenization_ape.py",
        repo_root() / "src" / "modernmolbert" / "tokenization_ape.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find tokenization_ape.py. Checked:\n{checked}")


def write_tokenizer_config(tmp: Path) -> None:
    tokenizer_config_path = tmp / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json: {tokenizer_config_path}")

    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

    tokenizer_config.pop("tokenizer_class", None)
    tokenizer_config.update(
        {
            "representation": "SELFIES",
            "model_max_length": MODEL_MAX_LENGTH,
            "model_input_names": ["input_ids", "attention_mask"],
            "use_fast": False,
            "auto_map": {
                "AutoTokenizer": [
                    "tokenization_ape.APEPreTrainedTokenizer",
                    None,
                ],
            },
        }
    )

    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stage_tokenizer_files(source_dir: Path, run_dir: Path, tmp: Path) -> None:
    from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

    vocab_path = find_tokenizer_vocab(source_dir, run_dir)

    tokenizer = APEPreTrainedTokenizer(
        representation="SELFIES",
        model_max_length=MODEL_MAX_LENGTH,
    )
    tokenizer.load_vocabulary_file(vocab_path)
    tokenizer.save_pretrained(str(tmp))

    write_tokenizer_config(tmp)

    tokenization_code = find_tokenization_code(source_dir, run_dir)
    shutil.copy(tokenization_code, tmp / "tokenization_ape.py")

    for metadata_name in (
        "tokenizer_metadata.json",
        "ape_tokenizer_metadata.json",
        "metadata.json",
    ):
        candidates = [
            source_dir / "ape_tokenizer" / metadata_name,
            source_dir / metadata_name,
            run_dir / "final_model" / "ape_tokenizer" / metadata_name,
            run_dir / "final_model" / metadata_name,
            run_dir / metadata_name,
        ]

        for candidate in candidates:
            if candidate.exists():
                shutil.copy(candidate, tmp / metadata_name)
                break


def build_readme(source_dir: Path, run_dir: Path, repo_id: str) -> str:
    config = load_and_patch_config(source_dir, run_dir)
    run_args = load_json_if_exists(run_dir / "run_args.json")
    state = load_json_if_exists(run_dir / "trainer_state.json")

    frontmatter = (
        "---\n"
        "library_name: transformers\n"
        "tags:\n"
        "  - chemistry\n"
        "  - selfies\n"
        "  - modernbert\n"
        "  - masked-language-modeling\n"
        "---\n"
    )

    model_details = (
        "## Model details\n\n"
        "| field | value |\n"
        "|-------|-------|\n"
        f"| model_type | {config.get('model_type', 'modernbert')} |\n"
        f"| vocab_size | {config.get('vocab_size', '-')} |\n"
        f"| hidden_size | {config.get('hidden_size', '-')} |\n"
        f"| num_hidden_layers | {config.get('num_hidden_layers', '-')} |\n"
        f"| num_attention_heads | {config.get('num_attention_heads', '-')} |\n"
        f"| intermediate_size | {config.get('intermediate_size', '-')} |\n"
        f"| max_position_embeddings | {config.get('max_position_embeddings', '-')} |\n"
    )

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
        try:
            training_rows.append(f"| best_eval_loss | {float(best_metric):.6f} |")
        except TypeError:
            training_rows.append(f"| best_eval_loss | {best_metric} |")

    if best_step is not None:
        training_rows.append(f"| best_global_step | {best_step} |")

    training_section = (
        "## Training\n\n| field | value |\n|-------|-------|\n" + "\n".join(training_rows) + "\n"
    )

    usage = (
        "## Usage\n\n"
        "```python\n"
        "from transformers import AutoModelForMaskedLM, AutoTokenizer\n\n"
        f"model = AutoModelForMaskedLM.from_pretrained('{repo_id}')\n"
        "tokenizer = AutoTokenizer.from_pretrained(\n"
        f"    '{repo_id}',\n"
        "    trust_remote_code=True,\n"
        "    use_fast=False,\n"
        ")\n"
        "```\n\n"
        "This model expects SELFIES strings. Convert SMILES before tokenization.\n"
    )

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

    config = load_and_patch_config(source_dir, run_dir)
    (tmp / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stage_tokenizer_files(source_dir, run_dir, tmp)

    for name in (
        "run_args.json",
        "trainer_state.json",
        "tokenizer_metadata.json",
        "ape_tokenizer_metadata.json",
        "eval_results.json",
        "train_results.json",
        "all_results.json",
        "best_span_run.json",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, tmp / name)

    (tmp / "README.md").write_text(
        build_readme(source_dir, run_dir, repo_id),
        encoding="utf-8",
    )


def validate_staged_files(tmp: Path) -> None:
    required = [
        "config.json",
        "model.safetensors",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_ape.py",
        "README.md",
    ]

    missing = [name for name in required if not (tmp / name).exists()]
    if missing:
        raise FileNotFoundError(f"Staging directory is missing files: {missing}")


def validate_staged_model(tmp: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    print("[validate] loading config", flush=True)
    config = AutoConfig.from_pretrained(tmp)

    print("[validate] loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tmp,
        trust_remote_code=True,
        use_fast=False,
    )

    print("[validate] loading model", flush=True)
    model = AutoModelForMaskedLM.from_pretrained(tmp)
    model.eval()

    if config.model_type != "modernbert":
        raise ValueError(f"Unexpected model_type={config.model_type!r}")

    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: tokenizer={tokenizer.vocab_size}, config={config.vocab_size}"
        )

    if tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError(
            f"pad_token_id mismatch: tokenizer={tokenizer.pad_token_id}, config={config.pad_token_id}"
        )

    example = (
        "[C][C][=C][C][Branch1][=N][N][N][=C][C][=Branch1][C][=O][NH1][C][Ring1][#Branch1][=O]"
    )

    batch = tokenizer(
        [example],
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    print("[validate] running forward pass", flush=True)
    with torch.no_grad():
        out = model(**batch)

    if not torch.isfinite(out.logits).all():
        raise ValueError("Model forward pass produced non-finite logits.")

    n_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"[validate] OK model: {n_params / 1e6:.1f}M parameters", flush=True)
    print(
        f"[validate] OK tokenizer: vocab_size={tokenizer.vocab_size}, max_length={tokenizer.model_max_length}",
        flush=True,
    )
    print(f"[validate] OK logits shape={tuple(out.logits.shape)}", flush=True)


def prepare_staging_dir(keep_staging_dir: Path | None) -> tuple[Path, Callable[[], None]]:
    if keep_staging_dir is not None:
        tmp = keep_staging_dir

        if tmp.exists():
            shutil.rmtree(tmp)

        tmp.mkdir(parents=True, exist_ok=True)

        def cleanup() -> None:
            return None

        return tmp, cleanup

    temp_dir = tempfile.TemporaryDirectory()
    tmp = Path(temp_dir.name)

    def cleanup() -> None:
        temp_dir.cleanup()

    return tmp, cleanup


def upload_model_to_hub(
    run_dir: Path,
    repo_id: str,
    checkpoint: str = "final",
    private: bool = False,
    commit_message: str = "Upload trained ModernMolBERT checkpoint",
    token: str | None = None,
    dry_run: bool = False,
    keep_staging_dir: Path | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    if not run_dir.is_absolute():
        run_dir = repo_root() / run_dir

    source_dir = resolve_source_dir(run_dir, checkpoint)
    print(f"Source: {source_dir}", flush=True)

    tmp, cleanup = prepare_staging_dir(keep_staging_dir)
    staged_names: list[str] = []

    try:
        print(f"[upload] staging directory: {tmp}", flush=True)

        build_staging_dir(source_dir, run_dir, repo_id, tmp)
        validate_staged_files(tmp)

        staged = sorted(tmp.iterdir())
        staged_names = [path.name for path in staged]
        print(f"[upload] staged {len(staged)} files: {staged_names}", flush=True)
        validate_staged_model(tmp)

        if dry_run:
            print(f"Dry run: skipped upload to https://huggingface.co/{repo_id}", flush=True)
        else:
            if api is None:
                api = HfApi(token=token)
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )

    finally:
        cleanup()
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "run_dir": str(run_dir),
        "source_dir": str(source_dir),
        "checkpoint": checkpoint,
        "private": private,
        "uploaded": not dry_run,
        "staged_files": staged_names,
        "staging_dir": str(tmp),
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None

    if args.hf_login:
        from huggingface_hub import login

        login(token=token)
        token = None

    result = upload_model_to_hub(
        run_dir=args.run_dir,
        repo_id=args.repo_id,
        checkpoint=args.checkpoint,
        private=args.private,
        commit_message=args.commit_message,
        token=token,
        dry_run=args.dry_run,
        keep_staging_dir=args.keep_staging_dir,
    )

    print(f"Done — {result['url']}", flush=True)


if __name__ == "__main__":
    main()
