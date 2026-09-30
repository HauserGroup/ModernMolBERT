# %%
"""Count parameters for Hugging Face model repos or raw checkpoint files.

This utility uses a temporary Hugging Face cache directory and removes it at the end.

Examples:
    uv run python analysis/examples/count_hf_params.py bert-base-uncased roberta-base
    uv run python analysis/examples/count_hf_params.py \
        --hf-file dptech/Uni-Mol-Models mol_pre_all_h_220816.pt
"""

import argparse
import gc
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count parameters for Hugging Face model IDs (AutoModel) and/or raw "
            "Hub files such as .pt checkpoints."
        ),
    )
    parser.add_argument(
        "model_ids",
        nargs="*",
        help="One or more Hugging Face model IDs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom modeling code from the model repository.",
    )
    parser.add_argument(
        "--hf-file",
        nargs=2,
        action="append",
        metavar=("REPO_ID", "FILENAME"),
        help=(
            "Raw file on Hugging Face Hub to load with torch.load. Can be provided multiple times."
        ),
    )
    return parser.parse_args()


def _count_tensors(obj: Any) -> tuple[int, set[int]]:
    """Recursively count tensor elements, deduplicating aliased storages."""
    if torch.is_tensor(obj):
        ptr = int(obj.untyped_storage().data_ptr())
        return int(obj.numel()), {ptr}

    if isinstance(obj, dict):
        total = 0
        ptrs: set[int] = set()
        for value in obj.values():
            n, p = _count_tensors(value)
            total += n
            ptrs |= p
        return total, ptrs

    if isinstance(obj, (list, tuple)):
        total = 0
        ptrs: set[int] = set()
        for value in obj:
            n, p = _count_tensors(value)
            total += n
            ptrs |= p
        return total, ptrs

    return 0, set()


def count_model_params(model_id: str, *, trust_remote_code: bool) -> None:
    # Download weights into an isolated temporary cache that is removed afterwards.
    # (Setting HF_HOME here would have no effect: huggingface_hub reads it at import.)
    tmp_root = Path(tempfile.mkdtemp(prefix="hf_param_count_"))
    transformers_cache = tmp_root / "transformers_cache"

    print(f"Model: {model_id}")
    print(f"Temporary cache root: {tmp_root}")

    model = None
    try:
        try:
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=str(transformers_cache),
                trust_remote_code=trust_remote_code,
            )
        except ValueError as exc:
            if "trust_remote_code=True" in str(exc) and not trust_remote_code:
                raise ValueError(
                    "This model requires custom Hub code. Re-run with --trust-remote-code."
                ) from exc
            raise

        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    finally:
        # Free model memory and remove all downloaded files.
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"Removed temporary cache: {tmp_root}")
        print()


def count_checkpoint_file_params(repo_id: str, filename: str) -> None:
    """Count tensor elements in a raw checkpoint file downloaded from Hugging Face Hub."""
    tmp_root = Path(tempfile.mkdtemp(prefix="hf_file_param_count_"))
    checkpoint_obj: Any = None

    print(f"Checkpoint file: {repo_id}/{filename}")
    print(f"Temporary cache root: {tmp_root}")

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            cache_dir=str(tmp_root / "hf_cache"),
        )
        checkpoint_obj = torch.load(file_path, map_location="cpu")
        total_elements, unique_ptrs = _count_tensors(checkpoint_obj)
        print(f"Total tensor elements: {total_elements:,}")
        print(f"Unique tensor storages: {len(unique_ptrs):,}")
    finally:
        if checkpoint_obj is not None:
            del checkpoint_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"Removed temporary cache: {tmp_root}")
        print()


def main() -> None:
    args = parse_args()

    if not args.model_ids and not args.hf_file:
        raise SystemExit("Pass at least one model ID or --hf-file.")

    for model_id in args.model_ids:
        count_model_params(model_id, trust_remote_code=args.trust_remote_code)

    for repo_id, filename in args.hf_file or []:
        count_checkpoint_file_params(repo_id=repo_id, filename=filename)


if __name__ == "__main__":
    main()
