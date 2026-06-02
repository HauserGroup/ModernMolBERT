#!/usr/bin/env python3

"""
Upload the prepared ChEMBL 36 SELFIES pre-training dataset to HuggingFace Hub.

uv run python -m modernmolbert.upload_dataset \
  --dataset_dir data/pretrain/chembl36_selfies \
  --repo_id HauserGroup/ChEMBL36-SELFIES \
  --private \
  --dry_run
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable

from dotenv import load_dotenv
from huggingface_hub import HfApi

from modernmolbert.utils import repo_root


DEFAULT_REPO_ID = "HauserGroup/ChEMBL36-SELFIES"
DEFAULT_DATASET_DIR = Path("data/pretrain/chembl36_selfies")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the prepared ChEMBL 36 SELFIES dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing train.parquet, valid.parquet, and metadata.json (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload ChEMBL 36 SELFIES pre-training dataset",
        help="Commit message for the HuggingFace upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the HuggingFace repo as private.",
    )
    parser.add_argument(
        "--hf_login",
        action="store_true",
        help="Call huggingface_hub.login() before uploading. Reads HF_TOKEN_ORG or HF_TOKEN from env or .env.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Stage and validate files without creating or uploading to the Hub.",
    )
    parser.add_argument(
        "--keep_staging_dir",
        type=Path,
        default=None,
        help="Keep staged upload files in this directory for debugging.",
    )
    return parser.parse_args()


def build_readme(metadata: dict) -> str:
    stats = metadata.get("preparation_stats", {})
    counts = metadata.get("row_counts", {})
    config = metadata.get("config", {})
    versions = metadata.get("versions", {})

    frontmatter = (
        "---\n"
        "license: cc-by-4.0\n"
        "task_categories:\n"
        "- fill-mask\n"
        "language:\n"
        "- en\n"
        "tags:\n"
        "- chemistry\n"
        "- molecules\n"
        "- selfies\n"
        "- chembl\n"
        "- pre-training\n"
        "---\n"
    )

    train_n = counts.get("train", "-")
    valid_n = counts.get("valid", "-")
    total_n = counts.get("prepared_total", "-")

    return (
        frontmatter
        + "\n# ChEMBL 36 SELFIES\n\n"
        + "Pre-training dataset for [ModernMolBERT](https://github.com/HauserGroup/ModernMolBERT). "
        "Contains ~2.4M drug-like small molecules from ChEMBL 36 represented as "
        "[SELFIES](https://github.com/aspuru-guzik-group/selfies) strings.\n\n"
        "## Dataset details\n\n"
        "| field | value |\n"
        "|-------|-------|\n"
        f"| source | [{metadata.get('dataset_name', '-')}](https://huggingface.co/datasets/{metadata.get('dataset_name', '-')}) |\n"
        f"| representation | {metadata.get('representation', '-')} |\n"
        f"| train rows | {train_n:,} |\n"
        f"| validation rows | {valid_n:,} |\n"
        f"| total rows | {total_n:,} |\n"
        f"| min heavy atoms | {config.get('min_heavy_atoms', '-')} |\n"
        f"| max heavy atoms | {config.get('max_heavy_atoms', '-')} |\n"
        f"| max MW | {config.get('max_mw', '-')} |\n"
        f"| deduplicated by | InChIKey |\n"
        f"| split method | deterministic hash on InChIKey |\n"
        f"| valid fraction | {config.get('valid_fraction', '-')} |\n\n"
        "## Preparation stats\n\n"
        f"- Input rows: {stats.get('input_rows', '-'):,}\n"
        f"- After deduplication: {stats.get('rows_after_dedupe', '-'):,}\n"
        f"- Valid SELFIES conversions: {stats.get('rows_valid_after_conversion', '-'):,}\n"
        f"- After physicochemical filters: {stats.get('rows_after_filters', '-'):,}\n"
        f"- Dropped (invalid or filtered): {stats.get('dropped_invalid_or_filtered', '-'):,}\n\n"
        "## Columns\n\n"
        "| column | description |\n"
        "|--------|-------------|\n"
        "| `selfies` | SELFIES string (primary pre-training input) |\n"
        "| `canonical_smiles` | original ChEMBL SMILES |\n"
        "| `smiles_canonical_clean` | RDKit-canonicalized SMILES |\n"
        "| `standard_inchi_key` | InChIKey used for deduplication and splitting |\n"
        "| `chembl_id` | ChEMBL compound identifier |\n"
        "| `qed_weighted` | QED drug-likeness score |\n"
        "| `heavy_atoms`, `mw_freebase`, `alogp`, ... | Physicochemical descriptors from ChEMBL |\n\n"
        "## Usage\n\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        f"ds = load_dataset('{DEFAULT_REPO_ID}')\n"
        "print(ds['train'][0]['selfies'])\n"
        "```\n\n"
        "## Versions\n\n" + "\n".join(f"- {k}: {v}" for k, v in versions.items()) + "\n\n"
        "## License\n\n"
        "ChEMBL data is released under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). "
        "The prepared dataset files in this repository are released under CC BY 4.0.\n"
    )


def validate_dataset_dir(dataset_dir: Path) -> dict:
    required = ["train.parquet", "valid.parquet", "metadata.json"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset directory {dataset_dir} is missing files: {missing}")

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))

    train_rows = metadata.get("row_counts", {}).get("train")
    valid_rows = metadata.get("row_counts", {}).get("valid")
    if not train_rows or not valid_rows:
        raise ValueError("metadata.json is missing row_counts.train or row_counts.valid")

    print(
        f"[validate] dataset OK: train={train_rows:,} rows, valid={valid_rows:,} rows",
        flush=True,
    )
    return metadata


def build_staging_dir(dataset_dir: Path, tmp: Path) -> dict:
    metadata = validate_dataset_dir(dataset_dir)

    shutil.copy(dataset_dir / "train.parquet", tmp / "train.parquet")
    shutil.copy(dataset_dir / "valid.parquet", tmp / "valid.parquet")
    shutil.copy(dataset_dir / "metadata.json", tmp / "metadata.json")

    if (dataset_dir / "example.tsv").exists():
        shutil.copy(dataset_dir / "example.tsv", tmp / "example.tsv")

    (tmp / "README.md").write_text(build_readme(metadata), encoding="utf-8")

    print(
        f"[stage] staged files: {sorted(p.name for p in tmp.iterdir())}",
        flush=True,
    )
    return metadata


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


def upload_dataset_to_hub(
    dataset_dir: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload ChEMBL 36 SELFIES pre-training dataset",
    token: str | None = None,
    dry_run: bool = False,
    keep_staging_dir: Path | None = None,
    api: HfApi | None = None,
) -> dict:
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root() / dataset_dir

    tmp, cleanup = prepare_staging_dir(keep_staging_dir)
    staged_names: list[str] = []

    try:
        print(f"[upload] staging directory: {tmp}", flush=True)
        metadata = build_staging_dir(dataset_dir, tmp)

        staged = sorted(tmp.iterdir())
        staged_names = [p.name for p in staged]
        print(f"[upload] staged {len(staged)} files: {staged_names}", flush=True)

        if dry_run:
            print(
                f"Dry run: skipped upload to https://huggingface.co/datasets/{repo_id}", flush=True
            )
        else:
            if api is None:
                api = HfApi(token=token)
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            print(f"[upload] uploaded to https://huggingface.co/datasets/{repo_id}", flush=True)

    finally:
        cleanup()

    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "dataset_dir": str(dataset_dir),
        "row_counts": metadata.get("row_counts", {}),
        "private": private,
        "uploaded": not dry_run,
        "staged_files": staged_names,
        "staging_dir": str(tmp) if keep_staging_dir is not None else None,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None

    if args.hf_login:
        from huggingface_hub import login

        login(token=token)
        token = None

    result = upload_dataset_to_hub(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        token=token,
        dry_run=args.dry_run,
        keep_staging_dir=args.keep_staging_dir,
    )

    print(f"Done — {result['url']}", flush=True)


if __name__ == "__main__":
    main()
