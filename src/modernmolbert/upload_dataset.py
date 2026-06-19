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
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from modernmolbert.hf_upload import make_staging_dir, push_folder_to_hub, resolve_hf_token
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


def _fmt_int(value: object) -> str:
    """Thousands-separated int, or the value as-is when it is not an int."""
    return f"{value:,}" if isinstance(value, int) else str(value)


# Ordered (column, group, description). Only columns present in metadata are
# rendered, so the table never claims a field the parquet does not contain.
COLUMN_DOCS: list[tuple[str, str, str]] = [
    ("selfies", "Primary input", "SELFIES string — the pre-training input."),
    (
        "smiles_canonical_clean",
        "Primary input",
        "RDKit-canonicalized SMILES the SELFIES was encoded from.",
    ),
    ("chembl_id", "Identifier", "ChEMBL compound identifier."),
    ("canonical_smiles", "Identifier", "Original ChEMBL canonical SMILES."),
    ("standard_inchi", "Identifier", "Standard InChI."),
    ("standard_inchi_key", "Identifier", "InChIKey — used for deduplication and splitting."),
    ("molecule_type", "Identifier", "ChEMBL molecule type."),
    ("mw_freebase", "Descriptor", "Molecular weight of the free base."),
    ("alogp", "Descriptor", "Calculated logP."),
    ("hba", "Descriptor", "Hydrogen-bond acceptors."),
    ("hbd", "Descriptor", "Hydrogen-bond donors."),
    ("psa", "Descriptor", "Polar surface area."),
    ("rtb", "Descriptor", "Rotatable bonds."),
    ("aromatic_rings", "Descriptor", "Aromatic ring count."),
    ("heavy_atoms", "Descriptor", "Heavy-atom count."),
    ("qed_weighted", "Descriptor", "QED drug-likeness score."),
    ("num_ro5_violations", "Descriptor", "Lipinski rule-of-five violations."),
    ("max_phase", "Annotation", "Maximum clinical trial phase reached."),
    ("first_approval", "Annotation", "Year of first approval."),
    ("oral", "Annotation", "Orally administered flag."),
    ("prodrug", "Annotation", "Prodrug flag."),
    ("natural_product", "Annotation", "Natural-product flag."),
    ("black_box_warning", "Annotation", "Black-box-warning flag."),
    ("withdrawn_flag", "Annotation", "Withdrawn-from-market flag."),
    ("therapeutic_flag", "Annotation", "Therapeutic-use flag."),
    ("is_valid", "Bookkeeping", "Passed sanitization and physicochemical filters."),
    ("sanitize_error", "Bookkeeping", "Sanitization error tag (empty when valid)."),
    ("split_key", "Bookkeeping", "Key hashed to assign the train/validation split."),
]

EXAMPLE_SELFIES = (
    "[C][C][=Branch1][C][=O][O][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][O]"
)


def _frontmatter(counts: dict) -> str:
    total = counts.get("prepared_total")
    if isinstance(total, int) and 1_000_000 <= total < 10_000_000:
        size_category = "1M<n<10M"
    elif isinstance(total, int) and 100_000 <= total < 1_000_000:
        size_category = "100K<n<1M"
    else:
        size_category = "unknown"
    return (
        "---\n"
        "pretty_name: ChEMBL 36 SELFIES\n"
        "license: cc-by-4.0\n"
        "task_categories:\n"
        "- fill-mask\n"
        "size_categories:\n"
        f"- {size_category}\n"
        "tags:\n"
        "- chemistry\n"
        "- molecules\n"
        "- selfies\n"
        "- chembl\n"
        "- pre-training\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: train.parquet\n"
        "  - split: validation\n"
        "    path: valid.parquet\n"
        "---\n"
    )


def _columns_table(metadata: dict) -> str:
    present = set(metadata.get("columns", []))
    rows = [
        f"| `{name}` | {group} | {desc} |" for name, group, desc in COLUMN_DOCS if name in present
    ]
    if not rows:
        return ""
    header = "| column | group | description |\n|--------|-------|-------------|\n"
    return "## Columns\n\n" + header + "\n".join(rows) + "\n\n"


def build_readme(metadata: dict) -> str:
    stats = metadata.get("preparation_stats", {})
    counts = metadata.get("row_counts", {})
    config = metadata.get("config", {})
    versions = metadata.get("versions", {})
    split = metadata.get("split_policy", {})
    sanitize = stats.get("sanitize_error_counts", {})

    train_n = counts.get("train", "-")
    valid_n = counts.get("valid", "-")
    total_n = counts.get("prepared_total", "-")

    overlap = metadata.get("split_overlap", {}).get("train_valid", {}).get("n_overlap", 0)

    sanitize_lines = ""
    if sanitize:
        sanitize_lines = "\nBreakdown of dropped rows:\n\n" + "".join(
            f"- {reason}: {_fmt_int(count)}\n"
            for reason, count in sanitize.items()
            if reason != "valid"
        )

    return (
        _frontmatter(counts) + "\n# ChEMBL 36 SELFIES\n\n" + "Pre-training dataset for "
        "[ModernMolBERT](https://github.com/HauserGroup/ModernMolBERT): ~2.4M drug-like "
        "small molecules from ChEMBL 36, deduplicated and physicochemically filtered, "
        "each represented as a [SELFIES](https://github.com/aspuru-guzik-group/selfies) "
        "string alongside its source SMILES and ChEMBL descriptors.\n\n"
        "## Dataset details\n\n"
        "| field | value |\n"
        "|-------|-------|\n"
        f"| source | [{metadata.get('dataset_name', '-')}](https://huggingface.co/datasets/{metadata.get('dataset_name', '-')}) |\n"
        f"| representation | {metadata.get('representation', '-')} |\n"
        f"| train rows | {_fmt_int(train_n)} |\n"
        f"| validation rows | {_fmt_int(valid_n)} |\n"
        f"| total rows | {_fmt_int(total_n)} |\n"
        f"| min heavy atoms | {config.get('min_heavy_atoms', '-')} |\n"
        f"| max heavy atoms | {config.get('max_heavy_atoms', '-')} |\n"
        f"| max MW | {config.get('max_mw', '-')} |\n"
        f"| deduplicated by | InChIKey ({config.get('dedupe_column', 'standard_inchi_key')}) |\n\n"
        "## Splitting\n\n"
        f"Split by **{split.get('method', 'deterministic hash')}** on `{split.get('key_column', 'split_key')}` "
        f"(seed {split.get('seed', '-')}), with a validation fraction of "
        f"{split.get('valid_fraction', config.get('valid_fraction', '-'))} and no test split. "
        "The split is reproducible: the same molecule always lands in the same split.\n\n"
        f"Train/validation molecule overlap: **{_fmt_int(overlap)}** "
        "(residual hash collisions; effectively disjoint).\n\n"
        "## Preparation stats\n\n"
        f"- Input rows: {_fmt_int(stats.get('input_rows', '-'))}\n"
        f"- After deduplication: {_fmt_int(stats.get('rows_after_dedupe', '-'))}\n"
        f"- Valid SELFIES conversions: {_fmt_int(stats.get('rows_valid_after_conversion', '-'))}\n"
        f"- After physicochemical filters: {_fmt_int(stats.get('rows_after_filters', '-'))}\n"
        f"- Dropped (invalid or filtered): {_fmt_int(stats.get('dropped_invalid_or_filtered', '-'))}\n"
        + sanitize_lines
        + "\n"
        + _columns_table(metadata)
        + "## Usage\n\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        f"ds = load_dataset('{DEFAULT_REPO_ID}')\n"
        "train, valid = ds['train'], ds['validation']\n\n"
        "print(train[0]['selfies'])\n"
        f"# {EXAMPLE_SELFIES}\n"
        "```\n\n"
        "## Citation\n\n"
        "If you use this dataset, please cite ChEMBL, SELFIES, and ModernMolBERT:\n\n"
        "```bibtex\n"
        "@article{zdrazil2024chembl,\n"
        "  title   = {The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods},\n"
        "  author  = {Zdrazil, Barbara and others},\n"
        "  journal = {Nucleic Acids Research},\n"
        "  volume  = {52},\n"
        "  number  = {D1},\n"
        "  pages   = {D1180--D1192},\n"
        "  year    = {2024}\n"
        "}\n\n"
        "@article{krenn2020selfies,\n"
        "  title   = {Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation},\n"
        "  author  = {Krenn, Mario and H{\\\"a}se, Florian and Nigam, AkshatKumar and Friederich, Pascal and Aspuru-Guzik, Al{\\'a}n},\n"
        "  journal = {Machine Learning: Science and Technology},\n"
        "  volume  = {1},\n"
        "  number  = {4},\n"
        "  pages   = {045024},\n"
        "  year    = {2020}\n"
        "}\n\n"
        "@article{madsen_modernmolbert,\n"
        "  title  = {ModernMolBERT: A ModernBERT Encoder Family for SELFIES Molecular Language Modeling},\n"
        "  author = {Madsen, Jakob S. and Angelucci, Sara and Hauser, Alexander S.},\n"
        "  year   = {2026}\n"
        "}\n"
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

    tmp, cleanup = make_staging_dir(keep_staging_dir)
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
            push_folder_to_hub(
                tmp,
                repo_id,
                repo_type="dataset",
                private=private,
                commit_message=commit_message,
                token=token,
                api=api,
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

    token = resolve_hf_token(args.hf_login)

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
