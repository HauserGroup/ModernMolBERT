#!/usr/bin/env python3
"""Inspect embedded joblib artifacts for metadata related to sequence length.

Run from repo root:

    uv run python analysis/validation/audit_joblib_metadata.py \
      --roots outputs results
"""

import argparse
from pathlib import Path
from typing import Any

import joblib


def print_metadata(path: Path, obj: Any) -> None:
    meta = getattr(obj, "metadata", None)
    if meta:
        print(path, "metadata", meta)

    for attr in ["max_seq_length", "pooling", "model_dir", "tokenizer_path"]:
        if hasattr(obj, attr):
            print(path, attr, getattr(obj, attr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect joblib artifacts for metadata fields.")
    parser.add_argument("--roots", nargs="+", default=["outputs", "results"])
    args = parser.parse_args()

    paths: list[Path] = []
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        paths.extend(root_path.rglob("*.joblib"))

    for path in sorted(paths):
        try:
            obj = joblib.load(path)
        except Exception:
            continue

        print_metadata(path, obj)


if __name__ == "__main__":
    main()
