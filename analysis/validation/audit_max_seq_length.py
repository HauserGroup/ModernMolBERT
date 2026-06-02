#!/usr/bin/env python3
"""Audit max-sequence-length provenance across text metadata files.

Run from repo root:

    uv run python analysis/validation/audit_max_seq_length.py \
      --roots runs outputs results tmp-hf-model tmp-hf-tokenizer \
      --expected 128
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

KEY_PATTERNS = re.compile(
    r"(max_?seq|seq_?len|model_max_length|max_position|n_positions|context)",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".log",
    ".args",
    ".py",
    ".sh",
}


def flatten_json(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_json(value, name))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            name = f"{prefix}[{i}]"
            out.extend(flatten_json(value, name))
    else:
        out.append((prefix, obj))

    return out


def inspect_json(path: Path) -> list[tuple[str, Any]]:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return []

    hits: list[tuple[str, Any]] = []
    for key, value in flatten_json(data):
        if KEY_PATTERNS.search(key):
            hits.append((key, value))
    return hits


def inspect_text(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return []

    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if KEY_PATTERNS.search(line):
            hits.append(f"L{i}: {line.strip()[:300]}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit max-sequence metadata provenance.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["runs", "outputs", "results", "tmp-hf-model", "tmp-hf-tokenizer"],
    )
    parser.add_argument("--expected", type=int, default=128)
    parser.add_argument("--max-size-mb", type=float, default=20)
    args = parser.parse_args()

    max_bytes = int(args.max_size_mb * 1024 * 1024)
    rows: list[tuple[Path, str, Any]] = []

    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            if path.suffix.lower() == ".json":
                for key, value in inspect_json(path):
                    rows.append((path, key, value))
            else:
                for hit in inspect_text(path):
                    rows.append((path, "text", hit))

    if not rows:
        print("No sequence-length metadata found.")
        return

    bad: list[tuple[Path, str, Any]] = []
    print("\nFound sequence-length-related metadata:\n")

    for path, key, value in rows:
        value_str = str(value)
        print(f"{path}\t{key}\t{value_str}")

        nums = [int(x) for x in re.findall(r"\b\d+\b", value_str)]
        if nums and args.expected not in nums and any(n in {256, 512, 1024} for n in nums):
            bad.append((path, key, value))

    print("\nSummary:")
    print(f"  hits: {len(rows)}")
    print(f"  suspicious non-{args.expected} hits: {len(bad)}")

    if bad:
        print("\nSuspicious hits:")
        for path, key, value in bad:
            print(f"  {path}\t{key}\t{value}")


if __name__ == "__main__":
    main()
