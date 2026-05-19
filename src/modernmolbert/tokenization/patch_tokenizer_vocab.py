"""
patch_tokenizer_vocab.py
------------------------
Inject missing SELFIES primitive symbols into an already-trained APE tokenizer
vocab file without retraining.

The vocab JSON is a flat {"token": id} mapping. This script:
  1. Reads symbols from --extra_file (one per line, # lines ignored).
  2. Appends any symbols not already present, assigning successive IDs.
  3. Writes the result to --output_file.
  4. Recomputes SHA256 and updates the companion metadata JSON (same stem,
     _metadata.json suffix) if one exists alongside the output file.

Usage (run from the project root):
    uv run python patch_tokenizer_vocab.py \\
        --input_file  tokenizer/chembl36_selfies_2m_ape_max8.json \\
        --extra_file  tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt \\
        --output_file tokenizer/chembl36_selfies_2m_ape_max8.json

    # preview without writing:
    uv run python patch_tokenizer_vocab.py ... --dry_run
"""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

SELFIES_SYMBOL_RE = re.compile(r"\[[^\]]+\]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch missing SELFIES symbols into a trained APE tokenizer vocab JSON."
    )
    p.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Existing vocab JSON file to patch.",
    )
    p.add_argument(
        "--extra_file",
        type=Path,
        required=True,
        help="Text file with one SELFIES primitive per line (# lines ignored).",
    )
    p.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Destination for the patched vocab JSON (may be the same as --input_file).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would change without writing anything.",
    )
    return p.parse_args()


def load_symbols(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            symbols.append(token)
    return symbols


def validate_symbols(symbols: list[str]) -> None:
    malformed = [s for s in symbols if SELFIES_SYMBOL_RE.fullmatch(s) is None]
    if malformed:
        raise ValueError(f"Malformed SELFIES symbols (must be [bracket] form): {malformed}")


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()

    for path, label in [(args.input_file, "--input_file"), (args.extra_file, "--extra_file")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    symbols = load_symbols(args.extra_file)
    print(f"Loaded {len(symbols)} symbols from {args.extra_file}")
    validate_symbols(symbols)

    vocab: dict[str, int] = json.loads(args.input_file.read_text(encoding="utf-8"))
    vocab_size_before = len(vocab)
    next_id = max(vocab.values()) + 1

    to_add = [s for s in symbols if s not in vocab]
    already_present = len(symbols) - len(to_add)

    print(f"Vocab size before : {vocab_size_before}")
    print(f"Symbols requested : {len(symbols)}")
    print(f"Already present   : {already_present}")
    print(f"To add            : {len(to_add)}")

    if args.dry_run:
        print("\n[DRY RUN] Symbols that would be added:")
        for i, s in enumerate(to_add):
            print(f"  {s!r:30s} -> id {next_id + i}")
        print("\n[DRY RUN] No files written.")
        return

    if not to_add:
        print("Nothing to do — all symbols already present.")
        return

    for symbol in to_add:
        vocab[symbol] = next_id
        next_id += 1

    vocab_size_after = len(vocab)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(vocab, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nVocab written to  : {args.output_file}")
    print(f"Vocab size after  : {vocab_size_after}  (+{vocab_size_after - vocab_size_before})")

    new_sha256 = sha256_of_file(args.output_file)
    print(f"SHA256            : {new_sha256}")

    # Update companion metadata JSON if it exists next to the output file.
    metadata_path = args.output_file.with_name(args.output_file.stem + "_metadata.json")
    if metadata_path.exists():
        metadata: dict = json.loads(metadata_path.read_text(encoding="utf-8"))

        patch_record = {
            "patched_at_utc": datetime.now(UTC).isoformat(),
            "patch_script": "patch_tokenizer_vocab.py",
            "input_file": str(args.input_file),
            "extra_file": str(args.extra_file),
            "output_file": str(args.output_file),
            "symbols_requested": len(symbols),
            "symbols_added": len(to_add),
            "symbols_already_present": already_present,
            "vocab_size_before": vocab_size_before,
            "vocab_size_after": vocab_size_after,
            "added_symbols": to_add,
        }

        metadata["vocab_size"] = vocab_size_after
        metadata["tokenizer_sha256"] = new_sha256
        metadata["patch_history"] = metadata.get("patch_history", []) + [patch_record]

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Metadata updated  : {metadata_path}")
    else:
        print(f"No metadata file at {metadata_path} — skipping.")

    print("\nPatch complete.")
    if to_add:
        print("Added:", to_add)


if __name__ == "__main__":
    main()
