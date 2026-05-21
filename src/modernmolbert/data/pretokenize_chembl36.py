#!/usr/bin/env python3
"""Deprecated ChEMBL36 SELFIES pre-tokenization helper.

This script is retained for reproducing older experiments. New pretraining runs
should use ``data/pretrain/chembl36_selfies`` directly and let
``modernmolbert.train_selfies_ape_modernbert`` tokenize SELFIES rows with the
current tokenizer at training time.

Run once before training:
    uv run python src/modernmolbert/data/pretokenize_chembl36.py

Input:  data/pretrain/chembl36_selfies/{train,valid}/*.parquet
        (SELFIES column: "selfies")
Output: data/pretrain/chembl36_selfies_tokenized/{train,valid}/*.parquet
        (adds column: "input_ids" — list[int], includes BOS and EOS)

The generated ``input_ids`` are tied to one tokenizer vocabulary. Reusing them
with a different tokenizer silently trains on the wrong token IDs.
"""

import json
import re
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# ── config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = Path("tokenizer/chembl36_selfies_2m_min2000.json")
INPUT_ROOT = Path("data/pretrain/chembl36_selfies")
OUTPUT_ROOT = Path("data/pretrain/chembl36_selfies_tokenized")
SPLITS = ["train", "valid"]
SELFIES_COL = "selfies"
NUM_WORKERS = min(cpu_count(), 8)  # capped at 8 — usually I/O bound before that

DEPRECATION_MESSAGE = (
    "modernmolbert.data.pretokenize_chembl36 is deprecated. "
    "Use data/pretrain/chembl36_selfies directly for training so SELFIES are "
    "encoded with the active tokenizer."
)

# ── tokenizer (loaded once per worker via initializer) ────────────────────────
SELFIES_RE = re.compile(r"\[[^\]]+\]")
_vocab: dict[str, int] = {}
_bos_id: int = 0
_eos_id: int = 2
_unk_id: int = 3


def _init_worker(vocab_path: str) -> None:
    global _vocab, _bos_id, _eos_id, _unk_id
    with open(vocab_path, encoding="utf-8") as f:
        _vocab = json.load(f)
    _bos_id = _vocab.get("<s>", 0)
    _eos_id = _vocab.get("</s>", 2)
    _unk_id = _vocab.get("<unk>", 3)


def _tokenize_one(selfies: str) -> list[int]:
    """Greedy longest-match APE tokenize, identical logic to ape_tokenize()."""
    pieces = SELFIES_RE.findall(selfies)
    if not pieces:
        return [_bos_id, _unk_id, _eos_id]

    ids: list[int] = [_bos_id]
    i = 0
    while i < len(pieces):
        for j in range(len(pieces), i, -1):
            candidate = "".join(pieces[i:j])
            if candidate in _vocab:
                ids.append(_vocab[candidate])
                i = j
                break
        else:
            ids.append(_unk_id)
            i += 1
    ids.append(_eos_id)
    return ids


def _process_shard(args: tuple[Path, Path]) -> tuple[str, int]:
    src, dst = args
    df = pd.read_parquet(src)
    results = []
    with tqdm(total=len(df), desc=src.name, unit="mol", leave=False) as bar:
        for selfies in df[SELFIES_COL]:
            results.append(_tokenize_one(selfies))
            bar.update(1)
    df["input_ids"] = results
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return src.name, len(df)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    jobs: list[tuple[Path, Path]] = []
    for split in SPLITS:
        src_dir = INPUT_ROOT / split
        dst_dir = OUTPUT_ROOT / split
        shards = sorted(src_dir.glob("*.parquet"))
        if not shards:
            print(f"  No parquet shards found in {src_dir}, skipping.")
            continue
        for shard in shards:
            dst = dst_dir / shard.name
            if dst.exists():
                print(f"  SKIP (exists): {dst}")
            else:
                jobs.append((shard, dst))

    if not jobs:
        print("Nothing to do — all shards already tokenized.")
        return

    print(f"Tokenizing {len(jobs)} shard(s) with {NUM_WORKERS} worker(s)...")
    t0 = time.perf_counter()
    total_rows = 0

    with (
        Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(str(TOKENIZER_PATH),),
        ) as pool,
        tqdm(total=len(jobs), desc="shards", unit="shard") as shard_bar,
    ):
        for name, n_rows in pool.imap_unordered(_process_shard, jobs):
            total_rows += n_rows
            elapsed = time.perf_counter() - t0
            shard_bar.update(1)
            tqdm.write(f"  done: {name}  ({n_rows:,} rows)  [{elapsed:.1f}s elapsed]")

    elapsed = time.perf_counter() - t0
    print(
        f"\nFinished: {total_rows:,} molecules in {elapsed:.1f}s "
        f"({total_rows / elapsed:,.0f} mol/s)"
    )

    metadata = {
        "selfies_column": SELFIES_COL,
        "pretokenized": True,
        "tokenizer_path": str(TOKENIZER_PATH),
        "deprecated": True,
        "deprecation_message": DEPRECATION_MESSAGE,
    }
    meta_path = OUTPUT_ROOT / "metadata.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {meta_path}")

    print(f"Output: {OUTPUT_ROOT}/")
    print(
        "\nDeprecated output. Prefer training with:\n"
        f"  --dataset_name {INPUT_ROOT}\n"
        "  --use_validation_split --validation_split valid\n"
        "  (SELFIES are tokenized with the active tokenizer at training time)"
    )


if __name__ == "__main__":
    main()
