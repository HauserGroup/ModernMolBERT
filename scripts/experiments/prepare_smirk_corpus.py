"""Exclude non-round-tripping/over-context SMILES while preserving source order."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modernmolbert.tokenization.load import load_verified_tokenizer
from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/pretrain/chembl36_selfies"
DESTINATION = ROOT / "data/pretrain/chembl36_smirk"
TOKENIZER = ROOT / "tokenizer/experimental_smirk_v1/smirk_smiles.json"
COLUMN = "smiles_canonical_clean"


def prepare(destination: Path, max_seq_length: int) -> dict:
    train_path = destination / "train.parquet"
    order_path = destination / "train_order_seed42.npy"
    summary_path = destination / "preparation.json"
    if any(path.exists() for path in (train_path, order_path, summary_path)):
        raise FileExistsError(f"SMIRK corpus destination must be empty: {destination}")
    tokenizer, metadata, _, _ = load_verified_tokenizer(TOKENIZER)
    source_path = SOURCE / "train.parquet"
    source_order_path = SOURCE / "train_order_seed42.npy"
    source_order = np.load(source_order_path, allow_pickle=False)
    parquet = pq.ParquetFile(source_path)
    destination.mkdir(parents=True, exist_ok=True)

    masks: list[np.ndarray] = []
    excluded: list[dict] = []
    writer = pq.ParquetWriter(train_path, pa.schema([(COLUMN, pa.large_string())]))
    row_offset = 0
    try:
        for batch in parquet.iter_batches(batch_size=10_000, columns=[COLUMN]):
            keep = np.ones(batch.num_rows, dtype=bool)
            for position, smiles in enumerate(batch.column(0).to_pylist()):
                if not isinstance(smiles, str) or not smiles:
                    reason = "empty"
                else:
                    tokens = tokenizer.tokenize(smiles)
                    if tokenizer.unk_token in tokens or "".join(tokens) != smiles:
                        reason = "not_lossless"
                    elif len(tokens) + 2 > max_seq_length:
                        reason = "over_context"
                    else:
                        reason = ""
                if reason:
                    keep[position] = False
                    excluded.append({"source_row": row_offset + position, "reason": reason})
            masks.append(keep)
            writer.write_batch(batch.filter(pa.array(keep)))
            row_offset += batch.num_rows
    finally:
        writer.close()

    mask = np.concatenate(masks)
    if row_offset != len(source_order) or not np.issubdtype(source_order.dtype, np.integer):
        raise ValueError("Original training order does not match the corpus")
    remap = np.cumsum(mask, dtype=np.uint32) - 1
    retained_order = remap[source_order[mask[source_order]]]
    if len(retained_order) != int(mask.sum()) or not np.array_equal(
        np.sort(retained_order), np.arange(len(retained_order))
    ):
        raise ValueError("Filtered training order is not a complete permutation")
    np.save(order_path, retained_order, allow_pickle=False)
    summary = {
        "source_train_sha256": file_sha256(source_path),
        "source_order_sha256": file_sha256(source_order_path),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "max_seq_length": max_seq_length,
        "source_rows": row_offset,
        "retained_rows": int(mask.sum()),
        "excluded_rows": excluded,
        "filtered_train_sha256": file_sha256(train_path),
        "filtered_order_sha256": file_sha256(order_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--max-seq-length", type=int, default=384)
    args = parser.parse_args()
    summary = prepare(args.destination, args.max_seq_length)
    print(
        f"Retained {summary['retained_rows']}/{summary['source_rows']} training rows; "
        f"excluded {len(summary['excluded_rows'])}."
    )
    print(f"Filtered Parquet SHA-256: {summary['filtered_train_sha256']}")
    print(f"Filtered order SHA-256: {summary['filtered_order_sha256']}")


if __name__ == "__main__":
    main()
