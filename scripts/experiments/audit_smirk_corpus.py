"""Audit every pretraining molecule for lossless SMIRK coverage and context length."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from modernmolbert.tokenization.load import load_verified_tokenizer
from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = ROOT / "tokenizer/experimental_smirk_v1/smirk_smiles.json"
TRAIN = ROOT / "data/pretrain/chembl36_smirk/train.parquet"
VALID = ROOT / "data/pretrain/chembl36_selfies/valid.parquet"
VALID_ROWS = ROOT / "data/pretrain/chembl36_selfies/validation_rows_seed42_4096.npy"
OUTPUT = ROOT / "outputs/experimental_smirk_v1/tokenizer_corpus_audit.json"


def audit(
    path: Path, tokenizer: object, max_length: int, selected_rows: set[int] | None = None
) -> dict:
    parquet = pq.ParquetFile(path)
    lengths: list[int] = []
    unknown_rows = 0
    lossy_rows = 0
    over_context_rows = 0
    examples: list[dict] = []
    row_id = 0
    for batch in parquet.iter_batches(batch_size=10_000, columns=["smiles_canonical_clean"]):
        for value in batch.column(0).to_pylist():
            if selected_rows is not None and row_id not in selected_rows:
                row_id += 1
                continue
            smiles = str(value)
            tokens = tokenizer.tokenize(smiles)  # type: ignore[attr-defined]
            unknown = tokenizer.unk_token in tokens  # type: ignore[attr-defined]
            lossy = "".join(tokens) != smiles
            length = len(tokens) + 2
            lengths.append(length)
            unknown_rows += int(unknown)
            lossy_rows += int(lossy)
            over_context_rows += int(length > max_length)
            if (unknown or lossy or length > max_length) and len(examples) < 10:
                examples.append(
                    {
                        "smiles": smiles,
                        "length": length,
                        "unknown": unknown,
                        "lossy": lossy,
                    }
                )
            row_id += 1
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "source_sha256": file_sha256(path),
        "rows": len(lengths),
        "unknown_rows": unknown_rows,
        "lossy_rows": lossy_rows,
        "over_context_rows": over_context_rows,
        "mean_length": float(values.mean()),
        "p95_length": float(np.percentile(values, 95)),
        "p99_length": float(np.percentile(values, 99)),
        "max_length": int(values.max()),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-seq-length", type=int, default=384)
    args = parser.parse_args()
    tokenizer, metadata, _, _ = load_verified_tokenizer(TOKENIZER)
    selected_validation = set(map(int, np.load(VALID_ROWS, allow_pickle=False)))
    results = {
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "max_seq_length": args.max_seq_length,
        "splits": {
            "train": audit(TRAIN, tokenizer, args.max_seq_length),
            "selected_validation": audit(
                VALID, tokenizer, args.max_seq_length, selected_validation
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    if any(
        row["unknown_rows"] or row["lossy_rows"] or row["over_context_rows"]
        for row in results["splits"].values()
    ):
        raise SystemExit("SMIRK corpus preflight found unsupported or over-context rows")


if __name__ == "__main__":
    main()
