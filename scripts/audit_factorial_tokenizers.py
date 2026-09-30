#!/usr/bin/env python3
"""Audit all four factorial tokenizers on every frozen ChEMBL train/valid row."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import selfies as sf
from rdkit import Chem

from modernmolbert.tokenization.load import load_verified_tokenizer
from modernmolbert.utils import file_sha256

VARIANTS = {
    "ape_selfies": ("ape_selfies.json", "selfies"),
    "ape_smiles": ("ape_smiles.json", "smiles_canonical_clean"),
    "bpe_selfies": ("bpe_selfies.json", "selfies"),
    "bpe_smiles": ("bpe_smiles.json", "smiles_canonical_clean"),
}


def _summary(lengths: list[int], unknown: int, lossy: int, longest_row: int) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "rows": len(values),
        "mean_including_bos_eos": float(values.mean()),
        "p50": int(np.percentile(values, 50, method="nearest")),
        "p95": int(np.percentile(values, 95, method="nearest")),
        "p99": int(np.percentile(values, 99, method="nearest")),
        "max": int(values.max()),
        "longest_source_row_index": longest_row,
        "over_128": int((values > 128).sum()),
        "over_192": int((values > 192).sum()),
        "over_256": int((values > 256).sum()),
        "unknown_rows": unknown,
        "lossy_rows": lossy,
    }


def audit_split(path: Path, tokenizers: dict[str, Any], *, check_pairs: bool) -> dict[str, Any]:
    columns = ["smiles_canonical_clean", "selfies"]
    parquet = pq.ParquetFile(path)
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise ValueError(f"Missing paired molecule columns in {path}")
    tracked = {
        name: {"lengths": [], "unknown": 0, "lossy": 0, "longest_row": -1} for name in tokenizers
    }
    pair_failures: list[dict[str, Any]] = []
    n_pairs_failed = 0
    row_offset = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        records = batch.to_pydict()
        for name, tokenizer in tokenizers.items():
            column = VARIANTS[name][1]
            sequences = records[column]
            if any(not isinstance(value, str) or not value for value in sequences):
                raise ValueError(f"Missing {column} in {path} near source row {row_offset}")
            encoded = tokenizer(
                sequences,
                padding=False,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            record = tracked[name]
            lengths = record["lengths"]
            for index, (sequence, ids) in enumerate(zip(sequences, encoded, strict=True)):
                source_row = row_offset + index
                length = len(ids)
                if not lengths or length > lengths[record["longest_row"]]:
                    record["longest_row"] = source_row
                lengths.append(length)
                record["unknown"] += int(tokenizer.unk_token_id in ids)
                pieces = tokenizer.convert_ids_to_tokens(ids[1:-1])
                record["lossy"] += int("".join(pieces) != sequence)

        if check_pairs:
            for index, (smiles, selfies) in enumerate(
                zip(records["smiles_canonical_clean"], records["selfies"], strict=True)
            ):
                try:
                    molecule = Chem.MolFromSmiles(sf.decoder(selfies))
                    decoded = (
                        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                        if molecule is not None
                        else None
                    )
                    # Older RDKit wrote some valid aromatic systems in a
                    # different canonical SMILES form. Canonicalize the saved
                    # source with the same current RDKit before comparison.
                    expected = smiles
                    if decoded != smiles:
                        source_molecule = Chem.MolFromSmiles(smiles)
                        expected = (
                            Chem.MolToSmiles(source_molecule, canonical=True, isomericSmiles=True)
                            if source_molecule is not None
                            else None
                        )
                except Exception:
                    decoded = None
                    expected = smiles
                if decoded is None or expected is None or decoded != expected:
                    n_pairs_failed += 1
                    if len(pair_failures) < 20:
                        pair_failures.append(
                            {
                                "source_row_index": row_offset + index,
                                "smiles": smiles,
                                "decoded_selfies_smiles": decoded,
                            }
                        )
        row_offset += batch.num_rows
        if row_offset % 200_000 < batch.num_rows:
            print(f"{path.name}: {row_offset:,}/{parquet.metadata.num_rows:,}", flush=True)

    if row_offset != parquet.metadata.num_rows:
        raise ValueError(f"Row count mismatch in {path}")
    return {
        "file": str(path),
        "sha256": file_sha256(path),
        "rows": row_offset,
        "variants": {
            name: _summary(
                record["lengths"], record["unknown"], record["lossy"], record["longest_row"]
            )
            for name, record in tracked.items()
        },
        "pair_identity_failures": n_pairs_failed if check_pairs else None,
        "pair_failure_examples": pair_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("tokenizer/revision_factorial_v1")
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/pretrain/chembl36_selfies"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/audit/revision_factorial_v1/tokenizers.json")
    )
    parser.add_argument("--check-pairs", action="store_true")
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="Recheck paired identity while preserving completed tokenizer statistics in output.",
    )
    args = parser.parse_args()

    tokenizers = {}
    metadata = {}
    for name, (filename, _column) in VARIANTS.items():
        tokenizer, details, _, _ = load_verified_tokenizer(args.tokenizer_dir / filename)
        tokenizers[name] = tokenizer
        metadata[name] = details
    samples = [details.get("aligned_sample") for details in metadata.values()]
    if not samples[0] or any(sample != samples[0] for sample in samples[1:]):
        raise ValueError("Four tokenizer artifacts do not share one sampled source-row order")
    train_path = args.corpus_dir / "train.parquet"
    if samples[0]["source_sha256"] != file_sha256(train_path):
        raise ValueError("Tokenizer sample source differs from the training Parquet")

    if args.pairs_only:
        if not args.output.is_file():
            raise FileNotFoundError("--pairs-only requires an existing audit output")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        for split in ("train", "valid"):
            checked = audit_split(args.corpus_dir / f"{split}.parquet", {}, check_pairs=True)
            if checked["sha256"] != result["splits"][split]["sha256"]:
                raise ValueError(f"{split} corpus changed since token-length audit")
            result["splits"][split]["pair_identity_failures"] = checked["pair_identity_failures"]
            result["splits"][split]["pair_failure_examples"] = checked["pair_failure_examples"]
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            "Paired-identity failures:",
            {s: result["splits"][s]["pair_identity_failures"] for s in ("train", "valid")},
        )
        if any(result["splits"][s]["pair_identity_failures"] for s in ("train", "valid")):
            raise RuntimeError("Paired-identity audit failed")
        return

    splits = {
        split: audit_split(
            args.corpus_dir / f"{split}.parquet", tokenizers, check_pairs=args.check_pairs
        )
        for split in ("train", "valid")
    }
    longest = max(
        summary["max"] for split in splits.values() for summary in split["variants"].values()
    )
    result = {
        "sample": samples[0],
        "tokenizers": {
            name: {
                "sha256": metadata[name]["tokenizer_sha256"],
                "vocab_size": metadata[name]["vocab_size"],
                "algorithm": metadata[name]["algorithm"],
                "representation": metadata[name]["representation"],
            }
            for name in VARIANTS
        },
        "splits": splits,
        "common_context_rounded_to_64": int(math.ceil(longest / 64) * 64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Common context candidate: {result['common_context_rounded_to_64']}")
    if any(
        summary["unknown_rows"] or summary["lossy_rows"]
        for split in splits.values()
        for summary in split["variants"].values()
    ) or (args.check_pairs and any(split["pair_identity_failures"] for split in splits.values())):
        raise RuntimeError(f"Tokenizer or paired-identity audit failed; inspect {args.output}")


if __name__ == "__main__":
    main()
