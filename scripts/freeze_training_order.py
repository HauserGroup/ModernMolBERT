#!/usr/bin/env python3
"""Freeze seeded ChEMBL source-row IDs for shared train or validation input."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from modernmolbert.utils import file_sha256


def freeze_order(
    source: Path, output: Path, *, seed: int, take: int | None = None
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to replace a frozen training order: {output}")
    rows = pq.ParquetFile(source).metadata.num_rows
    if rows >= np.iinfo(np.uint32).max:
        raise ValueError("Source has too many rows for uint32 row indices")
    if take is not None and not 0 < take <= rows:
        raise ValueError("--take must be between 1 and the number of source rows")
    order = np.random.default_rng(seed).permutation(rows).astype(np.uint32)
    if take is not None:
        order = order[:take]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, order, allow_pickle=False)
    manifest = {
        "source_file": source.name,
        "source_sha256": file_sha256(source),
        "source_rows": rows,
        "selected_rows": len(order),
        "seed": seed,
        "algorithm": "numpy.random.default_rng(seed).permutation(source_rows)",
        "order_sha256": file_sha256(output),
        "order_head": order[:10].tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--take", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(freeze_order(args.source, args.output, seed=args.seed, take=args.take), indent=2)
    )


if __name__ == "__main__":
    main()
