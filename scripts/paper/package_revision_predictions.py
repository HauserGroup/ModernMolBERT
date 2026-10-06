#!/usr/bin/env python3
"""Build a deterministic tar of the 625 verified selected predictions."""

import argparse
import csv
import hashlib
import io
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = [Path(row["prediction_path"]) for row in rows]
    if len(rows) != 625 or len(set(paths)) != 625:
        raise ValueError("Expected 625 unique selected prediction archives")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w", format=tarfile.PAX_FORMAT) as tar:
        manifest_bytes = args.manifest.read_bytes()
        metadata = tarfile.TarInfo("prediction_release_manifest.csv")
        metadata.size = len(manifest_bytes)
        metadata.mtime = 0
        metadata.mode = 0o644
        tar.addfile(metadata, io.BytesIO(manifest_bytes))
        for row, relative in sorted(zip(rows, paths, strict=True), key=lambda pair: pair[1]):
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe prediction path: {relative}")
            source = args.repo_root / relative
            if source.stat().st_size != int(row["archive_size_bytes"]):
                raise ValueError(f"Prediction archive size mismatch: {relative}")
            if sha256(source) != row["archive_sha256"]:
                raise ValueError(f"Prediction archive hash mismatch: {relative}")
            metadata = tarfile.TarInfo(relative.as_posix())
            metadata.size = source.stat().st_size
            metadata.mtime = 0
            metadata.mode = 0o644
            with source.open("rb") as handle:
                tar.addfile(metadata, handle)
    print(f"Wrote {len(rows)} selected archives to {args.output}")
    print(f"Size: {args.output.stat().st_size} bytes; SHA-256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
