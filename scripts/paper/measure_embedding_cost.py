#!/usr/bin/env python3
"""Measure the cost of extracting frozen molecular features on one machine.

Times each configuration on one fixed molecule list: ModernMolBERT
checkpoints through the benchmark featurizer (SMILES-to-SELFIES conversion,
lossless-tokenisation checks, tokenisation, encoder forward pass and mean
pooling), and ECFP4 through RDKit (SMILES parsing plus a radius-2, 2,048-bit
Morgan fingerprint, binary or count). ``embedding_cost_worker.py`` performs
each measurement in a fresh process, so peak resident memory belongs to one
configuration. Repeats are interleaved across configurations so that
background load affects them alike.

The molecule list is a seeded sample of distinct prepared test-split SMILES
from the configured benchmark datasets. Throughput counts every input
molecule, including those a featurizer rejects; ``n_valid`` gives coverage.
Absolute throughput depends on the hardware and on other load; compare
configurations only within one run of this script.

Usage:
    uv run python scripts/paper/measure_embedding_cost.py \\
        --checkpoint ModernMolBERT-small=runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_standard/final_model \\
        --checkpoint ModernMolBERT-base=runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_base/final_model \\
        --fingerprints --devices cpu mps --batch-sizes 32 128 --repeats 3 \\
        --output-dir outputs/audit/embedding_cost
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

from build_common_row_benchmark import git_revision
from embedding_cost_worker import ECFP_BITS, ECFP_RADIUS
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.utils import file_sha256

CONFIG = Path("src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml")
WORKER = Path(__file__).resolve().with_name("embedding_cost_worker.py")
FINGERPRINTS = ("ECFP4-binary", "ECFP4-count")
GROUP_COLUMNS = ["name", "kind", "device", "batch_size"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="ModernMolBERT model directory with its bundled tokenizer; repeatable.",
    )
    parser.add_argument("--fingerprints", action="store_true", help="Also time ECFP4.")
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--n-molecules", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.fingerprints:
        parser.error("pass at least one --checkpoint or --fingerprints")
    return args


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        if name in checkpoints:
            raise ValueError(f"Duplicate checkpoint name {name!r}")
        checkpoints[name] = Path(path)
    return checkpoints


def benchmark_test_smiles(
    config: Path, prepared_dir: Path
) -> tuple[list[str], dict[str, str], int]:
    """Return sorted distinct test SMILES, prepared-file hashes and the test-row count."""
    entries = yaml.safe_load(config.read_text())["datasets"]
    distinct: set[str] = set()
    hashes: dict[str, str] = {}
    n_rows = 0
    for entry in entries.values():
        path = prepared_dir / f"{entry['name']}.json"
        dataset = Dataset.deserialize_legacy(path)
        test_rows = list(dataset.splits.get("test", []))
        distinct.update(dataset.data["smiles"].astype(str).iloc[test_rows])
        hashes[entry["name"]] = file_sha256(path)
        n_rows += len(test_rows)
    return sorted(distinct), hashes, n_rows


def sample_molecules(smiles: list[str], n: int | None, seed: int) -> list[str]:
    order = np.random.default_rng(seed).permutation(len(smiles))
    return [smiles[i] for i in order[:n]]


def measurement_specs(
    checkpoints: dict[str, Path],
    *,
    fingerprints: bool,
    devices: list[str],
    batch_sizes: list[int],
    max_seq_length: int,
) -> list[dict[str, object]]:
    """Fingerprints run once on CPU; batch size does not change their path."""
    specs: list[dict[str, object]] = [
        {"name": name, "kind": "fingerprint", "device": "cpu", "batch_size": None, "path": None}
        for name in (FINGERPRINTS if fingerprints else ())
    ]
    specs += [
        {
            "name": name,
            "kind": "checkpoint",
            "device": device,
            "batch_size": batch_size,
            "path": str(path),
        }
        for device in devices
        for batch_size in batch_sizes
        for name, path in checkpoints.items()
    ]
    return [{**spec, "max_seq_length": max_seq_length} for spec in specs]


def run_worker(spec: dict[str, object], molecules_file: Path) -> dict[str, object]:
    env = {**os.environ, "TQDM_DISABLE": "1", "HF_HUB_OFFLINE": "1"}
    load_before = os.getloadavg()[0]
    completed = subprocess.run(
        [sys.executable, str(WORKER), json.dumps(spec), str(molecules_file)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Worker failed for {spec}:\n{completed.stderr}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        **spec,
        **result,
        "load_average_before": load_before,
        "load_average_after": os.getloadavg()[0],
    }


def summarise(measurements: pd.DataFrame) -> pd.DataFrame:
    frame = measurements.assign(
        molecules_per_second=measurements["n_molecules"] / measurements["end_to_end_seconds"],
        first_pass_molecules_per_second=(
            measurements["n_molecules"] / measurements["first_pass_seconds"]
        ),
    )
    summary = (
        frame.groupby(GROUP_COLUMNS, sort=False, dropna=False)
        .agg(
            repeats=("end_to_end_seconds", "size"),
            n_molecules=("n_molecules", "first"),
            n_valid=("n_valid", "first"),
            n_truncated=("n_truncated", "first"),
            median_molecules_per_second=("molecules_per_second", "median"),
            min_molecules_per_second=("molecules_per_second", "min"),
            max_molecules_per_second=("molecules_per_second", "max"),
            median_first_pass_molecules_per_second=("first_pass_molecules_per_second", "median"),
            median_end_to_end_seconds=("end_to_end_seconds", "median"),
            median_conversion_seconds=("conversion_seconds", "median"),
            median_tokenisation_seconds=("tokenisation_seconds", "median"),
            median_features_seconds=("features_seconds", "median"),
            median_import_seconds=("import_seconds", "median"),
            median_load_seconds=("load_seconds", "median"),
            median_rss_start_bytes=("rss_start_bytes", "median"),
            max_peak_rss_bytes=("peak_rss_bytes", "max"),
            max_peak_mps_driver_bytes=("peak_mps_driver_bytes", "max"),
            n_parameters=("n_parameters", "first"),
            feature_dim=("feature_dim", "first"),
        )
        .reset_index()
    )
    return summary.assign(float32_bytes_per_molecule=summary["feature_dim"] * 4)


def checkpoint_record(path: Path) -> dict[str, object]:
    weights = path / "model.safetensors"
    config = json.loads((path / "config.json").read_text())
    return {
        "path": str(path),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": file_sha256(weights),
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
    }


def hardware() -> dict[str, object]:
    brand = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "platform": platform.platform(),
        "processor": brand or platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
    }


def software() -> dict[str, str]:
    import rdkit
    import selfies
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "rdkit": rdkit.__version__,
        "selfies": selfies.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoints = parse_checkpoints(args.checkpoint)
    distinct, prepared_hashes, n_test_rows = benchmark_test_smiles(args.config, args.prepared_dir)
    smiles = sample_molecules(distinct, args.n_molecules, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    molecules_file = args.output_dir / "molecules.txt"
    molecules_file.write_text("\n".join(smiles) + "\n")

    specs = measurement_specs(
        checkpoints,
        fingerprints=args.fingerprints,
        devices=args.devices,
        batch_sizes=args.batch_sizes,
        max_seq_length=args.max_seq_length,
    )
    rows = []
    for repeat in range(args.repeats):
        for spec in specs:
            print(
                f"[repeat {repeat + 1}/{args.repeats}] {spec['name']} "
                f"{spec['device']} batch {spec['batch_size']}",
                flush=True,
            )
            rows.append({"repeat": repeat, **run_worker(spec, molecules_file)})
    measurements = pd.DataFrame(rows)
    summary = summarise(measurements)
    measurements.to_csv(args.output_dir / "measurements.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    manifest = {
        "command": [sys.executable, *sys.argv],
        "git": git_revision(),
        "hardware": hardware(),
        "software": software(),
        "molecules": {
            "source": "distinct prepared test-split SMILES of the configured datasets",
            "n_test_rows": n_test_rows,
            "n_distinct": len(distinct),
            "n_sampled": len(smiles),
            "seed": args.seed,
            "sha256": file_sha256(molecules_file),
            "prepared_sha256": prepared_hashes,
        },
        "ecfp": {
            "implementation": "RDKit Morgan generator",
            "radius": ECFP_RADIUS,
            "bits": ECFP_BITS,
        },
        "checkpoints": {name: checkpoint_record(path) for name, path in checkpoints.items()},
        "max_seq_length": args.max_seq_length,
        "repeats": args.repeats,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
