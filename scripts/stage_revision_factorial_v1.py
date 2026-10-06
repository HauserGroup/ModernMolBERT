"""Verify frozen campaign inputs once and write the shared runtime manifest."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import yaml

from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/revision_factorial_v1.json"
DATASETS = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/revision_factorial_v1/campaign_manifest.json"


def read_spec(path: Path = SPEC) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != 1 or len(spec.get("runs", {})) != 5:
        raise ValueError("Expected the five-run revision factorial campaign spec")
    return spec


def git_commit() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Stage campaign inputs from a clean Git checkout")
    return commit


def benchmark_names(spec: dict) -> list[str]:
    configs = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    names = sorted(
        info["name"]
        for info in configs.values()
        if info["name"] != spec["evaluation"]["excluded_task"]
    )
    if len(names) != spec["evaluation"]["task_count"] or len(set(names)) != len(names):
        raise ValueError("Frozen benchmark registry does not match the campaign spec")
    return names


def build_manifest(spec_path: Path = SPEC) -> dict:
    spec = read_spec(spec_path)
    commit = git_commit()
    for relative, expected in spec["frozen_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Frozen campaign input missing or changed: {path}")
        if relative.startswith("tokenizer/") and not relative.endswith(".metadata.json"):
            metadata = path.with_suffix(".metadata.json")
            if (
                not metadata.is_file()
                or json.loads(metadata.read_text(encoding="utf-8")).get("tokenizer_sha256")
                != expected
            ):
                raise ValueError(f"Tokenizer metadata does not match {path}")
    prepared = {}
    for name in benchmark_names(spec):
        path = ROOT / "data/prepared" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen benchmark task is missing: {path}")
        prepared[name] = file_sha256(path)
    baseline = ROOT / spec["baseline_csv"]
    if not baseline.is_file():
        raise FileNotFoundError(f"Imported baseline table is missing: {baseline}")
    return {
        "schema": 1,
        "campaign": spec["campaign"],
        "code_commit": commit,
        "spec_sha256": file_sha256(spec_path),
        "lockfile_sha256": file_sha256(ROOT / "uv.lock"),
        "frozen_files_sha256": spec["frozen_files"],
        "prepared_data_sha256": prepared,
        "baseline_csv": spec["baseline_csv"],
        "baseline_sha256": file_sha256(baseline),
        "run_ids": list(spec["runs"]),
        "seeds": spec.get("seeds", [42]),
        "evaluation": spec["evaluation"],
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "numpy", "scikit-learn")
            },
        },
    }


def write_manifest(output: Path, manifest: dict) -> None:
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Existing campaign manifest differs: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.spec)
    output = args.output or ROOT / "outputs" / manifest["campaign"] / "campaign_manifest.json"
    write_manifest(output, manifest)
    print(f"Verified {len(manifest['frozen_files_sha256'])} frozen inputs")
    print(f"Verified {len(manifest['prepared_data_sha256'])} benchmark tasks")
    print(f"Campaign manifest: {output} ({file_sha256(output)})")


if __name__ == "__main__":
    main()
