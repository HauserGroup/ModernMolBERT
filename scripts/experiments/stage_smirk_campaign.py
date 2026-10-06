"""Stage the isolated five-seed SMIRK pilot without changing the paper campaign."""

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import yaml

from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/experimental_smirk_v1.json"
OUTPUT = ROOT / "outputs/experimental_smirk_v1/campaign_manifest.json"
DATASETS = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml"


def build_manifest() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if (
        spec.get("schema") != 1
        or list(spec.get("runs", {})) != ["small_smirk_smiles"]
        or spec.get("seeds") != [42, 43, 44, 45, 46]
    ):
        raise ValueError("Unexpected SMIRK experiment recipe")
    if importlib.metadata.version("smirk") != "0.3.0":
        raise ValueError("Install smirk==0.3.0 without changing the locked main dependencies")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Stage the SMIRK pilot from a clean Git checkout")
    for relative, expected in spec["frozen_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Frozen SMIRK input missing or changed: {path}")
    tokenizer_path = ROOT / "tokenizer/experimental_smirk_v1/smirk_smiles.json"
    metadata = json.loads(tokenizer_path.with_suffix(".metadata.json").read_text())
    if metadata.get("tokenizer_sha256") != file_sha256(tokenizer_path):
        raise ValueError("SMIRK tokenizer metadata hash mismatch")
    registry = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    names = sorted({item["name"] for item in registry.values()} - {"ogbg-moltoxcast"})
    if len(names) != spec["evaluation"]["task_count"]:
        raise ValueError("The frozen benchmark registry no longer has 25 tasks")
    prepared = {}
    for name in names:
        path = ROOT / "data/prepared" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        prepared[name] = file_sha256(path)
    return {
        "schema": 1,
        "campaign": spec["campaign"],
        "code_commit": commit,
        "spec_sha256": file_sha256(SPEC),
        "lockfile_sha256": file_sha256(ROOT / "uv.lock"),
        "frozen_files_sha256": spec["frozen_files"],
        "prepared_data_sha256": prepared,
        "run_ids": list(spec["runs"]),
        "seeds": spec["seeds"],
        "evaluation": spec["evaluation"],
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("smirk", "torch", "transformers", "datasets", "numpy", "scikit-learn")
            },
        },
    }


def main() -> None:
    manifest = build_manifest()
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Existing SMIRK manifest differs: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(encoded, encoding="utf-8")
    print(f"Staged {OUTPUT} ({file_sha256(OUTPUT)})")


if __name__ == "__main__":
    main()
