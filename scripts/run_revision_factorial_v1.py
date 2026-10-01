"""Launch one pinned revision encoder from the staged campaign manifest."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/revision_factorial_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/revision_factorial_v1/campaign_manifest.json"
RUN_ROOT = Path("runs/revision_factorial_v1")


def read_spec(path: Path | None = None) -> dict:
    path = SPEC if path is None else path
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != 1 or len(spec.get("runs", {})) != 5:
        raise ValueError("Expected the five-run revision factorial campaign spec")
    if not spec.get("seeds", [42]) or len(set(spec.get("seeds", [42]))) != len(
        spec.get("seeds", [42])
    ):
        raise ValueError("Campaign seeds must be a nonempty unique list")
    return spec


def check_campaign_manifest(path: Path, spec: dict, spec_path: Path | None = None) -> None:
    spec_path = SPEC if spec_path is None else spec_path
    if not path.is_file():
        raise FileNotFoundError(f"Stage the campaign inputs before launch: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Production requires a clean Git checkout")
    if (
        manifest.get("schema") != 1
        or manifest.get("code_commit") != commit
        or manifest.get("spec_sha256") != file_sha256(spec_path)
        or manifest.get("run_ids") != list(spec["runs"])
        or manifest.get("seeds", [42]) != spec.get("seeds", [42])
    ):
        raise RuntimeError("Campaign manifest does not match this code and five-run recipe")
    if manifest.get("frozen_files_sha256") != spec["frozen_files"]:
        raise RuntimeError("Campaign manifest does not match the frozen input recipe")
    for relative, expected in spec["frozen_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Staged campaign input is missing or changed: {relative}")


def check_gpu_available() -> None:
    """Leave the shared Helios GPU alone while another compute process is active."""
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    )
    if processes.strip():
        raise RuntimeError(f"Helios GPU is already in use:\n{processes.strip()}")


def command_for(
    run_id: str,
    resume: Path | None,
    manifest: Path = DEFAULT_OUTPUT,
    spec: dict | None = None,
    seed: int = 42,
) -> list[str]:
    spec = read_spec() if spec is None else spec
    if seed not in spec.get("seeds", [42]):
        raise ValueError(f"Seed {seed} is not in the frozen campaign recipe")
    run = spec["runs"][run_id]
    tokenizer = run["tokenizer"]
    tokenizer_root = Path("tokenizer/revision_factorial_v1")
    destination = RUN_ROOT / run_id / f"seed{seed}"
    common_args = list(spec["common_args"])
    seed_flag = common_args.index("--seed") + 1
    common_args[seed_flag] = str(seed)
    command = [
        "/opt/lab/bin/uv",
        "run",
        "--locked",
        "python",
        "-m",
        "modernmolbert.train_selfies_ape_modernbert",
        "--output_dir",
        str(destination),
        "--campaign_manifest",
        str(manifest),
        "--tokenizer_vocab_path",
        str(tokenizer_root / f"{tokenizer}.json"),
        "--tokenizer_metadata_path",
        str(tokenizer_root / f"{tokenizer}.metadata.json"),
        "--molecule_column",
        run["molecule_column"],
        "--model_size",
        run["model_size"],
        *common_args,
    ]
    if resume is not None:
        command.extend(["--resume_from_checkpoint", str(resume)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    spec = read_spec(args.spec)
    if args.run_id not in spec["runs"]:
        parser.error(f"Unknown run ID: {args.run_id}")
    manifest = args.manifest or ROOT / "outputs" / spec["campaign"] / "campaign_manifest.json"
    command = command_for(args.run_id, args.resume_from_checkpoint, manifest, spec, args.seed)
    if args.dry_run:
        print(" ".join(command))
        return
    check_campaign_manifest(manifest, spec, args.spec)
    destination = ROOT / RUN_ROOT / args.run_id / f"seed{args.seed}"
    if args.resume_from_checkpoint is None and destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Fresh production destination is not empty: {destination}")
    if shutil.which("/opt/lab/bin/uv") is None:
        raise RuntimeError("Helios uv runtime is missing: /opt/lab/bin/uv")
    check_gpu_available()
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
