"""Launch one pinned SMIRK pilot model on an idle Helios GPU."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/experimental_smirk_v1.json"
MANIFEST = ROOT / "outputs/experimental_smirk_v1/campaign_manifest.json"
RUN_ROOT = ROOT / "runs/experimental_smirk_v1/small_smirk_smiles"


def command_for(seed: int, resume: Path | None = None) -> list[str]:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if seed not in spec["seeds"]:
        raise ValueError(f"Seed {seed} is not in the frozen SMIRK recipe")
    common_args = list(spec["common_args"])
    common_args[common_args.index("--seed") + 1] = str(seed)
    command = [
        sys.executable,
        "-m",
        "modernmolbert.train_selfies_ape_modernbert",
        "--output_dir",
        str(RUN_ROOT / f"seed{seed}"),
        "--campaign_manifest",
        str(MANIFEST),
        "--tokenizer_vocab_path",
        "tokenizer/experimental_smirk_v1/smirk_smiles.json",
        "--tokenizer_metadata_path",
        "tokenizer/experimental_smirk_v1/smirk_smiles.metadata.json",
        "--molecule_column",
        "smiles_canonical_clean",
        "--model_size",
        "small",
        *common_args,
    ]
    if resume is not None:
        command.extend(["--resume_from_checkpoint", str(resume)])
    return command


def check_campaign() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Production training requires a clean Git checkout")
    if (
        manifest.get("code_commit") != commit
        or manifest.get("spec_sha256") != file_sha256(SPEC)
        or manifest.get("frozen_files_sha256") != spec["frozen_files"]
    ):
        raise RuntimeError("SMIRK campaign manifest does not match this code and recipe")
    if importlib.metadata.version("smirk") != manifest["environment"]["packages"]["smirk"]:
        raise RuntimeError("SMIRK package version changed since staging")
    for relative, expected in spec["frozen_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Frozen SMIRK input is missing or changed: {relative}")


def check_gpu_idle() -> None:
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    )
    if processes.strip():
        raise RuntimeError(f"Helios GPU is in use:\n{processes.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=[42, 43, 44, 45, 46])
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = command_for(args.seed, args.resume_from_checkpoint)
    if args.dry_run:
        print(" ".join(command))
        return
    check_campaign()
    destination = RUN_ROOT / f"seed{args.seed}"
    if args.resume_from_checkpoint is None and destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Fresh SMIRK run destination is not empty: {destination}")
    check_gpu_idle()
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
