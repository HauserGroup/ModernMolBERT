"""Score one frozen five-model common cohort with fresh, verified inputs."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml"
SCORE = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/score.py"
SPEC = ROOT / "configs/revision_factorial_v1.json"
CAMPAIGN = ROOT / "outputs/revision_factorial_v1/campaign_manifest.json"
SPEC_DATA = json.loads(SPEC.read_text(encoding="utf-8"))
RUN_IDS = set(SPEC_DATA["runs"])


def command_for(
    *, manifest_path: Path, run_id: str, task: str, output_root: Path, n_jobs: int
) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 2 or set(manifest.get("run_ids", [])) != RUN_IDS:
        raise ValueError("Expected the frozen five-model common-row manifest")
    if not CAMPAIGN.is_file() or manifest.get("campaign_manifest_sha256") != file_sha256(CAMPAIGN):
        raise ValueError("Evaluation and campaign manifests differ")
    if manifest.get("common_prefix") != "REVISION_COMMON_":
        raise ValueError("Production scoring requires REVISION_COMMON_ embeddings")
    task_info = manifest["tasks"].get(task)
    if task_info is None or run_id not in task_info["models"]:
        raise ValueError(f"Missing common cohort for {task}/{run_id}")
    configs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["datasets"]
    config_names = [name for name, info in configs.items() if info["name"] == task]
    if len(config_names) != 1:
        raise ValueError(f"Expected one benchmark config for {task}")
    embedder = f"REVISION_COMMON_{run_id}"
    embedding = ROOT / "data/embedded" / task / f"{embedder}.joblib"
    expected = task_info["models"][run_id]["common_embedding_sha256"]
    if not embedding.is_file() or file_sha256(embedding) != expected:
        raise ValueError(f"Common embedding missing or changed: {embedding}")
    prepared = ROOT / "data/prepared" / f"{task}.json"
    if not prepared.is_file() or file_sha256(prepared) != task_info["prepared_sha256"]:
        raise ValueError(f"Frozen prepared dataset missing or changed: {prepared}")
    output = output_root / run_id / f"{task}.csv"
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    return [
        "/opt/lab/bin/uv",
        "run",
        "--locked",
        "python",
        str(SCORE),
        "--datasets",
        config_names[0],
        "--embedder",
        embedder,
        "--heads",
        *SPEC_DATA["evaluation"]["heads"],
        "--missing-labels",
        SPEC_DATA["evaluation"]["missing_labels"],
        "--resume",
        "--no-safe",
        "--n-jobs",
        str(n_jobs),
        "--output-csv",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=sorted(RUN_IDS))
    parser.add_argument("task")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/revision_factorial_v1/evaluation_manifest.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/eval/revision_factorial_v1/common_rows"
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = command_for(
        manifest_path=args.manifest,
        run_id=args.run_id,
        task=args.task,
        output_root=args.output_root,
        n_jobs=args.n_jobs,
    )
    if args.dry_run:
        print(" ".join(command))
        return
    Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
