"""Export verified identities for all 25 frozen revision encoders as CSV.

Run on Helios against the shared run and backup volumes. The output contains
hashes and recorded configuration, never model weights or private paths.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
RUN_IDS = (
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
)
FIELDS = (
    "run_id",
    "seed",
    "model_size",
    "representation",
    "tokenizer_algorithm",
    "vocab_size",
    "num_parameters",
    "context_tokens",
    "training_steps",
    "molecule_presentations",
    "train_runtime_seconds",
    "training_code_commit",
    "campaign_manifest_sha256",
    "run_identity_sha256",
    "final_model_sha256",
    "tokenizer_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path, backup_root: Path) -> list[dict[str, object]]:
    campaigns = {
        42: root / "outputs/revision_factorial_v1/campaign_manifest.json",
        43: root / "outputs/revision_factorial_multiseed_v1/campaign_manifest.json",
    }
    rows = []
    for seed in SEEDS:
        campaign_path = campaigns[42 if seed == 42 else 43]
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        campaign_hash = sha256(campaign_path)
        if set(campaign["run_ids"]) != set(RUN_IDS) or seed not in campaign.get("seeds", [42]):
            raise ValueError(f"Campaign has a different model/seed cohort: {campaign_path}")
        for run_id in RUN_IDS:
            run = root / "runs/revision_factorial_v1" / run_id / f"seed{seed}"
            backup = backup_root / run_id / f"seed{seed}"
            identity_path = run / "run_identity.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            result = identity["result"]
            args = identity["args"]
            metrics = json.loads((run / "train_results.json").read_text(encoding="utf-8"))
            weights = run / "final_model" / result["final_model_file"]
            backup_weights = backup / "final_model" / result["final_model_file"]
            if not (
                identity["schema"] == 2
                and args["seed"] == seed
                and identity["git"] == {"commit": campaign["code_commit"], "dirty": False}
                and identity["inputs"]["campaign_manifest_sha256"] == campaign_hash
                and result["terminal_step"] == result["selected_step"] == 30_000
                and metrics["train_samples_streaming"] == 7_680_000
                and sha256(weights) == result["final_model_sha256"]
                and sha256(backup_weights) == result["final_model_sha256"]
                and identity_path.read_bytes() == (backup / "run_identity.json").read_bytes()
            ):
                raise ValueError(f"Unaccepted run or backup: {run_id} seed {seed}")
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "model_size": args["model_size"],
                    "representation": args["representation"],
                    "tokenizer_algorithm": args["tokenizer_algorithm"],
                    "vocab_size": result["vocab_size"],
                    "num_parameters": result["num_parameters"],
                    "context_tokens": args["max_seq_length"],
                    "training_steps": result["terminal_step"],
                    "molecule_presentations": metrics["train_samples_streaming"],
                    "train_runtime_seconds": metrics["train_runtime"],
                    "training_code_commit": campaign["code_commit"],
                    "campaign_manifest_sha256": campaign_hash,
                    "run_identity_sha256": sha256(identity_path),
                    "final_model_sha256": result["final_model_sha256"],
                    "tokenizer_sha256": result["tokenizer_sha256"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root, args.backup_root)
    writer = csv.DictWriter[str](sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
