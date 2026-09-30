"""Launch one pinned G1–G7 production encoder on Helios after the preflight gate."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("data/pretrain/chembl36_selfies")
TOKENIZERS = Path("tokenizer/revision_factorial_v1")
RUN_ROOT = Path("runs/revision_factorial_v1")
GATE = Path("outputs/audit/revision_factorial_v1/production_gate.json")
RUNS = {
    "small_ape_selfies": ("ape_selfies", "selfies", "small"),
    "small_ape_smiles": ("ape_smiles", "smiles_canonical_clean", "small"),
    "small_bpe_selfies": ("bpe_selfies", "selfies", "small"),
    "small_bpe_smiles": ("bpe_smiles", "smiles_canonical_clean", "small"),
    "base_ape_selfies": ("ape_selfies", "selfies", "base"),
}
EXPECTED_SHA256 = {
    CORPUS / "train.parquet": "5ba76a62d62c7dc628e5af4a6707eb05f4e03f79d563fd895484e4ac6fccdc7e",
    CORPUS / "valid.parquet": "2426bc7f1514507ef17901db51e8c8bdf056e04d645b567e081c8ae96b12a19e",
    CORPUS
    / "train_order_seed42.npy": "60069ea515283bf5e7fb4ca28196092c41f849a5eb4b1fa157dba6441de126a1",
    CORPUS
    / "validation_rows_seed42_4096.npy": "8f5b52b43d88a766b2c1e806eaa0225b0d83c839bd882dfb9d1c14a4eb020040",
    TOKENIZERS
    / "ape_selfies.json": "8871e9414362c2f1fb313a2dc844a57f077b8e82c7e99195cfb88aa1d741f0f2",
    TOKENIZERS
    / "ape_smiles.json": "a6f40f48409378a8726bac930ecd68d1a57509851388fefa2aeccb6119a98bc5",
    TOKENIZERS
    / "bpe_selfies.json": "e33ac36a5f4e4907689b02d2b22efbea6612c5e56819d0d20d8d950c1782f7f6",
    TOKENIZERS
    / "bpe_smiles.json": "b08c4285f505cb94ba522209e499c291dcd6e9c66fb4252bde0f89c43d169d5f",
    Path("uv.lock"): "52f1cdc215c65ba309f4c3ba6a33acf31add980e235f42c11b356ed51a80b588",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_frozen_inputs(tokenizer: str) -> None:
    needed = {
        path: digest
        for path, digest in EXPECTED_SHA256.items()
        if path.parent != TOKENIZERS or path.name == f"{tokenizer}.json"
    }
    for relative, expected in needed.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen input is missing or has a changed SHA-256: {path}")
    metadata = ROOT / TOKENIZERS / f"{tokenizer}.metadata.json"
    if not metadata.is_file():
        raise RuntimeError(f"Tokenizer metadata is missing: {metadata}")
    if (
        json.loads(metadata.read_text(encoding="utf-8")).get("tokenizer_sha256")
        != needed[TOKENIZERS / f"{tokenizer}.json"]
    ):
        raise RuntimeError(f"Tokenizer metadata does not pin the tokenizer: {metadata}")


def check_launch_gate(gate_path: Path) -> None:
    if not gate_path.is_file():
        raise RuntimeError(f"G4–G7 production gate is missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Production requires a clean Git checkout")
    if gate.get("schema") != 1 or gate.get("ready") is not True or gate.get("git_commit") != head:
        raise RuntimeError("Production gate is incomplete or pinned to another commit")
    if set(gate.get("run_ids", [])) != set(RUNS):
        raise RuntimeError("Production gate must approve exactly the five frozen run IDs")
    for evidence in (
        "pilot_evidence_sha256",
        "capacity_evidence_sha256",
        "coverage_evidence_sha256",
    ):
        value = gate.get(evidence)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"Production gate lacks {evidence}")


def command_for(run_id: str, resume: Path | None) -> list[str]:
    tokenizer, molecule_column, model_size = RUNS[run_id]
    destination = RUN_ROOT / run_id / "seed42"
    command = [
        "/opt/lab/bin/uv",
        "run",
        "--locked",
        "python",
        "-m",
        "modernmolbert.train_selfies_ape_modernbert",
        "--output_dir",
        str(destination),
        "--dataset_name",
        str(CORPUS),
        "--data_files",
        str(CORPUS / "train.parquet"),
        "--molecule_column",
        molecule_column,
        "--train_split",
        "train",
        "--use_validation_split",
        "--validation_split",
        "valid",
        "--validation_row_ids_path",
        str(CORPUS / "validation_rows_seed42_4096.npy"),
        "--global_train_shuffle",
        "--train_order_path",
        str(CORPUS / "train_order_seed42.npy"),
        "--seed",
        "42",
        "--tokenizer_vocab_path",
        str(TOKENIZERS / f"{tokenizer}.json"),
        "--tokenizer_metadata_path",
        str(TOKENIZERS / f"{tokenizer}.metadata.json"),
        "--require_corpus_only_vocab",
        "--unk_rate_threshold",
        "0",
        "--model_size",
        model_size,
        "--max_seq_length",
        "384",
        "--masking_strategy",
        "standard",
        "--mlm_probability",
        "0.15",
        "--max_steps",
        "30000",
        "--learning_rate",
        "0.0004",
        "--optim",
        "adamw_torch",
        "--adam_beta1",
        "0.9",
        "--adam_beta2",
        "0.999",
        "--adam_epsilon",
        "1e-8",
        "--weight_decay",
        "0.01",
        "--max_grad_norm",
        "1.0",
        "--warmup_steps",
        "1500",
        "--per_device_train_batch_size",
        "32",
        "--gradient_accumulation_steps",
        "8",
        "--per_device_eval_batch_size",
        "32",
        "--eval_size",
        "4096",
        "--eval_steps",
        "5000",
        "--save_steps",
        "5000",
        "--save_total_limit",
        "3",
        "--logging_steps",
        "100",
        "--num_workers",
        "4",
        "--device_backend",
        "cuda",
        "--bf16",
        "--no-load_best_model_at_end",
        "--report_to",
        "tensorboard",
    ]
    if resume is not None:
        command.extend(["--resume_from_checkpoint", str(resume)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=RUNS)
    parser.add_argument("--resume_from_checkpoint", type=Path)
    parser.add_argument("--gate_file", type=Path, default=GATE)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    tokenizer = RUNS[args.run_id][0]
    check_frozen_inputs(tokenizer)
    command = command_for(args.run_id, args.resume_from_checkpoint)
    if args.dry_run:
        print(" ".join(command))
        return
    check_launch_gate(ROOT / args.gate_file)
    destination = ROOT / RUN_ROOT / args.run_id / "seed42"
    if args.resume_from_checkpoint is None and destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Fresh production destination is not empty: {destination}")
    if shutil.which("/opt/lab/bin/uv") is None:
        raise RuntimeError("Helios uv runtime is missing: /opt/lab/bin/uv")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
