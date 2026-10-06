#!/usr/bin/env python3
"""Stage one verified revision checkpoint with a representation-aware card."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_card(identity: dict, run_id: str, seed: int, repo_id: str) -> str:
    args = identity["args"]
    result = identity["result"]
    representation = args["representation"].upper()
    algorithm = args["tokenizer_algorithm"].upper()
    return f"""---
license: mit
library_name: transformers
pipeline_tag: fill-mask
tags:
- chemistry
- molecules
- {representation.lower()}
- {algorithm.lower()}-tokenizer
- modernbert
---

# {repo_id}

ModernMolBERT {run_id}, seed {seed}. This checkpoint is one of five independent
training seeds in a controlled five-configuration molecular encoder comparison.
It was trained from scratch with standard masked language modelling on the
curated ChEMBL 36 training split. Input representation: **{representation}**;
tokenizer: **{algorithm}**. Convert SMILES to SELFIES before tokenization only
for SELFIES models. The tokenizer was trained on the same training-only
two-million-molecule sample as the other configurations, without adding
benchmark-observed symbols.

## Identity and training

- Parameters: {result["num_parameters"]:,}; vocabulary size: {result["vocab_size"]:,}.
- Context: {args["max_seq_length"]} tokens; MLM mask probability: {args["mlm_probability"]}.
- Terminal checkpoint: step {result["terminal_step"]}; selected by the prespecified terminal-step rule.
- Training molecule presentations: 7,680,000; seed: {seed}.
- Training code commit: `{identity["git"]["commit"]}`.
- Campaign manifest SHA-256: `{identity["inputs"]["campaign_manifest_sha256"]}`.
- Model weight SHA-256: `{result["final_model_sha256"]}`.
- Tokenizer SHA-256: `{result["tokenizer_sha256"]}`.

The associated paper reports frozen-embedding ROC-AUC on 25 tasks using
training-side cross-validation to select downstream classifiers. It reports
five-seed aggregates rather than treating this seed as a separate claim.
The released evaluation uses common retained test rows within the five
internal configurations; imported baselines have unverified molecule and
head-selection provenance. Benchmark structures were not excluded from
self-supervised pretraining. This model has no 3D conformer information and
was not evaluated for molecular generation or general out-of-domain use.

## Load the checkpoint

Install the ModernMolBERT code package with its pinned dependencies, then:

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM
from modernmolbert.tokenization.load import load_checkpoint_tokenizer

path = snapshot_download("{repo_id}")
tokenizer, representation = load_checkpoint_tokenizer(path)
model = AutoModelForMaskedLM.from_pretrained(path).eval()
# Pass a {representation} string to tokenizer; for SELFIES inputs, first use
# selfies.encoder(smiles_string) when starting from SMILES.
```

The repository includes `run_identity.json` and `release_manifest.json` with
the exact source and staged-file hashes. Dataset records retain their own
ChEMBL-derived licensing terms; this model repository's code/weights use MIT.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source = run_dir / "final_model"
    identity_path = run_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    run_id, seed_dir = run_dir.parts[-2:]
    seed = int(seed_dir.removeprefix("seed"))
    recipe = identity["args"]
    result = identity["result"]
    if (
        identity["schema"] != 2
        or identity["git"]["dirty"]
        or recipe["seed"] != seed
        or recipe["max_seq_length"] != 384
        or result["terminal_step"] != 30000
        or result["selected_step"] != 30000
        or result["selection_rule"] != "terminal_step"
    ):
        raise ValueError("Run is not an accepted revision terminal checkpoint")
    algorithm = recipe["tokenizer_algorithm"].upper()
    representation = recipe["representation"].upper()
    if algorithm not in {"APE", "BPE"} or representation not in {"SELFIES", "SMILES"}:
        raise ValueError("Unexpected tokenizer or representation")
    expected_name = f"{recipe['model_size']}_{algorithm.lower()}_{representation.lower()}"
    if run_id != expected_name:
        raise ValueError(f"Run directory does not match run identity: {run_id}")
    if sha256(source / "model.safetensors") != result["final_model_sha256"]:
        raise ValueError("Final model weight hash mismatch")
    tokenizer_file = source / (
        "tokenizer.json" if algorithm == "BPE" else f"{representation.lower()}_vocab.json"
    )
    if sha256(tokenizer_file) != result["tokenizer_sha256"]:
        raise ValueError("Final tokenizer hash mismatch")
    if args.output_dir.exists():
        raise FileExistsError(f"Release directory already exists: {args.output_dir}")
    shutil.copytree(source, args.output_dir, ignore=shutil.ignore_patterns("training_args.bin"))
    shutil.copy2(identity_path, args.output_dir / "run_identity.json")
    (args.output_dir / "README.md").write_text(
        make_card(identity, run_id, seed, args.repo_id), encoding="utf-8"
    )

    from modernmolbert.tokenization.load import load_checkpoint_tokenizer
    from transformers import AutoModelForMaskedLM

    tokenizer, loaded_representation = load_checkpoint_tokenizer(args.output_dir)
    if loaded_representation != representation or len(tokenizer) != result["vocab_size"]:
        raise ValueError("Staged tokenizer identity mismatch")
    sample = "CCO"
    if representation == "SELFIES":
        import selfies

        sample = selfies.encoder(sample)
    encoded = tokenizer(sample, return_tensors="pt")
    model = AutoModelForMaskedLM.from_pretrained(args.output_dir).eval()
    import torch

    with torch.no_grad():
        output = model(**encoded).logits
    if not torch.isfinite(output).all() or output.shape[-1] != result["vocab_size"]:
        raise ValueError("Staged model inference failed")

    files = {
        path.relative_to(args.output_dir).as_posix(): sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "repo_id": args.repo_id,
        "run_id": run_id,
        "seed": seed,
        "representation": representation,
        "tokenizer_algorithm": algorithm,
        "source_run_identity_sha256": sha256(identity_path),
        "source_weight_sha256": result["final_model_sha256"],
        "source_tokenizer_sha256": result["tokenizer_sha256"],
        "files_sha256": files,
    }
    (args.output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated {run_id} seed {seed}: {len(files)} staged files")


if __name__ == "__main__":
    main()
