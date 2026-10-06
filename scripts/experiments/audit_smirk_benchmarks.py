"""Preflight SMIRK coverage on the frozen 25-task prepared benchmark."""

import json
from pathlib import Path

import yaml

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.tokenization.load import load_verified_tokenizer
from modernmolbert.utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml"
TOKENIZER = ROOT / "tokenizer/experimental_smirk_v1/smirk_smiles.json"
OUTPUT = ROOT / "outputs/experimental_smirk_v1/tokenizer_benchmark_audit.json"


def main() -> None:
    tokenizer, metadata, _, _ = load_verified_tokenizer(TOKENIZER)
    registry = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    names = sorted({entry["name"] for entry in registry.values()} - {"ogbg-moltoxcast"})
    if len(names) != 25:
        raise ValueError("Expected 25 prepared benchmark tasks")
    tasks = {}
    for name in names:
        path = ROOT / "data/prepared" / f"{name}.json"
        dataset = Dataset.deserialize_legacy(path)
        failures = []
        for row, raw in enumerate(dataset.data["smiles"]):
            smiles = str(raw).strip()
            if not smiles:
                reason = "empty"
            else:
                tokens = tokenizer.tokenize(smiles)
                if tokenizer.unk_token in tokens or "".join(tokens) != smiles:
                    reason = "not_lossless"
                elif len(tokens) + 2 > 384:
                    reason = "over_context"
                else:
                    reason = ""
            if reason:
                failures.append({"source_row": row, "reason": reason})
        tasks[name] = {
            "prepared_sha256": file_sha256(path),
            "source_rows": len(dataset.data),
            "failure_count": len(failures),
            "failures": failures,
        }
        print(f"{name}: {len(failures)}/{len(dataset.data)} SMIRK failures", flush=True)
    result = {"tokenizer_sha256": metadata["tokenizer_sha256"], "tasks": tasks}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}; total failures {sum(x['failure_count'] for x in tasks.values())}")


if __name__ == "__main__":
    main()
