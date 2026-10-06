#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
if [[ $(cat outputs/revision_factorial_v1/queue_status.txt) != complete\ * ]]; then
  echo "Five-model training queue has not completed" >&2
  exit 1
fi

"$uv_bin" run --locked python - <<'PY'
import json
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.common.config import (
    expand_dataset_selection,
    load_dataset_config,
)
from modernmolbert.utils import file_sha256

root = Path.cwd()
campaign_path = root / "outputs/revision_factorial_v1/campaign_manifest.json"
campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
campaign_sha = file_sha256(campaign_path)
if len(campaign["prepared_data_sha256"]) != 25 or len(campaign["run_ids"]) != 5:
    raise ValueError("Campaign manifest lacks the 25 tasks or five models")
config_dir = root / "src/modernmolbert/eval/benchmarking_molecular_models/config"
registry_names = {
    load_dataset_config(config_dir, config_name).name
    for config_name in expand_dataset_selection(config_dir, ["all"])
}
if registry_names != set(campaign["prepared_data_sha256"]):
    raise ValueError("Dataset registry differs from the frozen 25-task campaign")
for name, expected in campaign["prepared_data_sha256"].items():
    if file_sha256(root / "data/prepared" / f"{name}.json") != expected:
        raise ValueError(f"Prepared task changed since campaign staging: {name}")
for run_id in campaign["run_ids"]:
    run = root / "runs/revision_factorial_v1" / run_id / "seed42"
    identity = json.loads((run / "run_identity.json").read_text(encoding="utf-8"))
    result = identity["result"]
    if (
        identity["inputs"]["campaign_manifest_sha256"] != campaign_sha
        or identity["git"]["commit"] != campaign["code_commit"]
        or result["terminal_step"] != 30_000
        or result["selected_step"] != 30_000
        or file_sha256(run / "final_model" / result["final_model_file"])
        != result["final_model_sha256"]
    ):
        raise ValueError(f"Final model identity mismatch: {run_id}")
print("All five final models and 25 frozen benchmark inputs verified")
PY

mkdir -p outputs/revision_factorial_v1/logs
for run_id in \
  small_ape_selfies small_ape_smiles small_bpe_selfies small_bpe_smiles base_ape_selfies; do
  occupied=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
  if [[ -n "$occupied" ]]; then
    echo "Shared GPU is occupied before embedding $run_id: $occupied" >&2
    exit 1
  fi
  echo "Embedding $run_id at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$uv_bin" run --locked python \
    src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
    --datasets all \
    --model-dir "runs/revision_factorial_v1/$run_id/seed42/final_model" \
    --tokenizer-path "runs/revision_factorial_v1/$run_id/seed42/final_model" \
    --embedder "REVISION_$run_id" \
    --batch-size 32 --device cuda --pooling mean \
    2>&1 | tee "outputs/revision_factorial_v1/logs/embed_$run_id.log"
done

"$uv_bin" run --locked python scripts/materialize_revision_common_embeddings.py \
  2>&1 | tee outputs/revision_factorial_v1/logs/materialize_common.log
