#!/usr/bin/env bash
# Embed the 25 frozen tasks with one accepted seed of all five configurations.
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(43|44|45|46)$ ]]; then
  echo "Usage: $0 <43|44|45|46>" >&2
  exit 2
fi
seed=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
queue_root=outputs/revision_factorial_multiseed_v1
if [[ $(cat "$queue_root/queue_status.txt") != complete\ 20/20\ * ]]; then
  echo "Multiseed training queue has not completed" >&2
  exit 1
fi
campaign="$queue_root/campaign_manifest.json"
source_prefix="REVISION_s${seed}_"
common_prefix="REVISION_COMMON_s${seed}_"
output="$queue_root/evaluation_seed${seed}.json"
mkdir -p "$queue_root/logs"

"$uv_bin" run --locked python - "$seed" "$campaign" <<'PY'
import json
import sys
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.common.config import (
    expand_dataset_selection,
    load_dataset_config,
)
from modernmolbert.utils import file_sha256

seed, campaign_path = int(sys.argv[1]), Path(sys.argv[2])
root = Path.cwd()
campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
campaign_sha = file_sha256(campaign_path)
if seed not in campaign.get("seeds", []) or len(campaign["run_ids"]) != 5:
    raise ValueError("Seed/model set differs from the frozen multiseed campaign")
config_dir = root / "src/modernmolbert/eval/benchmarking_molecular_models/config"
registry = {
    load_dataset_config(config_dir, name).name
    for name in expand_dataset_selection(config_dir, ["all"])
}
if registry != set(campaign["prepared_data_sha256"]):
    raise ValueError("Benchmark registry differs from the frozen campaign")
for task, expected in campaign["prepared_data_sha256"].items():
    if file_sha256(root / "data/prepared" / f"{task}.json") != expected:
        raise ValueError(f"Prepared task changed since campaign staging: {task}")
for run_id in campaign["run_ids"]:
    run = root / "runs/revision_factorial_v1" / run_id / f"seed{seed}"
    identity = json.loads((run / "run_identity.json").read_text(encoding="utf-8"))
    result = identity["result"]
    if (
        identity["args"]["seed"] != seed
        or identity["inputs"]["campaign_manifest_sha256"] != campaign_sha
        or identity["git"]["commit"] != campaign["code_commit"]
        or result["terminal_step"] != 30_000
        or result["selected_step"] != 30_000
        or file_sha256(run / "final_model" / result["final_model_file"])
        != result["final_model_sha256"]
    ):
        raise ValueError(f"Final model identity mismatch: {run_id} seed {seed}")
print(f"Verified seed {seed}: five models and 25 frozen tasks")
PY

for run_id in \
  small_ape_selfies small_ape_smiles small_bpe_selfies small_bpe_smiles base_ape_selfies; do
  occupied=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
  if [[ -n "$occupied" ]]; then
    echo "Shared GPU occupied before seed $seed embedding $run_id: $occupied" >&2
    exit 1
  fi
  echo "Embedding seed $seed / $run_id at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$uv_bin" run --locked python \
    src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
    --datasets all \
    --model-dir "runs/revision_factorial_v1/$run_id/seed$seed/final_model" \
    --tokenizer-path "runs/revision_factorial_v1/$run_id/seed$seed/final_model" \
    --embedder "${source_prefix}${run_id}" \
    --batch-size 32 --device cuda --pooling mean \
    > "$queue_root/logs/embed_seed${seed}_${run_id}.log" 2>&1
done

"$uv_bin" run --locked python scripts/materialize_revision_common_embeddings.py \
  --seed "$seed" --campaign-manifest "$campaign" \
  --source-prefix "$source_prefix" --common-prefix "$common_prefix" --output "$output" \
  > "$queue_root/logs/materialize_seed${seed}.log" 2>&1

"$uv_bin" run --locked python - "$output" <<'PY'
import json
import sys
from pathlib import Path

seed_manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
reference = json.loads(
    Path("outputs/revision_factorial_v1/evaluation_manifest.json").read_text(encoding="utf-8")
)
for task, record in reference["tasks"].items():
    other = seed_manifest["tasks"][task]
    for key in (
        "common_supervised_rows", "common_source_row_indices_sha256", "labels_sha256",
        "split_source_row_indices_sha256",
    ):
        if record[key] != other[key]:
            raise ValueError(f"Cross-seed common cohort differs for {task}: {key}")
print("Cross-seed supervised rows, labels, and splits match seed 42")
PY
