#!/usr/bin/env bash
# Score the frozen five-model common cohorts on CPU, with per-pair progress logs.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
n_jobs=${N_JOBS:-4}
manifest=outputs/revision_factorial_v1/evaluation_manifest.json
status=outputs/revision_factorial_v1/final_scoring_queue.status
log_dir=outputs/revision_factorial_v1/logs/final_scoring
mkdir -p "$log_dir"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mapfile -t tasks < <("$uv_bin" run --locked python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema") != 2 or manifest.get("common_prefix") != "REVISION_COMMON_":
    raise ValueError("Expected the final common-row evaluation manifest")
run_ids = {
    "small_ape_selfies", "small_ape_smiles", "small_bpe_selfies",
    "small_bpe_smiles", "base_ape_selfies",
}
if set(manifest.get("run_ids", [])) != run_ids or len(manifest.get("tasks", {})) != 25:
    raise ValueError("Expected five models and 25 frozen tasks")
for task, record in manifest["tasks"].items():
    if set(record["models"]) != run_ids:
        raise ValueError(f"Incomplete common cohort: {task}")
for task in sorted(
    manifest["tasks"],
    key=lambda name: (manifest["tasks"][name]["common_supervised_rows"], name),
):
    print(task)
PY
)
if [[ ${#tasks[@]} -ne 25 ]]; then
  echo "Expected 25 verified common-cohort tasks" >&2
  exit 1
fi

completed=0
for task in "${tasks[@]}"; do
  for run_id in \
    small_ape_selfies small_ape_smiles small_bpe_selfies small_bpe_smiles base_ape_selfies; do
    printf 'active %s %s %s/125 %s\n' \
      "$task" "$run_id" "$((completed + 1))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
    echo "Scoring $task / $run_id ($((completed + 1))/125)"
    if "$uv_bin" run --locked python scripts/run_revision_common_scoring.py \
      "$run_id" "$task" --n-jobs "$n_jobs" \
      > "$log_dir/${task}_${run_id}.log" 2>&1; then
      completed=$((completed + 1))
    else
      printf 'failed %s %s %s/125 %s\n' \
        "$task" "$run_id" "$((completed + 1))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
      echo "Scoring failed: $task / $run_id; see $log_dir/${task}_${run_id}.log" >&2
      exit 1
    fi
  done
done
printf 'complete 125/125 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
