#!/usr/bin/env bash
# Evaluate five completed SMIRK seeds without changing the accepted paper outputs.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
training_root=${SMIRK_TRAINING_ROOT:-/home/jakob/projects/ModernMolBERT}
pilot=outputs/experimental_smirk_v1
status="$pilot/evaluation_queue_status.txt"
mkdir -p "$pilot/logs"

seed=setup
task=none
model=none
trap 'printf "failed seed%s %s %s %s\n" "$seed" "$task" "$model" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"' ERR

while [[ $(cat "$training_root/outputs/experimental_smirk_v1/queue_status.txt") != complete\ 5/5\ * ]]; do
  training_status=$(cat "$training_root/outputs/experimental_smirk_v1/queue_status.txt")
  if [[ $training_status == failed* ]]; then
    echo "Training queue failed: $training_status" >&2
    exit 1
  fi
  printf 'waiting_for_training %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
  sleep 60
done

for seed in 42 43 44 45 46; do
  while occupied=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) \
    && [[ -n "$occupied" ]]; do
    printf 'waiting_for_gpu seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$status"
    sleep 60
  done
  task=all
  model=smirk
  printf 'embedding seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
  "$uv_bin" run --locked --no-sync python scripts/experiments/evaluate_smirk_seed.py embed "$seed" \
    > "$pilot/logs/embed_seed${seed}.log" 2>&1
  printf 'materializing seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
  "$uv_bin" run --locked --no-sync python scripts/experiments/evaluate_smirk_seed.py \
    materialize "$seed" > "$pilot/logs/materialize_seed${seed}.log" 2>&1

  pairs_text=$("$uv_bin" run --locked --no-sync python - "$seed" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
path = Path(f"outputs/experimental_smirk_v1/evaluation_seed{seed}.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest["seed"] != seed or len(manifest["tasks"]) != 25:
    raise ValueError("Incomplete SMIRK pilot cohort")
for task in sorted(manifest["tasks"], key=lambda t: (manifest["tasks"][t]["matched_rows"], t)):
    print(f"{task} smirk")
    for model in sorted(manifest["tasks"][task]["comparators_rescored"]):
        print(f"{task} {model}")
PY
  )
  mapfile -t pairs <<< "$pairs_text"
  if [[ ${#pairs[@]} -lt 25 ]]; then
    echo "Expected at least 25 SMIRK scoring pairs for seed $seed" >&2
    exit 1
  fi
  for pair in "${pairs[@]}"; do
    read -r task model <<< "$pair"
    printf 'scoring seed%s %s %s %s\n' "$seed" "$task" "$model" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
    "$uv_bin" run --locked --no-sync python scripts/experiments/evaluate_smirk_seed.py \
      score "$seed" "$task" "$model" --n-jobs "${N_JOBS:-4}" \
      > "$pilot/logs/score_seed${seed}_${task}_${model}.log" 2>&1
  done
  printf 'scored seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
done

printf 'summarizing %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
"$uv_bin" run --locked --no-sync python scripts/experiments/summarize_smirk_pilot.py \
  > "$pilot/logs/summary.log" 2>&1
printf 'complete 5/5 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
