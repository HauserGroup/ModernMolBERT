#!/usr/bin/env bash
# Train five SMIRK seeds sequentially on the shared Helios GPU.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
queue_root=outputs/experimental_smirk_v1
manifest="$queue_root/campaign_manifest.json"
backup_root=/data/modernmolbert_experimental_smirk_v1
mkdir -p "$queue_root/logs" "$backup_root"

seed=setup
trap 'printf "failed seed%s %s\n" "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$queue_root/queue_status.txt"' ERR

verify_run() {
  "$uv_bin" run --locked --no-sync python - "$1" "$2" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

from modernmolbert.utils import file_sha256

root, seed, manifest_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
record = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
result = record["result"]
assert record["schema"] == 2
assert record["args"]["seed"] == seed
assert record["args"]["tokenizer_algorithm"] == "SMIRK"
assert record["args"]["representation"] == "SMILES"
assert record["git"]["commit"] == campaign["code_commit"]
assert record["inputs"]["campaign_manifest_sha256"] == file_sha256(manifest_path)
assert result["terminal_step"] == result["selected_step"] == 30_000
weights = root / "final_model" / result["final_model_file"]
assert file_sha256(weights) == result["final_model_sha256"]
metrics = json.loads((root / "train_results.json").read_text(encoding="utf-8"))
assert metrics["train_samples_streaming"] == 7_680_000
print(f"Verified SMIRK seed {seed}: {result['final_model_sha256']}")
PY
}

for seed in 42 43 44 45 46; do
  source="runs/experimental_smirk_v1/small_smirk_smiles/seed$seed"
  destination="$backup_root/small_smirk_smiles/seed$seed"
  if verify_run "$source" "$seed" > /dev/null 2>&1; then
    echo "Already trained: SMIRK seed $seed"
  else
    resume=()
    if [[ -d "$source" && -n $(ls -A "$source") ]]; then
      latest=
      while IFS= read -r candidate; do
        if [[ -f "$candidate/trainer_state.json" && -f "$candidate/optimizer.pt" \
          && -f "$candidate/scheduler.pt" && -f "$candidate/rng_state.pth" ]]; then
          latest=$candidate
          break
        fi
      done < <(find "$source" -maxdepth 1 -type d -name 'checkpoint-*' | sort -Vr)
      if [[ -z "$latest" ]]; then
        echo "Nonempty partial SMIRK run without a complete checkpoint: $source" >&2
        false
      fi
      resume=(--resume-from-checkpoint "$latest")
    fi
    while occupied=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) \
      && [[ -n "$occupied" ]]; do
      printf 'waiting_for_gpu seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$queue_root/queue_status.txt"
      sleep 60
    done
    printf 'running seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$queue_root/queue_status.txt"
    "$uv_bin" run --locked --no-sync python scripts/experiments/run_smirk_train.py "$seed" \
      "${resume[@]}" > "$queue_root/logs/train_seed${seed}.log" 2>&1
    verify_run "$source" "$seed"
  fi

  mkdir -p "$destination"
  rsync -a "$source/" "$destination/"
  verify_run "$destination" "$seed" > /dev/null
  cmp "$source/run_identity.json" "$destination/run_identity.json"
  printf 'backed_up seed%s %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$queue_root/queue_status.txt"
done

printf 'complete 5/5 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$queue_root/queue_status.txt"
