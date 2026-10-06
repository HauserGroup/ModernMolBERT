#!/usr/bin/env bash
# Train and back up four additional seeds for each of the five frozen configurations.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
spec=configs/revision_factorial_multiseed_v1.json
manifest=outputs/revision_factorial_multiseed_v1/campaign_manifest.json
queue_root=outputs/revision_factorial_multiseed_v1
backup_root=/data/modernmolbert_revision_factorial_v1
mkdir -p "$queue_root/logs" "$backup_root"

seed=setup
run_id=setup
trap 'printf "failed seed%s %s %s\n" "$seed" "$run_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$queue_root/queue_status.txt"' ERR

verify_run() {
  "$uv_bin" run --locked python - "$1" "$2" "$3" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

from modernmolbert.utils import file_sha256

root, run_id, seed, manifest_path = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
record = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
result = record["result"]
campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
assert record["schema"] == 2
assert record["args"]["seed"] == seed
assert record["git"]["commit"] == campaign["code_commit"]
assert record["inputs"]["campaign_manifest_sha256"] == file_sha256(manifest_path)
assert result["terminal_step"] == result["selected_step"] == 30_000
weights = root / "final_model" / result["final_model_file"]
assert file_sha256(weights) == result["final_model_sha256"]
metrics = json.loads((root / "train_results.json").read_text(encoding="utf-8"))
assert metrics["train_samples_streaming"] == 7_680_000
print(f"Verified {run_id} seed {seed}: {result['final_model_sha256']}")
PY
}

for seed in 43 44 45 46; do
  for run_id in \
    small_ape_selfies small_ape_smiles small_bpe_selfies small_bpe_smiles base_ape_selfies; do
    source="runs/revision_factorial_v1/$run_id/seed$seed"
    destination="$backup_root/$run_id/seed$seed"
    if verify_run "$source" "$run_id" "$seed" > /dev/null 2>&1; then
      echo "Already trained: $run_id seed $seed"
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
          echo "Nonempty partial run without a checkpoint: $source" >&2
          false
        fi
        resume=(--resume_from_checkpoint "$latest")
      fi
      while occupied=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) \
        && [[ -n "$occupied" ]]; do
        printf 'waiting_for_gpu seed%s %s %s\n' \
          "$seed" "$run_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$queue_root/queue_status.txt"
        sleep 60
      done
      printf 'running seed%s %s %s\n' \
        "$seed" "$run_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$queue_root/queue_status.txt"
      "$uv_bin" run --locked python scripts/run_revision_factorial_v1.py "$run_id" \
        --spec "$spec" --manifest "$manifest" --seed "$seed" "${resume[@]}" \
        > "$queue_root/logs/${run_id}_seed${seed}.log" 2>&1
      verify_run "$source" "$run_id" "$seed"
    fi

    mkdir -p "$destination"
    rsync -a "$source/" "$destination/"
    verify_run "$destination" "$run_id" "$seed" > /dev/null
    cmp "$source/run_identity.json" "$destination/run_identity.json"
    printf 'backed_up seed%s %s %s\n' \
      "$seed" "$run_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$queue_root/queue_status.txt"
  done
done

printf 'complete 20/20 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$queue_root/queue_status.txt"
