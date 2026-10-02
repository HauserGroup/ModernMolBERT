#!/usr/bin/env bash
# Build verified per-seed score matrices, then five-seed manuscript artifacts.
set -euo pipefail

usage() {
  echo "Usage: $0 select-seed <42|43|44|45|46> | aggregate | paper" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
stage=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
uv_bin=${UV_BIN:-/opt/lab/bin/uv}
root=outputs/eval/revision_factorial_multiseed_v1
campaign=outputs/revision_factorial_multiseed_v1
baseline=data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv
overlap=outputs/audit/revision_factorial_v1/split_overlap_test_rows.csv
run_ids=(small_ape_selfies small_ape_smiles small_bpe_selfies small_bpe_smiles base_ape_selfies)
labels=(MMB-small-APE-SELFIES MMB-small-APE-SMILES MMB-small-BPE-SELFIES MMB-small-BPE-SMILES MMB-base-APE-SELFIES)
baseline_embedders=(ECFP ChemBERTa-77M-MLM SELFormer MoLFormer-XL-both-10pct)
baseline_labels=(ECFP4 ChemBERTa-2 SELFormer MoLFormer)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

manifest_for_seed() {
  if [[ $1 == 42 ]]; then
    printf '%s\n' outputs/revision_factorial_v1/evaluation_manifest.json
  else
    printf '%s\n' "$campaign/evaluation_seed$1.json"
  fi
}

score_root_for_seed() {
  if [[ $1 == 42 ]]; then
    printf '%s\n' outputs/eval/revision_factorial_v1/common_rows
  else
    printf '%s\n' "$root/seed$1/common_rows"
  fi
}

case $stage in
  select-seed)
    [[ $# -eq 2 && $2 =~ ^(42|43|44|45|46)$ ]] || usage
    seed=$2
    if [[ $seed == 42 ]]; then
      status=outputs/revision_factorial_v1/final_scoring_queue.status
      prefix=REVISION_COMMON_
    else
      status="$campaign/scoring_seed$seed.status"
      prefix="REVISION_COMMON_s${seed}_"
    fi
    [[ -f $status && $(cat "$status") == complete\ 125/125\ * ]] || {
      echo "Seed $seed scoring queue has not completed 125/125 pairs" >&2
      exit 1
    }
    manifest=$(manifest_for_seed "$seed")
    score_root=$(score_root_for_seed "$seed")
    [[ -f $manifest && -f $baseline && -f $overlap ]] || {
      echo "Required evaluation manifest, baseline table, or split-overlap audit is missing" >&2
      exit 1
    }
    "$uv_bin" run --locked python - "$manifest" "$score_root" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
run_ids = set(manifest["run_ids"])
tasks = set(manifest["tasks"])
if len(run_ids) != 5 or len(tasks) != 25:
    raise ValueError("Expected five frozen models and 25 tasks")
expected = {root / run_id / f"{task}.csv" for run_id in run_ids for task in tasks}
actual = set(root.glob("*/*.csv"))
if actual != expected or any(path.stat().st_size == 0 for path in expected):
    raise ValueError(f"Expected exactly 125 nonempty score files; missing={expected-actual}, extra={actual-expected}")
print("Verified 125 score files for frozen five-model cohort")
PY
    shopt -s nullglob
    score_files=("$score_root"/*/*.csv)
    embedders=()
    matrix_labels=()
    for i in "${!run_ids[@]}"; do
      embedder="${prefix}${run_ids[$i]}"
      embedders+=("$embedder")
      matrix_labels+=("${embedder}=${labels[$i]}")
    done
    for i in "${!baseline_embedders[@]}"; do
      matrix_labels+=("${baseline_embedders[$i]}=${baseline_labels[$i]}")
    done
    selected="$root/seed$seed/selected_heads"
    "$uv_bin" run --locked python scripts/paper/build_common_row_benchmark.py \
      --results "${score_files[@]}" \
      --embedders "${embedders[@]}" \
      --table-results "$baseline" --table-embedders "${baseline_embedders[@]}" \
      --matrix-labels "${matrix_labels[@]}" \
      --evaluation-manifest "$manifest" --split-overlap-rows "$overlap" \
      --paired-reference "${prefix}small_ape_selfies" \
      --output-dir "$selected"
    echo "Verified selected-head matrix: $selected/common_task_matrix.csv"
    ;;
  aggregate)
    [[ $# -eq 1 ]] || usage
    matrix_args=()
    evaluation_args=()
    for seed in 42 43 44 45 46; do
      matrix_args+=(--seed-matrix "$seed=$root/seed$seed/selected_heads/common_task_matrix.csv")
      evaluation_args+=(--evaluation-manifest "$seed=$(manifest_for_seed "$seed")")
    done
    "$uv_bin" run --locked python scripts/paper/aggregate_revision_seeds.py \
      "${matrix_args[@]}" "${evaluation_args[@]}" --output-dir "$root/five_seed_aggregate"
    ;;
  paper)
    [[ $# -eq 1 ]] || usage
    aggregate="$root/five_seed_aggregate"
    paper="$root/paper"
    seed42="$root/seed42/selected_heads"
    [[ -f $aggregate/manifest.json && -f $seed42/manifest.json ]] || {
      echo "Five-seed aggregate or seed-42 selected-head matrix is missing" >&2
      exit 1
    }
    mkdir -p "$paper"
    "$uv_bin" run --locked python scripts/paper/compute_revision_contrast_intervals.py \
      --mean-matrix "$aggregate/mean_common_task_matrix.csv" \
      --seed-contrast-summary "$aggregate/overall_contrast_summary.csv" \
      --output-dir "$paper/contrasts"
    "$uv_bin" run --locked python scripts/paper/build_paper_results.py \
      --task-matrix "$seed42/task_matrix.csv" \
      --common-task-matrix "$aggregate/mean_common_task_matrix.csv" \
      --reference MMB-small-APE-SELFIES --out-dir "$paper"
    "$uv_bin" run --locked python scripts/paper/make_appendix_table.py \
      --matrix "$paper/results_matrix_25task.csv" --out "$paper/table_pertask.tex" \
      --models "${baseline_labels[@]}" "${labels[@]}"
    "$uv_bin" run --locked python scripts/paper/make_paper_figures.py \
      --matrix "$paper/results_matrix_25task.csv" --reference MMB-small-APE-SELFIES \
      --figure-dir "$paper/figures" --source-data-dir "$paper/source_data"
    echo "Paper artifacts ready for review under $paper"
    ;;
  *) usage ;;
esac
