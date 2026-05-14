#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_scoring.sh <embedder_name> [output_csv]"
  echo "Expected embeddings: data/embedded/<dataset>/<embedder_name>.joblib"
  exit 1
fi

EMBEDDER_NAME="$1"
OUTPUT_CSV="${2:-data/classificationreport.csv}"
DATETIME=$(date '+%Y%m%d_%H%M')
LOG_DIR="logs_scoring"
LOG_FILE="${LOG_DIR}/${EMBEDDER_NAME}_${DATETIME}.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOG_DIR}"

(
  python "${SCRIPT_DIR}/score.py" --embedder "${EMBEDDER_NAME}"
  python -m modernmolbert.eval.benchmarking_molecular_models.export_results \
    --database data/meta.db \
    --output-csv "${OUTPUT_CSV}"
) > "${LOG_FILE}" 2>&1 &

echo "PID $! started"
echo "Log: ${LOG_FILE}"
echo "CSV export target: ${OUTPUT_CSV}"
