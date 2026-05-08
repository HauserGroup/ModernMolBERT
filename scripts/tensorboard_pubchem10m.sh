#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-runs/pubchem10m_mps_base_overnight}"

uv run tensorboard --logdir "${OUTPUT_DIR}"
