#!/usr/bin/env bash
set -euo pipefail

uv run python -m modernmolbert.eval.cli.prepare_moleculenet \
  --datasets esol freesolv lipophilicity bbbp bace clintox tox21 sider \
  --split scaffold \
  --output_root data/eval/moleculenet_sanitized
