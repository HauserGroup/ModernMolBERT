# Tests and readiness checks

This document collects testing commands, readiness gates, and optional smoke tests.

## Fast checks

Run linting and the full test suite:

```bash
uv run ruff check .
uv run pytest
```

Run fast evaluation tests:

```bash
uv run pytest tests/test_eval_*.py -q
```

Run suite/config/reporting tests:

```bash
uv run pytest \
  tests/test_eval_suite.py \
  tests/test_eval_suite_configs.py \
  tests/test_eval_reporting.py \
  -q
```

## Readiness gate before long training

Do not launch long training until all of the following pass.

### 1. Validate tokenizer

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.metadata.json \
  --n 1000
```

### 2. Run debug training

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --model_size base \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.metadata.json
```

### 3. Reload the debug checkpoint

```bash
uv run python - <<'PY'
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("runs/debug_selfies/final_model")
tok = AutoTokenizer.from_pretrained(
    "runs/debug_selfies/final_model/ape_tokenizer",
    trust_remote_code=True,
)
batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print("reload ok", out.logits.shape)
PY
```

### 4. Run focused core tests

```bash
uv run pytest \
  tests/test_tokenizer_training.py \
  tests/test_collator.py \
  tests/test_training_cli.py \
  tests/test_tokenizer_validation.py \
  tests/test_smoke_training.py \
  tests/test_checkpoint_reload.py \
  tests/test_eval_modernmolbert_selfies.py \
  -q
```

## Optional Mac smoke tests

The optional pytest smoke test is skipped unless explicitly enabled.

```bash
MODERNMOLBERT_RUN_SMOKE=1 MODERNMOLBERT_RUN_MPS=1 \
  uv run pytest -m "smoke and mps" -s
```

Use `--num_workers 0` for MPS runs.

## Optional external-baseline tests

MoLFormer tests should be run in the separate MoLFormer-only environment described in `docs/baselines.md`.

```bash
PYTHONPATH="$PWD/src" MODERNMOLBERT_RUN_MOLFORMER_TESTS=1 \
  python -m pytest tests/test_eval_molformer.py -q -s
```

## What the eval tests cover

The evaluation tests should cover:

- dataset loaders and `EvalDataset` validation,
- `FeatureBatch` contract,
- featurizer registry config loading,
- feature cache hashing and reuse,
- downstream model factories,
- task-level label/feature alignment,
- single-run runner outputs,
- suite runner smoke tests,
- CLI output-directory validation,
- checked-in suite config parsing,
- reporting summaries and plots.

## When to run what

For ordinary development:

```bash
uv run pytest tests/test_eval_*.py -q
```

Before a training run:

```bash
uv run ruff check .
uv run pytest
```

Before treating benchmark numbers as meaningful:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core \
  --overwrite

uv run python -m modernmolbert.eval.cli.report_benchmark_results \
  --results_csv outputs/eval/pilot_core/results.csv \
  --output_dir outputs/eval/pilot_core/report
```
