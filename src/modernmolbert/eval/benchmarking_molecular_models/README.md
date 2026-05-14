# ModernMolBERT Molecular Benchmark

This package is the focused benchmark entrypoint for trained ModernMolBERT
checkpoints. The old upstream model-zoo implementation has been moved to
`src/modernmolbert/eval/junk/benchmarking_molecular_models_upstream/` for later
deletion.

This benchmark evaluates frozen molecular representations by training
lightweight supervised heads on top of cached embeddings. It does not fine-tune
the encoder and should not be used as a substitute for downstream fine-tuning
experiments.

## Run

Prepare MoleculeNet datasets first:

```bash
uv run python -m modernmolbert.eval.cli.prepare_moleculenet \
  --split scaffold \
  --seed 13 \
  --frac_train 0.8 \
  --frac_valid 0.1 \
  --frac_test 0.1 \
  --output_root data/eval/moleculenet_sanitized \
  --deepchem_data_dir data/deepchem/raw \
  --deepchem_save_dir data/deepchem/processed
```

Run the focused benchmark:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.run \
  --model-path runs/pubchem10m_mps_base_pilot_256/final_model \
  --output-dir outputs/molecular_eval/modernmolbert_pilot \
  --datasets bbbp bace esol \
  --pooling mean \
  --heads auto \
  --batch-size 64 \
  --embed-batch-size 32 \
  --max-length 256 \
  --device auto
```

Run the classification-only lightweight parity mode:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.run \
  --model-path runs/pubchem10m_mps_base_pilot_256/final_model \
  --output-dir outputs/molecular_eval/modernmolbert_pilot_lightweight_parity \
  --datasets bbbp bace tox21 \
  --parity lightweight \
  --heads auto
```

`--parity lightweight` is opt-in and keeps the default benchmark path unchanged.
It evaluates classification datasets with the lightweight benchmark's `rf`,
`ridge`, and `knn` grid-search heads and AUROC prediction shaping. Regression
datasets are intentionally rejected in this mode because the checked-in
lightweight scoring path calls classifier-only `predict_proba`.

Do not add `scikit-fingerprints` to the active eval dependency group for this
mode. Its released dependency bounds conflict with this project's newer
NumPy/RDKit stack. The active parity implementation uses local NumPy/sklearn
replacements for the two lightweight helpers it needs: count-vector Tanimoto
distance and multioutput AUROC.

Outputs:

```text
outputs/molecular_eval/modernmolbert_pilot/
  embeddings/
  manifest.json
  results.csv
  results.jsonl
  run_config.json
  summary.csv
```

## Output Shape

All modes write the same top-level files:

- `results.csv`: machine-readable benchmark rows.
- `results.jsonl`: JSONL copy of `results.csv`.
- `summary.csv`: one selected row per dataset/task group, ranked by `roc_auc`
  descending when present, then `rmse`/`mae` ascending.
- `manifest.json`: suite metadata, cache directory, and run records.
- `run_config.json`: CLI/API parameters used for this run.
- `embeddings/`: cached feature batches keyed by dataset split and featurizer.

Default mode (`--parity none`) writes one result row per
dataset/task/head/seed. Important columns include:

```text
dataset, task, task_type, split, featurizer, featurizer_type,
downstream_name, downstream_model, seed,
n_train, n_eval, n_train_total, n_eval_total,
n_train_feature_valid, n_eval_feature_valid,
train_feature_invalid_rate, eval_feature_invalid_rate,
train_feature_cache_key, eval_feature_cache_key,
train_feature_dim, eval_feature_dim,
accuracy, balanced_accuracy, roc_auc, average_precision,
mae, rmse, r2,
downstream_*
```

Metric columns are task-dependent: classification rows usually contain
`accuracy`, `balanced_accuracy`, `roc_auc`, and `average_precision`; regression
rows usually contain `mae`, `rmse`, and `r2`.

Lightweight parity mode (`--parity lightweight`) writes one result row per
dataset/head. Multi-task classification datasets are evaluated jointly, so
`task` is `__all__` and task names are stored in `task_names`.
Important columns include:

```text
dataset, task, task_names, n_tasks, task_type, split,
featurizer, featurizer_type,
downstream_name, downstream_model, seed,
n_train, n_eval, n_train_total, n_eval_total,
n_train_feature_valid, n_eval_feature_valid,
train_feature_invalid_rate, eval_feature_invalid_rate,
train_feature_cache_key, eval_feature_cache_key,
train_feature_dim, eval_feature_dim,
roc_auc, cv_roc_auc, downstream_best_params
```

In lightweight parity mode, `roc_auc` is the final test metric and `cv_roc_auc`
is the `GridSearchCV.best_score_` from the training split.

## Determinism

Default mode accepts `--seed` and passes it to the native downstream sklearn
heads as `random_state`. With the same checkpoint, tokenizer, data splits,
device, dependency versions, and cache state, default-mode results are expected
to be repeatable. This is not a cross-device bit-for-bit guarantee: PyTorch,
CUDA, MPS, BLAS, and sklearn parallelism can still introduce small numerical
differences.

Lightweight parity mode records the suite seed but does not force every
lightweight-compatible estimator to use it. This is intentional: the original
lightweight RF classifier did not set `random_state`, and parity mode preserves
that behavior. The `ridge` and `knn` parity heads are effectively deterministic
for fixed inputs; the `rf` parity head can vary between runs.

## Original Lightweight Comparison

The checked-in lightweight reference is
`src/modernmolbert/eval/benchmarking_molecular_models_lightweight/`. This
active package does not import it at runtime; it reimplements the metric
contract needed for ModernMolBERT benchmarking.

| Behavior | Original lightweight | Active default | `--parity lightweight` |
| --- | --- | --- | --- |
| Entry point | Hydra `score.py` over precomputed joblib embeddings | Native CLI embeds and scores in one run | Native CLI embeds and scores in one run |
| Heads | `rf`, `ridge`, `knn` | `logistic_regression`, `ridge`, optional RF variants | `rf`, `ridge`, `knn` |
| Model selection | `GridSearchCV(cv=5)` | Fixed sklearn config per head | `GridSearchCV(cv=5)` with lightweight grids |
| Multi-task classification | Joint multioutput classifier/scorer | One task column per row | Joint dataset-level scorer |
| Label NaNs | Train NaNs filled with `0`; test NaNs preserved for multioutput AUROC | Filtered per task | Matches lightweight classification behavior |
| Outputs | DB rows and prediction `.npy` files | CSV/JSONL/manifest/run config | CSV/JSONL/manifest/run config |
| Regression | Defined grids, but scoring path calls `predict_proba` | Supported | Rejected with a clear error |

`datasets.yaml` lists the prepared datasets available to the wrapper. The
default implementation delegates feature extraction, caching, task alignment,
metrics, and sklearn heads to the native `modernmolbert.eval` modules. The
lightweight parity mode reuses the native embedding cache but uses a separate
classification scorer to match the old lightweight metric contract.
