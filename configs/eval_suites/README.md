> **Deprecated.** The `eval.cli.run_benchmark_suite` / `report_benchmark_results` suite runners referenced below have moved to `eval/deprecated/` and are no longer the supported path. Use the download/embed/score pipeline in [../../docs/evaluation.md](../../docs/evaluation.md) instead. This file is kept for reference.

## Contributed datasets

Reusable datasets are contributed through Python code, not by adding standalone dataset YAML files.

A contributed dataset must provide:

```text
1. a loader function returning EvalDataset
2. a DatasetSpec registration
3. a test
4. source/license/split metadata
```

**See CONTRIBUTING.md for the full dataset contribution guide.**

Reusable datasets are contributed through Python loaders and `DatasetSpec`, not by standalone YAML files.

Suite configs may reference registered datasets:


```yaml
datasets:
  - loader: registered
    name: my_activity
    root: data/eval/my_activity
```

---

# 6. Tests to update
## File

```text

tests/test_eval_dataset_registry.py
```


See CONTRIBUTING.md for the full dataset contribution guide.

The current benchmark path supports only:

- regression
- binary classification

Add one test that makes the missing-label policy explicit for the example loader.

# Evaluation suite configs

This directory contains benchmark suite configs for frozen-representation evaluation.

A suite config defines the Cartesian product of:

```text
datasets × featurizers × downstream models × seeds
```

Each run is executed by:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core
```

Use `--overwrite` to replace an existing non-empty output directory:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core \
  --overwrite
```

## Available configs

### `pilot_core.yaml`

Small first-pass suite for debugging and pilot comparisons.

Datasets:

- BBBP
- BACE
- ESOL

Featurizers:

- ECFP4
- ModernMolBERT pilot checkpoint

Downstream models:

- logistic regression for classification
- ridge regression for regression

### `core_moleculenet.yaml`

Larger current MoleculeNet suite.

Datasets:

- ESOL
- FreeSolv
- Lipophilicity
- BBBP
- BACE
- ClinTox
- Tox21
- SIDER

Featurizers:

- ECFP4
- ModernMolBERT pilot checkpoint

Downstream models:

- logistic regression
- random forest classifier
- ridge regression
- random forest regressor

## Schema

### Top-level fields

```yaml
name: pilot_core

datasets: [...]
featurizers: [...]
downstream_models: {...}

seeds: [13]
eval_split: test
batch_size: 64
use_cache: true
```

### Dataset entries

Prepared MoleculeNet datasets use:

```yaml
- name: bbbp
  loader: prepared_moleculenet
  dataset_dir: data/eval/moleculenet_sanitized/bbbp
  eval_split: test
  merge_train_valid: true
```

Generic table split datasets use:

```yaml
- name: my_dataset
  loader: table_splits
  task_type: classification
  task_names: label
  train_path: data/my_dataset/train.csv
  valid_path: data/my_dataset/valid.csv
  test_path: data/my_dataset/test.csv
  smiles_column: smiles
```

Single-table datasets with a split column use:

```yaml
- name: my_dataset
  loader: table_with_split_column
  task_type: classification
  task_names: label
  table_path: data/my_dataset/all.parquet
  split_column: split
  train_value: train
  valid_value: valid
  test_value: test
  smiles_column: smiles
```

### Featurizer entries

ECFP4:

```yaml
- type: ecfp4
  name: ecfp4_2048
  n_bits: 2048
  radius: 2
```

ModernMolBERT SELFIES:

```yaml
- type: modernmolbert_selfies
  name: modernmolbert_pilot
  model_dir: runs/pubchem10m_mps_base_pilot_256/final_model
  tokenizer_path: runs/pubchem10m_mps_base_pilot_256/final_model
  max_seq_length: 256
  pooling: mean
  device: auto
  batch_size: 32
```

`type` selects the registered featurizer implementation.

`name` labels this featurizer instance in outputs and cache metadata.

### Downstream model entries

Classification models:

```yaml
downstream_models:
  classification:
    - name: logistic_balanced
      model_type: logistic_regression
      standardize: true
      params:
        class_weight: balanced
        max_iter: 5000
        C: 1.0
```

```yaml
downstream_models:
  classification:
    - name: random_forest_classifier
      model_type: random_forest_classifier
      standardize: false
      params:
        n_estimators: 500
        class_weight: balanced
        n_jobs: -1
```

Regression models:

```yaml
downstream_models:
  regression:
    - name: ridge
      model_type: ridge
      standardize: true
      params:
        alpha: 1.0
```

```yaml
downstream_models:
  regression:
    - name: random_forest_regressor
      model_type: random_forest_regressor
      standardize: false
      params:
        n_estimators: 500
        n_jobs: -1
```

## Outputs

A suite run writes:

```text
output_dir/
  results.csv
  skipped_tasks.csv        # only if tasks were skipped
  manifest.json
  cache/
```

`results.csv` contains one row per evaluated dataset/task/featurizer/downstream model/seed.

`manifest.json` records the resolved suite run metadata.

The feature cache is safe to delete and can also be shared across runs with:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core_v2 \
  --cache_dir outputs/eval/shared_feature_cache
```

## Recommended workflow

Start with:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core \
  --overwrite
```

If that succeeds, run:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/core_moleculenet.yaml \
  --output_dir outputs/eval/core_moleculenet \
  --overwrite
```

## Eval test commands

Run the fast evaluation tests with:

```bash
uv run pytest tests/test_eval_*.py -q

# or

uv run pytest tests/test_eval_suite.py tests/test_eval_suite_configs.py -q

```

## Reporting

After running a suite, generate summary tables and plots with:

```bash
uv run python -m modernmolbert.eval.cli.report_benchmark_results \
  --results_csv outputs/eval/pilot_core/results.csv \
  --output_dir outputs/eval/pilot_core/report
```

```text
outputs/eval/pilot_core/report/
  tables/
    summary.csv
    <metric>_matrix.csv
    <metric>_average_rank.csv
  plots/
    <metric>_by_dataset.png

```

Use `--no_plots` to write only summary tables.
