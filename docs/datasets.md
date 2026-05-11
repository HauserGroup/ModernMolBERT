# Datasets

This document covers dataset preparation and dataset configuration for ModernMolBERT evaluation.

The benchmark runner expects each evaluation dataset to become an `EvalDataset` with fixed train/valid/test splits, a SMILES column, optional SELFIES column, task labels, and metadata.

## MoleculeNet preparation

MoleculeNet datasets are prepared locally from DeepChem into sanitized Parquet files containing raw SMILES, canonical SMILES, SELFIES, validity flags, and task labels.

The preparation pipeline loads each DeepChem dataset unsplit, sanitizes molecules first, converts valid molecules to canonical SMILES and SELFIES, and then applies a local train/validation/test split. This avoids DeepChem/RDKit scaffold splitting before invalid molecules have been removed.

Prepared layout:

```text
data/eval/moleculenet_sanitized/<dataset>/
  metadata.json
  train.parquet
  valid.parquet
  test.parquet
  example.tsv
```

Prepare the core suite:

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

List available datasets:

```bash
uv run python -m modernmolbert.eval.cli.prepare_moleculenet --list_datasets
```

## Current core datasets

The current core MoleculeNet suite uses:

- ESOL
- FreeSolv
- Lipophilicity
- BBBP
- BACE
- ClinTox
- Tox21
- SIDER

The pilot suite uses:

- BBBP
- BACE
- ESOL

## Dataset config examples

Benchmark suite dataset configs live in `configs/eval_suites/*.yaml`.

### Prepared MoleculeNet dataset

```yaml
- name: bbbp
  loader: prepared_moleculenet
  dataset_dir: data/eval/moleculenet_sanitized/bbbp
  eval_split: test
  merge_train_valid: true
```

### Generic train/test table splits

```yaml
- name: my_dataset
  loader: table_splits
  task_type: classification
  task_names: label
  train_path: data/my_dataset/train.csv
  valid_path: data/my_dataset/valid.csv
  test_path: data/my_dataset/test.csv
  smiles_column: smiles
  selfies_column: selfies
```

### Single table with a split column

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
  selfies_column: selfies
```

## Dataset contract

An `EvalDataset` should provide:

- `name`
- `task_type`: `classification` or `regression`
- `task_names`: one or more label columns
- `train`, `valid`, and `test` frames
- `smiles_column`
- optional `selfies_column`
- metadata

For MoleculeNet datasets, use canonical SMILES by default:

```text
smiles_column: smiles_canonical
selfies_column: selfies
```

## Notes on invalid molecules

The preparation pipeline records invalid molecules with:

```text
is_valid
sanitize_error
```

Most benchmark runs use only rows valid after sanitization. Featurizers still return a `FeatureBatch.valid_mask` to handle invalid molecules consistently across all representations.

## Recommended workflow

1. Prepare MoleculeNet data.
2. Inspect `example.tsv` for a few datasets.
3. Run `pilot_core.yaml`.
4. Only run `core_moleculenet.yaml` after the pilot suite succeeds.

Example:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core \
  --overwrite
```
