# Datasets

This document covers dataset preparation and dataset configuration for ModernMolBERT evaluation.

The benchmark runner expects each evaluation dataset to become an `EvalDataset` with fixed train/valid/test splits, a SMILES column, optional SELFIES column, task labels, and metadata.

## ChEMBL36 SELFIES pretraining data

Prepare a curated ChEMBL36 SELFIES pretraining dataset from canonical SMILES.
This is the canonical full-dataset preparation command:

```bash
uv sync --group pretrain-data

uv run python -m modernmolbert.data.prepare_chembl36_selfies \
  --dataset_name lukaskim/ChEMBL-36 \
  --dataset_config molecules \
  --split train \
  --smiles_column canonical_smiles \
  --output_dir data/pretrain/chembl36_selfies \
  --seed 13 \
  --valid_fraction 0.01 \
  --test_fraction 0.01 \
  --dedupe_column standard_inchi_key \
  --min_heavy_atoms 3 \
  --max_heavy_atoms 100 \
  --max_mw 1000.0
```

For a smoke test:

```bash
uv run python -m modernmolbert.data.prepare_chembl36_selfies \
  --output_dir data/pretrain/chembl36_selfies_10k \
  --max_rows 10000
```

Expected layout:

```text
data/pretrain/chembl36_selfies/
  train.parquet
  valid.parquet
  test.parquet
  metadata.json
  example.tsv
```

The preparation pipeline validates and canonicalizes source SMILES with RDKit,
converts the canonical SMILES to SELFIES, applies light small-molecule filters,
deduplicates molecules, and writes deterministic hash splits. The trainer can
consume the prepared files through `--data_files` and `--selfies_column selfies`.

`data/pretrain/chembl36_selfies_tokenized` and `modernmolbert.data.pretokenize_chembl36` are deprecated for new runs. The precomputed `input_ids` are tied to the tokenizer that created them, so train from `data/pretrain/chembl36_selfies` and let the training CLI encode SELFIES with the active tokenizer.

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
