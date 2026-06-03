# Datasets

This document covers the ChEMBL 36 SELFIES **pretraining** dataset. For **benchmark/evaluation** datasets (TDC + OGB) see [evaluation.md](evaluation.md).

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

## Benchmark datasets

Evaluation datasets (TDC ADMET/HTS and OGB graph datasets) are downloaded and
prepared by the benchmark pipeline. See [evaluation.md](evaluation.md) for the
dataset list, `config/datasets.yaml` schema, and how to add a custom dataset.
