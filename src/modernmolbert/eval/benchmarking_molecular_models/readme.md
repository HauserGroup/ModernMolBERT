# Molecular Embedding Benchmark Scoring

This directory contains the focused benchmark runtime for pretrained molecular
embedding tables. It does three things only:

1. download and prepare benchmark datasets;
2. score existing embedding files with the benchmark heads;
3. write results directly to the public CSV schema.

External model wrappers, notebook visualizations, historical paper artifacts,
and dependency-management scripts are intentionally not part of this package.

## Dataset Setup

Download/prepare datasets:

```sh
python download.py --datasets all
```

Dataset definitions live in `config/datasets.yaml`. Prepared datasets are
stored according to `config/embedding/default.yaml`, usually under
`data/prepared`.

## Expected Embeddings

Scoring expects precomputed embeddings at:

```text
data/embedded/<dataset>/<embedder>.joblib
```

Each file must contain an `EmbeddedDataset` object with:

- `X`: numeric embedding matrix;
- `y`: label dataframe;
- `splits`: train/valid/test indices;
- `task`: `classification` or `regression`;
- `embedder`: embedder name.

The benchmark keeps the original split and metric behavior: train+valid is used
for supervised head fitting, test is used for final metrics, failed embeddings
are removed before scoring, and classification metrics use positive-class
probabilities with multioutput AUROC support.

## Scoring

Run scoring for one existing embedder:

```sh
./run_scoring.sh <embedder_name>
```

Equivalent direct commands:

```sh
python score.py --embedder <embedder_name> --output-csv data/benchmark_results.csv
```

The stripped benchmark runner uses `argparse` plus YAML loading directly; it
does not require Hydra, OmegaConf, SQL, or database files.

The scoring heads are `rf`, `ridge`, and `knn`. Their grids are defined in
`supervised/models.py`; changing them changes benchmark results and
the `library_hash`.

## Skipping datasets

Pass `--skip_datasets NAME [NAME ...]` to `score.py` to exclude specific datasets by name.
Names match the `name` field in `config/datasets.yaml` (e.g. `ogbg-molmuv`, `CYP1A2_Veith`).
`download.py` and `embed_modernmolbert.py` use `--datasets` for explicit inclusion; omit them or pass `all` for all datasets.

## ChEMBL36 sweep — best run (lr=1e-4)

Full pipeline from scratch for `mask_standard__mlm_0p15__lr_1e-4`:

```sh
# 1. Clean old data
rm -rf data/prepared data/embedded data/benchmark_results.csv

# 2. Download / prepare all datasets
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets all

# 3. Embed with best model
uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/chembl36_small_mask_mlm_lr_sweep/mask_standard__mlm_0p15__lr_1e-4/final_model \
  --embedder modernmolbert_chembl36_lr1e4 \
  --batch-size 32 --device auto --max-seq-length 256 --pooling mean

# 4. Score (skip the five largest / slowest datasets)
uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --embedder modernmolbert_chembl36_lr1e4 \
  --output-csv outputs/eval/praski_chembl36_lr1e4/results.csv \
  --checkpoint-dir outputs/eval/praski_chembl36_lr1e4/checkpoints \
  --skip_datasets ogbg-molmuv ogbg-molhiv CYP2C19_Veith CYP2D6_Veith CYP1A2_Veith
```

## ModernMolBERT Praski Run

Smoke-test the trained `runs/pubchem10m_mps_base_pilot_256/final_model`
checkpoint on `clf_AMES`:

```sh
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets clf_AMES

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets clf_AMES \
  --model-dir runs/pubchem10m_mps_base_pilot_256/final_model \
  --tokenizer-path runs/pubchem10m_mps_base_pilot_256/final_model \
  --embedder modernmolbert_pubchem10m_mps_base_pilot_256 \
  --batch-size 32 \
  --device auto \
  --max-seq-length 256 \
  --pooling mean

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets clf_AMES \
  --embedder modernmolbert_pubchem10m_mps_base_pilot_256 \
  --output-csv outputs/eval/praski_pubchem10m_mps_base_pilot_256_smoke/results.csv \
  --checkpoint-dir outputs/eval/praski_pubchem10m_mps_base_pilot_256_smoke/checkpoints
```

Run the full Praski registry after the smoke run succeeds:

```sh
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets all

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/pubchem10m_mps_base_pilot_256/final_model \
  --tokenizer-path runs/pubchem10m_mps_base_pilot_256/final_model \
  --embedder modernmolbert_pubchem10m_mps_base_pilot_256 \
  --batch-size 32 \
  --device auto \
  --max-seq-length 256 \
  --pooling mean

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets all \
  --embedder modernmolbert_pubchem10m_mps_base_pilot_256 \
  --output-csv outputs/eval/praski_pubchem10m_mps_base_pilot_256_full/results.csv \
  --checkpoint-dir outputs/eval/praski_pubchem10m_mps_base_pilot_256_full/checkpoints
```

Per-dataset checkpoint CSVs are written as `<checkpoint-dir>/<dataset>.csv`.
The aggregate CSV remains the canonical result table.

## Exporting Prediction Artifacts

`score.py` writes the canonical aggregate CSV during scoring. The prediction
artifacts under `data/predictions` are primarily for plot-level diagnostics, but
`.npz` artifacts can also be converted back into a Praski-schema CSV:

```sh
uv run python -m modernmolbert.eval.benchmarking_molecular_models.prediction_export \
  --predictions-dir data/predictions \
  --output-csv data/prediction_results.csv
```

This recomputes held-out `test_metric` from `y_true` and `y_score`. Prediction
artifacts do not contain selected hyperparameters or cross-validation scores, so
`hyperparams` and `cv_metric` are blank in this recovered table. Legacy `.npy`
prediction files are ignored because they do not include ground-truth labels.

## Output Schema

CSV results use exactly this column order:

```text
id,dataset,task,embedder,model,hyperparams,library_hash,cv_metric_name,cv_metric,test_metric_name,test_metric,key
```

Column meanings:

- `id`: monotonically increasing row id assigned when appending to the CSV;
- `dataset`: benchmark dataset name;
- `task`: dataset task string, usually `classification` or `regression`;
- `embedder`: embedding file/model name;
- `model`: supervised head, one of `rf`, `ridge`, `knn`;
- `hyperparams`: sorted JSON of selected `GridSearchCV` parameters;
- `library_hash`: stable digest of the scoring grids;
- `cv_metric_name`, `cv_metric`: model-selection metric and score;
- `test_metric_name`, `test_metric`: held-out test metric and score;
- `key`: `{dataset}_{embedder}_{model}`.
