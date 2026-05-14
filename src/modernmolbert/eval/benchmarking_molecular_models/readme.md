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
`src/eval/supervised/models.py`; changing them changes benchmark results and
the `library_hash`.

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
