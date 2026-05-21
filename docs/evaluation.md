# Evaluation — Benchmark Scoring Pipeline

All scripts run from repo root unless noted. The pipeline has three stages: download, embed, score.

```
download.py  →  embed_modernmolbert.py  →  score.py  →  data/benchmark_results.csv
```

Scripts live in `src/modernmolbert/eval/benchmarking_molecular_models/`.

---

## 1. Download and prepare datasets

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets all
```

Downloads and prepares all datasets defined in `config/datasets.yaml`. Prepared datasets are stored under `data/prepared/<dataset>.joblib`. Pass specific config stems to prepare a subset, e.g. `--datasets clf_AMES clf_hERG`.

---

## 2. Embed

Generates embeddings from a ModernMolBERT checkpoint and stores them as joblib files.

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --model-dir runs/<run>/final_model \
  --embedder <embedder_name> \
  --datasets all \
  --batch-size 32 \
  --device auto \
  --max-seq-length 256 \
  --pooling mean
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `runs/pubchem10m_mps_base_pilot_256/final_model` | Path to the checkpoint directory |
| `--tokenizer-path` | same as `--model-dir` | Path to tokenizer (defaults to model-dir) |
| `--embedder` | `modernmolbert_pubchem10m_mps_base_pilot_256` | Name used for output files and results CSV |
| `--datasets` | `all` | Dataset config stems, globs, or `all` |
| `--batch-size` | `32` | Inference batch size |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--max-seq-length` | `256` | Truncation length |
| `--pooling` | `mean` | `mean` or `cls` |
| `--overwrite` | false | Re-embed even if output file exists |

### Output location

```
data/embedded/<dataset>/<embedder_name>.joblib
```

Each file contains an `EmbeddedDataset` with fields `X` (embedding matrix), `y` (labels), `splits` (train/valid/test indices), `task`, and `embedder`.

A warning is printed if `--model-dir` does not contain `best` in its path — use a `runs/best_<name>` symlink or naming convention to suppress it.

---

## 3. Score

Score precomputed embeddings across all benchmark datasets and heads.

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --embedder <embedder_name> \
  --output-csv data/benchmark_results.csv
```

Or via the shell wrapper (runs in background, logs to `logs_scoring/`):

```bash
cd src/modernmolbert/eval/benchmarking_molecular_models
./run_scoring.sh <embedder_name> [output_csv]
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--embedder` / `--model-name` | required | Embedder name; reads `data/embedded/<dataset>/<embedder>.joblib` |
| `--datasets` | from `config/score.yaml` | Dataset config stems, globs, or `all` |
| `--heads` | `rf ridge knn` | Supervised heads to run |
| `--skip_datasets` | — | Skip named datasets; accepts `ogbg-molhiv` or `clf_ogbg-molhiv` |
| `--output-csv` | `data/benchmark_results.csv` | Accumulated results CSV |
| `--checkpoint-dir` | — | Directory for per-dataset checkpoint CSVs |
| `--resume` / `--no-resume` | `true` | Skip dataset/embedder pairs with existing checkpoints |
| `--cache` / `--no-cache` | from `score.yaml` | Skip already-evaluated rows in the output CSV (`override = not cache`) |
| `--safe` / `--no-safe` | from `score.yaml` | Log errors and continue instead of aborting on first failure |
| `overrides` | — | Positional `key=value` pairs, e.g. `model_name=my_embedder` |

### Config files

`config/score.yaml` — default datasets and flags:

```yaml
cache: true
model_name: null
datasets:
  - clf_ogbg-molhiv
```

`config/embedding/default.yaml` — directory layout:

```yaml
embedded_directory: data/embedded
prepared_directory: data/prepared
predictions_directory: data/predictions
```

### Run plan

Before scoring starts, `score.py` prints a full plan:

```
[score] embedder=my_model  expanded_datasets=26  datasets_to_run=21  heads=['rf', 'ridge', 'knn']  ...
[score] skipped datasets:
  - ogbg-molmuv: requested skip
[score] run plan:
  [ 1/21] AMES
  [ 2/21] hERG
  ...
```

Skip logic (applied once, in order):
1. `--skip_datasets` — explicit user exclusions.
2. Checkpoint resume — if `--checkpoint-dir` is set and `<dir>/<dataset>__<embedder>.csv` exists with content, the dataset is skipped.

### Scoring heads

Three heads run per dataset by default. Each is a `sklearn` `Pipeline` with `GridSearchCV`.

| Head | Model | Grid |
|------|-------|------|
| `rf` | `RandomForestClassifier` / `RandomForestRegressor` | `min_samples_split` ∈ [2,4,6,8,10], `n_estimators=500` |
| `ridge` | `LogisticRegression` (clf) / `Ridge` (reg), `StandardScaler` | `C` / `alpha` log-spaced over 10 values |
| `knn` | `KNeighborsClassifier` / `KNeighborsRegressor`, `StandardScaler` | `n_neighbors` ∈ [1,3,5,7,9]; cosine distance for float embeddings, Tanimoto for integer |

KNN is skipped on MUV datasets. Multi-output classification uses `MultiOutputClassifier(LogisticRegression())`.

### Checkpoint resume

Pass `--checkpoint-dir <dir>` to write a per-dataset CSV after each dataset finishes:

```
<dir>/<dataset>__<embedder>.csv
```

On re-run with `--resume`, any dataset with a non-empty checkpoint is skipped entirely. If all heads fail for a dataset, no checkpoint is written. Use `--no-resume` to force re-scoring everything.

### Output schema

Results append to the output CSV with this column order:

```
id, dataset, task, embedder, model, hyperparams, library_hash,
cv_metric_name, cv_metric, test_metric_name, test_metric, key
```

| Column | Description |
|--------|-------------|
| `id` | Monotonically increasing row id |
| `dataset` | Benchmark dataset name |
| `task` | `classification` or `regression` |
| `embedder` | Embedding file name |
| `model` | Head: `rf`, `ridge`, or `knn` |
| `hyperparams` | Sorted JSON of selected `GridSearchCV` params |
| `library_hash` | Stable digest of scoring grids; changes if grids change |
| `cv_metric` | Model-selection score (train+valid) |
| `test_metric` | Held-out test score |
| `key` | `{dataset}_{embedder}_{model}` |

---

## Datasets

Defined in `config/datasets.yaml`. Sources are TDC ADMET/HTS benchmarks and OGB graph datasets.

### TDC classification (roc_auc)

AMES, Bioavailability_Ma, CYP1A2_Veith, CYP2C19_Veith, CYP2C9_Substrate_CarbonMangels, CYP2C9_Veith, CYP2D6_Substrate_CarbonMangels, CYP2D6_Veith, CYP3A4_Substrate_CarbonMangels, CYP3A4_Veith, DILI, HIA_Hou, PAMPA_NCATS, Pgp_Broccatelli, SARSCoV2_3CLPro_Diamond, SARSCoV2_Vitro_Touret, hERG, hERG_Karim

### OGB classification (roc_auc)

ogbg-molbace, ogbg-molbbbp, ogbg-molclintox, ogbg-molhiv, ogbg-molmuv, ogbg-molsider, ogbg-moltox21, ogbg-moltoxcast

Large datasets (`ogbg-molmuv`, `ogbg-moltoxcast`, `ogbg-moltox21`) carry a `memory_weight` field that the runner uses to scale memory limits.

### Adding a custom dataset

Add an entry to `config/datasets.yaml` under `datasets:`:

```yaml
my_assay:
  name: my_assay
  task: classification       # or regression
  metric: roc_auc
  source:
    name: local
    root: data/contributed/my_assay   # relative to CWD
    smiles_column: smiles             # optional, default: smiles
    label_columns: [active]           # optional, default: all non-smiles columns
```

Place `train.csv` and `test.csv` (and optionally `valid.csv`) in the `root` directory. Each CSV needs a SMILES column and label column(s). Invalid SMILES and non-numeric labels are dropped automatically.

---

## Full pipeline examples

### Smoke test (single dataset)

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets clf_AMES

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets clf_AMES \
  --model-dir runs/my_run/final_model \
  --embedder my_model \
  --batch-size 32 --device auto

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets clf_AMES \
  --embedder my_model \
  --output-csv outputs/eval/smoke/results.csv \
  --checkpoint-dir outputs/eval/smoke/checkpoints
```

### Full benchmark run

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets all

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/my_run/final_model \
  --embedder my_model \
  --batch-size 32 --device auto --pooling mean

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --embedder my_model \
  --output-csv outputs/eval/my_run/results.csv \
  --checkpoint-dir outputs/eval/my_run/checkpoints
```

### Skip slow datasets

The five largest datasets add significant runtime. Skip them for faster iteration:

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --embedder my_model \
  --output-csv outputs/eval/my_run/results.csv \
  --skip_datasets ogbg-molmuv ogbg-molhiv CYP2C19_Veith CYP2D6_Veith CYP1A2_Veith
```

---

## Recovering results from prediction artifacts

`score.py` is the canonical result source. If only prediction `.npz` files exist under `data/predictions/`, recover a results CSV with:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.prediction_export \
  --predictions-dir data/predictions \
  --output-csv data/prediction_results.csv
```

`hyperparams` and `cv_metric` are blank in recovered tables because prediction artifacts do not store them.
