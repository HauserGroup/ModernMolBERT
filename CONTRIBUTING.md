# Contributing

ModernMolBERT evaluates molecular representations through a shared frozen-feature benchmark pipeline (download → embed → score). Contributions should preserve comparability across datasets, featurizers, and metrics.

This document focuses on **evaluation dataset contributions**.

---

## Scope

The benchmark stack supports:

```text
regression
binary classification
```

Contributions should be reproducible, testable, documented, and narrow enough to review. Do not add dataset-specific runners, downstream models, or metric code.

---

## How datasets are added

Benchmark datasets are declared in `config/datasets.yaml`. A contribution adds one entry and points it at a local prepared-data directory. See [docs/evaluation.md](docs/evaluation.md) for the full pipeline.

Add an entry under `datasets:`:

```yaml
my_assay:
  name: my_assay
  task: classification        # or regression
  metric: roc_auc
  source:
    name: local
    root: data/contributed/my_assay   # relative to CWD
    smiles_column: smiles             # optional, default: smiles
    label_columns: [active]           # optional, default: all non-smiles columns
```

Place `train.csv` and `test.csv` (and optionally `valid.csv`) in `root`. Each CSV needs one SMILES column and one or more label columns. Invalid SMILES and non-numeric labels are dropped automatically by the pipeline.

The data itself should be hosted somewhere stable or generated from a documented source — do not commit large raw data files to the repository. The `root` path refers to a local cache/prepared-data directory.

---

## Required data shape

Fixed splits, each with one molecule column and the label column(s):

```csv
smiles,active
CCO,0
CCN,1
CCC,
```

Binary labels must be `0/1`. Regression labels must be numeric. Missing labels are allowed in the source CSV (they are dropped before scoring).

---

## Dataset properties

- No larger than a couple of thousand molecules.
- Binary classification datasets should not be extremely imbalanced — as a rough guideline, the minority class should be at least 10% of labeled rows. More imbalanced datasets may still be useful but should be discussed first.
- Suitable for benchmarking — not impossible for a featurizer (no harder than a Random Forest can solve).
- For regression, approximately normally distributed (e.g. pEC50, not raw EC50).
- Clearly cited and appropriately licensed.
- Do not contribute datasets from MoleculeNet.

---

## Checklist

- [ ] Entry added to `config/datasets.yaml` with `name`, `task`, `metric`, and `source`.
- [ ] Task is regression or binary classification.
- [ ] Binary labels are `0/1`; regression labels are numeric.
- [ ] Prepared data hosted/generated from a documented source — no large raw files committed.
- [ ] Source URL, license, and citation recorded.
- [ ] Pipeline runs end-to-end on the new dataset (`download` → `embed` → `score`).

---

## What not to do

- Do not add a new benchmark runner for one dataset.
- Do not add dataset-specific featurization code.
- Do not compute metrics or train downstream models in dataset preparation.
- Do not commit large raw datasets unless explicitly approved.
- Do not contribute multiclass datasets unless the benchmark stack has first been extended to support multiclass classification.
