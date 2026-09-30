# Vendored Harness Provenance & Local Modifications

## Upstream Provenance

- **Repository**: [epfl-lep/benchmarking-molecular-models](https://github.com/epfl-lep/benchmarking-molecular-models)
- **Imported results**: the reference tables and `data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv`
  are byte-identical to the authors' public CSV at upstream commit `17d2aa1` (see `docs/baselines.md`).
- **Code revision**: the upstream commit this code was copied from was not recorded when it was
  vendored and is not known; the local changes below are therefore listed relative to the
  imported results, not to a code diff.
- **Location**: `src/modernmolbert/eval/benchmarking_molecular_models/`

## Purpose

This harness downloads benchmark datasets, generates molecular representations/embeddings, fits supervised heads (Ridge, Random Forest, k-Nearest Neighbors), and computes cross-validated evaluation metrics.

The evaluation splits, baseline results, and core scoring metrics are preserved to remain directly comparable with the published Praski benchmark tables.

## Behaviour-Changing Local Modifications & Extensions

1. **SMILES Parsing & Featurization Guard**:
   - Explicit guard rejecting unparsable SMILES or invalid token lengths upfront instead of silently ignoring or corrupting arrays.

2. **Scaffold Split Reproducibility**:
   - Reimplemented deterministic TDC scaffold splits to ensure consistent split assignments matching published datasets without upstream library version drift.

3. **Missing-Label Handling (`--missing-labels`)**:
   - Added support for `as-negative` and `observed` modes. Defaulted to `as-negative` for consistent internal model evaluation matching campaign protocol C2.

4. **Multi-Output Finite-Label Fitting**:
   - In `fit_multioutput_finite_label_model`, handles endpoints where subsets of samples have missing labels, and fails closed when candidate CV scores are non-finite.

5. **kNN Configuration on Highly Imbalanced Datasets**:
   - Disabled the kNN head on `clf_ogbg-molhiv` and `clf_ogbg-molmuv` in `get_disabled_reason`. The imported table has no kNN rows for these two datasets, so the shared candidate set for internal and external comparison excludes it (`docs/baselines.md`).

6. **Regression Scoring Support**:
   - Added multi-target and single-target regression evaluation paths, including solver compatibility fixes for Ridge regression (`clf__solver: ["auto"]`).

7. **Row Provenance & Traceability**:
   - Embedded datasets and prediction archives record `source_row_indices`, `test_source_row_indices`, and `prepared_data_sha256` to allow exact one-to-one validation between raw dataset rows and test predictions.

8. **Atomic File Writes**:
   - Embedding dumps (`joblib.dump`), prediction archives (`.npz`), prepared datasets, and results CSV files are written to temporary files and atomically renamed via `os.replace` to prevent file corruption from interrupted runs.
