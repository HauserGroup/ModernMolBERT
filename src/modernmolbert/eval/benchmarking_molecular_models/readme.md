# Benchmark harness

The focused benchmark runtime for frozen embeddings. It downloads and prepares the benchmark
datasets, scores existing embedding files with the `rf`, `ridge` and `knn` heads, and writes
results to the public CSV schema. External model wrappers, notebooks, historical paper
artefacts and dependency-management scripts are not part of this package. Local changes
relative to the upstream code are listed in [VENDORED.md](VENDORED.md).

The commands, flags, config files, output schema and full pipeline examples are in
[docs/evaluation.md](../../../../docs/evaluation.md); how the results become paper artefacts is in
[docs/revision_run.md](../../../../docs/revision_run.md).

## Contract

Scoring expects precomputed embeddings at `data/embedded/<dataset>/<embedder>.joblib`, each an
`EmbeddedDataset` with `X` (numeric matrix), `y` (label dataframe), `splits` (train/valid/test
indices), `task` and `embedder`.

The original split and metric behaviour is kept so results stay comparable with the Praski et al.
tables: train+valid fits the head, test gives the final metric, failed embeddings are removed
before scoring, and classification metrics use positive-class probabilities with multi-output
AUROC. The head grids are in `supervised/models.py`; changing them changes benchmark results and
`library_hash`.

Multi-endpoint datasets with missing labels (Tox21, MUV) count missing labels as negatives during
fitting and CV by default (`--missing-labels as-negative`), as the Praski et al. benchmark did.
Do not mix this with `--missing-labels observed` in one results file; the mode is recorded in each
row's `missing_labels` column. kNN is not scored on HIV and MUV, which have no kNN head in the
Praski et al. table.
