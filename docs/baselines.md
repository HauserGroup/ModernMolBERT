# Baselines

The revised paper does not re-embed the baselines. Their scores come from the
table published with Praski et al.,
`data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv`, which records
training-side CV and test ROC-AUC for every downstream head. The file is
byte-identical to the authors' public CSV at commit `17d2aa1` of their
`benchmarking_molecular_models` repository. The harness in
`src/modernmolbert/eval/benchmarking_molecular_models/` is a stripped copy of
that code, with the same splits, heads, CV folds and hyperparameter grids.

| Paper label | Table embedder | Representation used by the original wrapper |
|---|---|---|
| ECFP4 | `ECFP` | scikit-fingerprints ECFP defaults |
| ChemBERTa-2 | `ChemBERTa-77M-MLM` | CLS token |
| MoLFormer | `MoLFormer-XL-both-10pct` | `pooler_output` |
| SELFormer | `SELFormer` | mean over tokens |

## How they enter the comparison

`scripts/paper/build_common_row_benchmark.py --table-results ... --table-embedders ...`
reads the table heads and picks each dataset × baseline head by CV ROC-AUC from
the same shared candidate set as ModernMolBERT. kNN has no table rows on HIV
and MUV, so it drops out of the candidate set there for every model. The full
command is in [revision_run.md](revision_run.md) §5.

The table's scorer treats missing labels in multi-endpoint datasets (Tox21,
MUV) as negatives. Score ModernMolBERT with `--missing-labels as-negative` so
both sides follow the same rule ([revision_run.md](revision_run.md) §4).

The table has no test-row identities, so baseline test rows cannot be matched
to ours, and per-dataset paired intervals cannot include the baselines.
