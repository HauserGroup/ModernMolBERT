# Praski Benchmark Reference Tables

This directory contains the reference baseline tables from the Praski et al. benchmarking paper:

> **Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning**
> Mateusz Praski, Jakub Adamczyk, Wojciech Czech
> arXiv:2508.06199 (2025)

## Files

- **Praski_table_1.tsv**: Summary ranking of 25 embedding models by mean rank and AUROC across the full benchmark.
- **Praski_table_4.tsv**: Dataset statistics: name, source, sample counts, task counts, and positive sample percentages for all 26 benchmark datasets.
- **Praski_table_6.tsv**: Detailed per-head performance: rank and AUROC for each model × head (rf, knn, linear) combination.

## Usage

The bundled TSVs here are **reference artifacts only** — human-readable baselines from the paper.
They are not direct input to the comparison script.

To compare a new embedder against these baselines, first run the benchmark to produce a raw
per-dataset result CSV (one row per dataset × embedder × head), then pass that CSV to the
comparison script:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.compare_praski_tables \
  --baseline data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv \
  --ours outputs/eval/results.csv \
  --our-embedder ModernMolBERT_SELFIES_ChEMBL36_2M \
  --output-dir outputs/eval/comparison
```

If your results are already in the same file as the baselines (single combined CSV), omit `--ours`:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.compare_praski_tables \
  --baseline data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv \
  --our-embedder ModernMolBERT_SELFIES_ChEMBL36_2M \
  --output-dir outputs/eval/comparison
```

To annotate any result table with model family and class metadata:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.annotate_model_table \
  --input data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv \
  --output outputs/eval/annotated_results.csv
```

**Output files** written to `--output-dir`:

| File | Contents |
|------|----------|
| `table6_like.csv` | Mean rank and mean metric per model for best head, kNN, RF, and linear/ridge |
| `table1_like.csv` | Compact summary: mean rank and mean metric after best-head selection |
| `dataset_winners.csv` | Best embedder/head per dataset (use `--write-debug-tables`) |
| `pairwise_vs_ours.csv` | Win/loss breakdown vs our embedder on shared datasets |
| `manifest.csv` | Input paths, row counts, and dataset/embedder counts |

## Citation

```bibtex
@article{praski2025benchmarking,
  title={Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning},
  author={Praski, Mateusz and Adamczyk, Jakub and Czech, Wojciech},
  journal={arXiv preprint arXiv:2508.06199},
  year={2025}
}
```

## Source

Official repository: https://github.com/MLCIL/benchmarking_molecular_models
