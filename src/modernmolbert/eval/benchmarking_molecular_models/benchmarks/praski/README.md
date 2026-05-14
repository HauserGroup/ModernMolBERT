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

These tables serve as:
1. **Reference baselines** for comparing new embedder implementations
2. **Dataset documentation** (table 4 describes the benchmark scope)
3. **Head-specific performance tracking** (table 6 shows head-specific strengths/weaknesses)

To compare a new embedder against these baselines, use:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.praski_compare \
  --baseline src/modernmolbert/eval/benchmarking_molecular_models/benchmarks/praski/Praski_table_1.tsv \
  --ours <your_results.csv> \
  --our-embedder <embedder_name> \
  --output-dir <comparison_output>
```

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
