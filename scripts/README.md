# scripts/

Standalone command-line scripts. Each is independent (no cross-imports) and is
**run from the repo root**, e.g. `uv run python scripts/paper/make_paper_figures.py`.

Reusable code that the package itself imports does **not** live here — for
example model-card generation moved to `src/modernmolbert/model_cards.py`
(`python -m modernmolbert.model_cards`), since the upload scripts import it.

## `paper/` — manuscript artifact generation

Read already-computed results/embeddings and emit tables and figures for the
paper. No model training. Tested scripts have unit tests under `tests/`.

| Script | Purpose | Tested |
|--------|---------|--------|
| `build_benchmark_results_frames.py` | Build unified benchmark result frames from the Praski benchmark CSV and our own `outputs/eval` results | |
| `build_paper_results.py` | Derive all paper-facing numbers (incl. Wilcoxon tests) from the `task_matrix.csv` written by `build_common_row_benchmark.py` (`--task-matrix`), or from the legacy `outputs/eval/best_metric_by_dataset_embedder.csv` | |
| `compute_bootstrap_cis.py` | Paired bootstrap 95% CIs on mean ΔROC-AUC for the four key ModernMolBERT-vs-baseline comparisons, resampling tasks and, as a sensitivity, the task families in `config/task_families.yaml` | ✓ |
| `audit_task_overlap.py` | Test molecules shared by every pair of benchmark datasets, flagged by task family; checks the families file against the dataset config | ✓ |
| `compute_property_regression.py` | Ridge regression of mean-pooled embeddings → 9 ChEMBL physicochemical descriptors; reports test R² | ✓ |
| `make_ape_token_table.py` | Supplementary table of the most frequent APE merged tokens (Appendix B) | ✓ |
| `make_appendix_table.py` | Per-task full ROC-AUC table (Appendix C / S3) from the main-analysis matrix | |
| `make_paper_figures.py` | Paper figures (baselines scatter, group summaries; Fig 2 internal comparison only when small, base and span are all present) from the 25-task matrix | |
| `make_loss_curves.py` | Training/validation loss curves from a run's `trainer_state.json` | |
| `arrange_panes.py` | Compose a labeled overview image from the QED PaCMAP PNG panes | |

## `sweeps/` — pre-training launchers

`run_sweep.py` drives a masking × MLM-prob × learning-rate grid, launching each
run sequentially via `accelerate launch -m modernmolbert.train_selfies_ape_modernbert`.
Batch geometry, warmup, and the default LR grid follow `--model-size`; every
axis is overridable, and populated run directories are skipped on re-run.

```bash
# full small-model sweep (3 maskings × 3 MLM × 3 LR = 27 runs)
python scripts/sweeps/run_sweep.py --model-size small

# standard-masking only (9 runs); base preset
python scripts/sweeps/run_sweep.py --model-size base --masking standard

# preview without launching
python scripts/sweeps/run_sweep.py --model-size small --dry-run
```

This replaces the former `train_chembl36_*.sh` trio, which were ~95% duplicated
and differed only in model size and which strategies/LRs to sweep.
