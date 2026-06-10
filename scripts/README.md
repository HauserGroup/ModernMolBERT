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
| `build_paper_results.py` | Derive all paper-facing numbers (incl. Wilcoxon tests) from `outputs/eval/best_metric_by_dataset_embedder.csv` | |
| `compute_bootstrap_cis.py` | Paired bootstrap 95% CIs on mean ΔROC-AUC for the four key ModernMolBERT-vs-baseline comparisons | ✓ |
| `compute_property_regression.py` | Ridge regression of mean-pooled embeddings → 9 ChEMBL physicochemical descriptors; reports test R² | ✓ |
| `make_ape_token_table.py` | Supplementary table of the most frequent APE merged tokens (Appendix B) | ✓ |
| `make_appendix_table.py` | Per-task full ROC-AUC table (Appendix C / S3) from the main-analysis matrix | |
| `make_paper_figures.py` | Paper figures (Fig 2 internal comparison, baselines scatter) from the 25-task matrix | |
| `make_loss_curves.py` | Training/validation loss curves from a run's `trainer_state.json` | |
| `align_sweep_result_csvs.py` | Reconcile old/new schema sweep CSVs under `results/` into one aligned table | |
| `arrange_panes.py` | Compose a labeled overview image from the QED PaCMAP PNG panes | |

## `sweeps/` — pre-training launchers

Bash hyperparameter sweeps that call `python -m modernmolbert.train_selfies_ape_modernbert`.

| Script | Sweep |
|--------|-------|
| `train_chembl36_small_sweep.sh` | small model, masking (3) × MLM prob (3) × LR (3) = 27 runs |
| `train_chembl36_small_sweep_standard.sh` | small model, standard masking only = 9 runs |
| `train_chembl36_base_sweep_standard.sh` | base model, standard masking only = 9 runs |

## `maintenance/` — one-off fixes (kept for provenance)

| Script | Purpose |
|--------|---------|
| `patch_model_max_length.py` | Force `model_max_length: 128` in specific released runs' `tokenizer_config.json` (hardcoded paths) |
