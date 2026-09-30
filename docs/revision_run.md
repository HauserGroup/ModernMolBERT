# From embeddings to paper artefacts

How scores become the manuscript's tables and figures: embed, score, select heads by
cross-validation, compare on common test rows, then generate the paper artefacts. The steps
apply to any newly trained encoder. For the five-model campaign, the inputs, the shared
supervised cohort and the launch gates are in
[revision_factorial_v1_handoff.md](revision_factorial_v1_handoff.md) and
[revision_factorial_v1_preflight.md](revision_factorial_v1_preflight.md); the plan is
`MASTER_REVISION_PLAN.md` in the manuscript repository (G7, C2, C3).

The one-model run this file was first written for (`revision_clean_small_v1`, stopped at step
16,051 of 30,000, no final model) is summarised in [revision_run_record.md](revision_run_record.md).
Its training commands are not kept: they are superseded by `scripts/run_revision_factorial_v1.py`.

Run everything from the repository root. Use a unique embedder name and a fresh output
directory per model, and never overwrite historical embeddings or result CSVs.

## 1. Embed and score

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py --datasets all

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/<run>/final_model \
  --tokenizer-path runs/<run>/final_model \
  --embedder <embedder> \
  --batch-size 32 --device auto --pooling mean

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets all --skip_datasets ogbg-molhiv ogbg-molmuv \
  --heads rf ridge knn --missing-labels as-negative \
  --embedder <embedder> \
  --output-csv outputs/eval/<run>/results.csv \
  --checkpoint-dir outputs/eval/<run>/checkpoints

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets ogbg-molhiv ogbg-molmuv \
  --heads rf ridge --missing-labels as-negative \
  --embedder <embedder> \
  --output-csv outputs/eval/<run>/results.csv \
  --checkpoint-dir outputs/eval/<run>/checkpoints
```

- **Prepared data.** `download.py` reuses cached prepared datasets; freeze and verify those
  splits rather than replacing them. The 25 entries in
  `src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml` define the
  benchmark; score all eligible heads without subsampling.
- **Embedding.** The context defaults to the model's trained context and larger values are
  rejected. Molecules with lossy or unknown tokenisation, or over the context, are rejected,
  not truncated. Each embedding stores retained and failed prepared-row indices, the
  prepared-file SHA-256 and split counts; each prediction archive stores its test
  source-row indices. Mean pooling excludes special tokens.
- **`--missing-labels as-negative`** (the CLI default) treats missing labels in Tox21 and
  MUV as negatives during fitting and CV, as the imported Praski et al. scorer did. Test
  ROC-AUC ignores missing test labels either way. Do not mix the two modes in one results
  file; every row records its mode.
- **kNN is not scored on HIV and MUV** (`score.py` disables it): the imported table has no
  kNN rows there, so the shared candidate set would drop it anyway. The second command adds
  the other two heads for those datasets to the same results file.
- **Resume.** Cache and resume skip on dataset, embedder and head only, so a regenerated
  embedding or a changed mode keeps old rows (audit finding R6). Score with `--no-cache
  --no-resume` into a fresh output directory until that is closed.
- Select the paper-facing head per dataset from the training-side `cv_metric` only, and
  evaluate the selected head's held-out `test_metric` once.

## 2. Select heads by CV and compare on common test rows

`scripts/paper/build_common_row_benchmark.py` reads the head-level CSVs and prediction
archives and never modifies them. For each dataset and embedder it:

- restricts candidate heads to those available for every included embedder, then picks the
  head with the best training-side CV ROC-AUC;
- checks that the saved predictions reproduce the archived `test_metric`, that their source
  rows are unique prepared test rows, that labels match the prepared labels and that the
  archive records the same prepared-file SHA-256;
- reports test coverage, then rescores every model that passed on the test rows all of them
  predicted. ROC-AUC, average precision and positive counts come from the same fixed
  predictions; average precision never influences selection.

```bash
uv run python scripts/paper/audit_split_overlap.py \
  --summary-output outputs/audit/<run>/split_overlap_summary.csv \
  --row-output outputs/audit/<run>/split_overlap_test_rows.csv

uv run python scripts/paper/build_common_row_benchmark.py \
  --results outputs/eval/<run>/results.csv \
  --table-results data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv \
  --table-embedders ECFP ChemBERTa-77M-MLM MoLFormer-XL-both-10pct SELFormer \
  --matrix-labels <embedder>=<label> ECFP=ECFP4 ChemBERTa-77M-MLM=ChemBERTa-2 \
                  MoLFormer-XL-both-10pct=MoLFormer \
  --exclude-datasets ogbg-moltoxcast \
  --split-overlap-rows outputs/audit/<run>/split_overlap_test_rows.csv \
  --output-dir outputs/eval/<run>/common_rows
```

The baselines come from the imported Praski et al. table (see [baselines.md](baselines.md)).
Their heads are chosen by the same CV rule from the same shared candidate set. They have no
prediction archives, so they get status `table_only` and stay out of the common-row and
paired outputs; the paper must say that their test molecules could not be matched to ours.

Outputs:

- `head_candidates.csv`: the selection record; flags heads excluded from the shared set.
- `selected_heads.csv`: archive checks, coverage and file hashes.
- `common_row_scores.csv` and `common_row_scores_no_split_overlap.csv`: paired inputs, the
  latter after removing test rows that share an InChIKey or stereo-insensitive canonical
  SMILES with training-side rows. The primary scores keep the configured splits.
- `paired_task_differences.csv`: per dataset and model pair, the ROC-AUC difference on the
  common rows with a paired bootstrap interval. `clear_winner` names a model only when the
  interval excludes zero; count per-dataset wins from it, not from raw differences.
  `--paired-reference` limits the pairs, `--n-boot` sets the resamples.
- `task_matrix.csv`: each model's own test ROC-AUC of its CV-selected head, including
  table-only baselines.
- `common_task_matrix.csv` and `common_task_matrix_status.csv`: ROC-AUC on shared test rows
  for a fixed cohort of archive-backed models (imported baselines excluded). Pass all
  intended models through `--embedders` so a missing one is an error. If any model fails
  verification or lacks a dataset, every score for that dataset stays missing. Status values
  are `ok`, `incomplete_cohort`, `no_common_rows` and `undefined_roc_auc`.
  `common_task_matrix_no_split_overlap*.csv` repeat both after the overlap exclusion.
- `manifest.json`: input hashes, code revision, arguments and the definition of each matrix.

Use `common_task_matrix.csv` for internal comparisons and `task_matrix.csv` for the external
context. The common-row export does not verify shared supervised training rows or CV folds
(G7.1, audit finding R114). Archives from before commit `112efc5` have no source-row indices
and are left out (`no_row_ids`). Repeated or conflicting runs for one head stop the script.

For aggregate intervals, `scripts/paper/compute_bootstrap_cis.py` resamples tasks and, as
units, the 18 task families in `config/task_families.yaml`, and reports a family-weighted mean
and family win counts. The families were fixed before corrected results existed; do not
regroup tasks after seeing scores. `scripts/paper/audit_task_overlap.py` records the
test-molecule overlap behind them.

## 3. Build the paper tables and figures

Every result comes from one task matrix. Write outputs to a new directory and copy them into
the manuscript after review; the default output paths of some generators hold the archived
analysis.

```bash
OUT=outputs/eval/<run>/paper
uv run python scripts/paper/build_paper_results.py \
  --task-matrix outputs/eval/<run>/common_rows/task_matrix.csv \
  --reference <label> --out-dir $OUT
uv run python scripts/paper/make_appendix_table.py \
  --matrix $OUT/results_matrix_25task.csv --out $OUT/table_pertask.tex \
  --models ECFP4 ChemBERTa-2 SELFormer MoLFormer <label>
uv run python scripts/paper/compute_bootstrap_cis.py \
  --matrix $OUT/results_matrix_25task.csv --out_dir $OUT --figure_dir $OUT/figures \
  --reference <label> --baselines SELFormer ChemBERTa-2 ECFP4 MoLFormer
uv run python scripts/paper/make_paper_figures.py \
  --matrix $OUT/results_matrix_25task.csv --figure-dir $OUT/figures \
  --reference <label> --source-data-dir $OUT/source_data
uv run python scripts/paper/make_loss_curves.py \
  --run-dir runs/<run> --figure-dir $OUT/figures
```

| Output | Manuscript |
|---|---|
| `table2.tex`, `group_means.csv` | `tables/main_results_table.tex` (Table 2) |
| `stats.txt` | win counts, Wilcoxon tests and means quoted in Results |
| `table_pertask.tex` | `tables/pertask_table.tex` (all 25 datasets) |
| `table_bootstrap.tex`, `bootstrap_cis.csv` | `tables/table_bootstrap.tex`; `source_data/` |
| `figures/bootstrap_ci_forest.pdf` | `figures/bootstrap_ci_forest.pdf` |
| `figures/Fig_baselines.pdf` | `figures/Fig_baselines.pdf` |
| `figures/Fig_task_group_distributions.pdf`, `source_data/Fig_task_group_distributions.csv` | `figures/Fig5_task_group_distributions.pdf`; `source_data/` |
| `figures/Supplementary_2.pdf` | `figures/Supplementary_2.pdf` (loss panel only; the trainer does not log masked-token accuracy) |

`Fig_2` needs the released small, base and span columns and is not written. The generators
still assume the historical column set and up to seven bars; the known gaps for five internal
models (R10–R13, R42, R66, R67, R78) are listed in
[code_audit_2026-09-29.md](code_audit_2026-09-29.md). The tokenizer table and the embedders
table's parameter counts are filled by hand from the audit outputs.
