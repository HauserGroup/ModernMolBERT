# Historical clean small-model run guide

> **Historical one-model recipe.** The subsequent five-model SSH/GPU proposal is in the
> [canonical master plan](</Users/skn506/HauserGroup Dropbox/Jakob Madsen/PhD/Manuscripts/ModernMolBERT pre-print manuscript/MASTER_REVISION_PLAN.md#gpu-training-plan--five-model-experiment-draft-29-september-2026>), G1–G7.
> It uses five fresh encoders and supersedes this document's one-model/partial-run
> prescription for that campaign. Keep the details below as the earlier run record
> and command reference; do not launch the expanded experiment from these old commands.

This is a proposed, **single full run** for the submission revision: train one
APE tokenizer using only the ChEMBL 36 training split, pretrain one small
ModernMolBERT encoder with standard masking, then embed and score every one of
the 25 datasets in the benchmark configuration. Keep the released preprint
tokenizer, checkpoints, embeddings, and result CSVs as historical artifacts.
The full encoder command began on 29 September 2026 but stopped after step
16,051/30,000; it has no final model. Its status and the exact remaining
requirement are in [revision_run_record.md](revision_run_record.md) and
[model_retraining_requirement.md](model_retraining_requirement.md).

Run from the repository root. Before starting, freeze the code commit, the
prepared ChEMBL data (`data/pretrain/chembl36_selfies/metadata.json` and Parquet
hashes), the benchmark configuration and prepared splits, package lockfile, and
the hyperparameters below in a run record. The tokenizer's metadata
records its own vocabulary and scanned training-Parquet hashes; the model writes
`run_args.json` and run metadata. Do not label the new run as a reproduction of
the released weights: its tokenizer and training recipe are new.

The run record for `revision_clean_small_v1` is
[revision_run_record.md](revision_run_record.md). It records the frozen
inputs, the verified tokenizer, step-2 results, and the interrupted full-run
status. Do not start or resume model training under the current no-retraining
scope.

## 1. Train the corpus-only tokenizer

The new name is deliberate; the committed 631-token preprint vocabulary must
remain intact. APE learns merges from two million **training-split** SELFIES.
The additional scan adds any primitives seen in the *full* training Parquet,
including component separators. It does not read validation or benchmark
molecules and cannot be combined with either `--extra_vocab_*` option.

```bash
uv run python -m modernmolbert.train_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column selfies --representation SELFIES \
  --tokenizer_train_size 2000000 --max_vocab_size 2000 \
  --min_freq_for_merge 3000 --max_merge_pieces 2 --seed 42 \
  --corpus_primitive_parquet data/pretrain/chembl36_selfies/train.parquet
```

Check the adjacent `.metadata.json`: `extra_vocab_symbols_requested` and
`extra_vocab_symbols_added` must both be zero, and `corpus_primitive_scan`
must identify the complete training Parquet with its row count and SHA256.
The encoder's `--require_corpus_only_vocab` gate compares that hash against
the exact local `train.parquet` it will stream; it rejects a different corpus.
Check lossless round-trip for disconnected SELFIES (`[C].[O]`). If any of these
checks fail, fix the input or tokenizer before model training.

## 2. Validate and preflight

The validator samples rows; a zero unknown rate in a sample is not proof of
full-corpus coverage. The full training-split primitive scan above supplies
that coverage check. Also inspect validation and prepared benchmark coverage
separately; unsupported benchmark symbols should be counted, never added to
the tokenizer from those benchmark inputs.

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --dataset_name data/pretrain/chembl36_selfies --molecule_column selfies \
  --split train --n 10000 --max_seq_length 128 --unk_rate_threshold 0

uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --dataset_name data/pretrain/chembl36_selfies --molecule_column selfies \
  --split valid --n 10000 --max_seq_length 128 --unk_rate_threshold 0

uv run python scripts/audit_benchmark_inputs.py \
  --vocab tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --max-length 128 \
  --output outputs/audit/revision_clean_small_v1_benchmark_inputs.csv
```

Record unknown, conversion-failure, disconnected-component, and truncation
counts by split. If the validation check finds unknowns, report and resolve
that before choosing how validation loss will be used. Do not add its symbols
to the training vocabulary. `scripts/audit_benchmark_inputs.py` reads the
prepared benchmark JSON files; confirm that all 25 configured datasets are
represented in its output.

Use a **separate** debug directory for the short smoke test. `--debug` caps
training at 200 steps, so its loss is not a scientific result.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug --output_dir runs/revision_clean_small_v1_global_shuffle_debug \
  --dataset_name data/pretrain/chembl36_selfies --selfies_column selfies \
  --train_split train --use_validation_split --validation_split valid \
  --global_train_shuffle \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --require_corpus_only_vocab --model_size small --max_seq_length 128 \
  --masking_strategy standard --mlm_probability 0.15 \
  --per_device_train_batch_size 128 --gradient_accumulation_steps 2 \
  --learning_rate 4e-4 --seed 42 --no-bf16
```

Reload `runs/revision_clean_small_v1_global_shuffle_debug/final_model` with
`AutoModelForMaskedLM` and load its tokenizer from the `ape_tokenizer/`
subdirectory, as in [tests.md](tests.md). Require finite logits and a
successful validation pass before the full run.

## 3. One full encoder pretraining run

The archived selected small standard run's `run_args.json` records 30,000
optimizer steps, learning rate `4e-4`, masking probability `0.15`, 128-token
context, seed 42, 1,500 warmup steps, and validation loss for checkpoint
selection. The command keeps those settings and its effective single-device
batch of 256, using 128 × 2 gradient accumulation as a proposed hardware
arrangement. Freeze that arrangement and precision before launch; record
any difference from the archived run. Keep the evaluation schedule aligned
with the save schedule. `--no-bf16` is the portable full-precision setting;
supported CUDA hardware may use `--bf16`. The local training Parquet is ordered
by ChEMBL identifier. `--global_train_shuffle` first shuffles all training
rows as an Arrow index, then applies the streaming buffer shuffle. This
changes the archived data order without changing the training file or the
tokenizer's corpus hash; its cache is kept under the new run directory.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/revision_clean_small_v1 \
  --dataset_name data/pretrain/chembl36_selfies --selfies_column selfies \
  --train_split train --use_validation_split --validation_split valid \
  --global_train_shuffle \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --require_corpus_only_vocab --model_size small --max_seq_length 128 \
  --masking_strategy standard --mlm_probability 0.15 \
  --max_steps 30000 --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 2 --learning_rate 4e-4 \
  --warmup_steps 1500 --eval_size 4096 \
  --eval_steps 5000 --save_steps 5000 \
  --seed 42 --no-bf16
```

On completion, use `runs/revision_clean_small_v1/final_model` as the selected
checkpoint; the trainer loads the best evaluated checkpoint by `eval_loss`
before saving `final_model/`. Keep its tokenizer bundle and metadata with the
weights. Check `trainer_state.json`, final evaluation metrics, run arguments,
and checkpoint reload. Report the observed 128-token truncation rate on the
population actually measured.

## 4. Re-embed and re-score the full configured benchmark

Use a unique embedder name and output CSV. Specify the new model path in both
model and tokenizer arguments so the run record is explicit. The embedding
script now defaults the tokenizer to the supplied model directory if the
argument is omitted. It rejects lossy or unknown tokenization and removes
failed rows before scoring. Each new embedding stores retained and failed
prepared-row indices, the prepared-file SHA256, and split counts; each new
prediction archive stores its test source-row indices. Report valid/failed
test counts and use these indices to form common-row comparisons. Mean
pooling excludes special tokens.

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets all

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/revision_clean_small_v1/final_model \
  --tokenizer-path runs/revision_clean_small_v1/final_model \
  --embedder modernmolbert_revision_clean_small_v1 \
  --batch-size 32 --device auto --max-seq-length 128 --pooling mean

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets all --skip_datasets ogbg-molhiv ogbg-molmuv \
  --heads rf ridge knn --missing-labels as-negative \
  --embedder modernmolbert_revision_clean_small_v1 \
  --output-csv outputs/eval/revision_clean_small_v1/results.csv \
  --checkpoint-dir outputs/eval/revision_clean_small_v1/checkpoints

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets ogbg-molhiv ogbg-molmuv \
  --heads rf ridge --missing-labels as-negative \
  --embedder modernmolbert_revision_clean_small_v1 \
  --output-csv outputs/eval/revision_clean_small_v1/results.csv \
  --checkpoint-dir outputs/eval/revision_clean_small_v1/checkpoints
```

Two settings keep the scoring comparable with the imported Praski et al.
baselines (§5):

- `--missing-labels as-negative` treats missing labels in the multi-endpoint
  datasets (Tox21 and MUV) as negatives during head fitting and CV, as the
  scorer behind that table did. The default, `observed`, fits each endpoint on
  its observed labels only; do not mix the two settings in one results file.
  Each result row records the setting in its `missing_labels` column. Test
  ROC-AUC still ignores missing test labels in both settings.
- kNN is not scored on HIV and MUV. The imported table has no kNN head there
  (it did not finish on MUV in their runs), so the shared-candidate rule would
  drop it anyway, and MUV kNN is the slowest head. The second command appends
  the random-forest and logistic heads for those two datasets to the same
  results file.

`download.py` uses cached prepared datasets by default. Freeze and verify
those splits rather than silently replacing them. The 25 entries in
`src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml`
define this full run; score all eligible heads without
subsampling. New embeddings live at
`data/embedded/<dataset>/modernmolbert_revision_clean_small_v1.joblib`, so
the historical embedding files remain available. Inspect every failed head
and confirm one score row for each eligible dataset/head pair. Select the
paper-facing head for each dataset using **training-side `cv_metric` only**;
evaluate the selected head's held-out `test_metric` once. Keep the complete
head-level CSV and the selected-head table, and derive the manuscript table,
paired statistics, and quantitative figures from that same verified table.
The existing paper aggregation scripts use historical embedder names and
mixed-provenance source files; update their input mapping and verify baseline
comparability before using them for the revision. The run above alone does not
establish that older baseline scores used identical prepared rows or grids.

## 5. Select heads by CV and compare on common test rows

`scripts/paper/build_common_row_benchmark.py` connects the scoring output to
the paper. It reads the head-level CSVs and prediction archives and never
modifies them.

For each dataset × embedder it:

- restricts candidate heads to those available for every included embedder
  in that dataset, then picks the head with the best training-side CV ROC-AUC;
- checks that the saved predictions reproduce the archived `test_metric`,
  that their source rows are unique prepared test rows, and that the labels at
  those rows match the prepared labels and the archive records the same
  prepared-file SHA-256;
- reports test coverage.

For each dataset it then rescores the models that passed those checks on the
test rows all of them predicted. ROC-AUC, average precision, and positive
counts come from the same fixed predictions, and average precision never
influences head selection. The output also records how many endpoints retain
both test classes and the fewest positives among them. `head_candidates.csv` flags heads excluded from
the shared candidate set; `manifest.json` records that set per dataset. Pass
every run that should share a comparison in
one call:

```bash
uv run python scripts/paper/audit_split_overlap.py \
  --summary-output outputs/audit/revision_clean_small_v1/split_overlap_summary.csv \
  --row-output outputs/audit/revision_clean_small_v1/split_overlap_test_rows.csv

uv run python scripts/paper/build_common_row_benchmark.py \
  --results outputs/eval/revision_clean_small_v1/results.csv \
  --table-results data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv \
  --table-embedders ECFP ChemBERTa-77M-MLM MoLFormer-XL-both-10pct SELFormer \
  --matrix-labels modernmolbert_revision_clean_small_v1=MMB-small \
                  ECFP=ECFP4 ChemBERTa-77M-MLM=ChemBERTa-2 \
                  MoLFormer-XL-both-10pct=MoLFormer \
  --exclude-datasets ogbg-moltoxcast \
  --split-overlap-rows outputs/audit/revision_clean_small_v1/split_overlap_test_rows.csv \
  --output-dir outputs/eval/revision_clean_small_v1/common_rows
```

The baselines come from the imported Praski et al. table, which records
training-side CV and test ROC-AUC for every head. Their heads are chosen by
the same CV rule from the same shared candidate set (so kNN drops out on HIV
and MUV for every model), and their test scores are used as published. They
have no prediction archives, so they get status `table_only` and stay out of
the common-row and paired per-dataset outputs; the paper must say that their
test molecules could not be matched to ours. This keeps the revision minimal:
no baseline is re-embedded.

It writes the following outputs (sensitivity files require a split-overlap audit):

- `head_candidates.csv`: the selection record.
- `selected_heads.csv`: archive checks, coverage, and file hashes.
- `common_row_scores.csv`: paired inputs.
- `common_row_scores_no_split_overlap.csv`: paired sensitivity after removing
  prepared test rows that share an InChIKey or stereo-insensitive canonical
  SMILES with training-side rows; the primary scores retain the configured splits.
- `paired_task_differences.csv`: for each dataset and pair of verified models,
  the ROC-AUC difference on the common rows with a paired bootstrap interval
  (the same resampled test molecules score both models). `clear_winner` names
  a model only when the interval excludes zero; count per-dataset wins from
  this column, not from raw score differences. `--paired-reference` limits
  the pairs to one model against each other, and `--n-boot` sets the resamples.
- `task_matrix.csv`: test ROC-AUC of each CV-selected head, one column per
  model (verified archives and table baselines), labelled by `--matrix-labels`.
- `common_task_matrix.csv`: ROC-AUC recomputed on the shared test rows for a
  fixed cohort of archive-backed models; imported table baselines are excluded.
  Pass all intended local model names through `--embedders` so a completely
  missing model is an error. If any model fails verification or lacks a dataset,
  every score for that dataset is left missing, rather than using a smaller cohort.
- `common_task_matrix_status.csv`: dataset status (`ok`, `incomplete_cohort`,
  `no_common_rows`, or `undefined_roc_auc`), expected/verified model counts,
  missing models and the common test-row count where defined.
- `common_task_matrix_no_split_overlap.csv` and
  `common_task_matrix_no_split_overlap_status.csv`: the same export and eligibility
  checks after excluding the flagged supervised-overlap rows.
- `manifest.json`: input hashes, code revision, arguments and explicit definitions
  of each task matrix, including the fixed cohort and status counts.

Use `common_task_matrix.csv` for internal comparisons and bootstrap summaries on
verified shared test molecules. The older `task_matrix.csv` retains its meaning:
each model's own test score, including table-only baselines. The common-row export
does **not** verify shared supervised training rows or CV folds; those require the
additional G7 safeguards in the master plan. Missing dataset rows stay visible as
NaNs and must be accounted for when reporting aggregate denominators.


For the aggregate intervals, run `scripts/paper/compute_bootstrap_cis.py` on
that task matrix:

```bash
uv run python scripts/paper/compute_bootstrap_cis.py \
  --matrix outputs/eval/revision_clean_small_v1/common_rows/task_matrix.csv \
  --out_dir outputs/eval/revision_clean_small_v1/bootstrap \
  --reference MMB-small --baselines SELFormer ChemBERTa-2 ECFP4 MoLFormer
```
 Besides resampling tasks, it resamples the task
families in
`src/modernmolbert/eval/benchmarking_molecular_models/config/task_families.yaml`
as units and reports a family-weighted mean and family win counts.
`scripts/paper/audit_task_overlap.py` records the test-molecule overlap that
supports those families. The families were fixed before corrected results
existed; do not regroup tasks after seeing scores.

Archives from before commit `112efc5` have no source-row indices. They are
reported with status `no_row_ids` and left out of the common-row comparison. A
results CSV with repeated or conflicting runs for one head stops the script;
choose the run explicitly first.

Before reporting a cross-model comparison, record dataset/split IDs, the
number of retained rows for each model and baseline, scoring-grid/version
hashes, and the handling of conversion failures. Report any test-set overlap
with pretraining separately. Cite the historical preprint results only as
historical results until these checks pass.

## 6. Build the paper tables and figures from the task matrix

The paper reports the retrained small model (`MMB-small`) against the four
baselines. The released small, base and span checkpoints are preprint history:
their archived scores used test-selected heads and do not enter the rebuilt
tables or figures, and the internal-comparison figure (`Fig_2`) is dropped.
Every result below comes from the one `task_matrix.csv` written in §5. Write
outputs to a new directory and copy them into the manuscript after review;
the default output paths hold the archived analysis.

```bash
OUT=outputs/eval/revision_clean_small_v1/paper
uv run python scripts/paper/build_paper_results.py \
  --task-matrix outputs/eval/revision_clean_small_v1/common_rows/task_matrix.csv \
  --reference MMB-small --out-dir $OUT
uv run python scripts/paper/make_appendix_table.py \
  --matrix $OUT/results_matrix_25task.csv --out $OUT/table_pertask.tex \
  --models ECFP4 ChemBERTa-2 SELFormer MoLFormer MMB-small
uv run python scripts/paper/compute_bootstrap_cis.py \
  --matrix $OUT/results_matrix_25task.csv --out_dir $OUT --figure_dir $OUT/figures \
  --reference MMB-small --baselines SELFormer ChemBERTa-2 ECFP4 MoLFormer
uv run python scripts/paper/make_paper_figures.py \
  --matrix $OUT/results_matrix_25task.csv --figure-dir $OUT/figures \
  --reference MMB-small --source-data-dir $OUT/source_data
uv run python scripts/paper/make_loss_curves.py \
  --run-dir runs/revision_clean_small_v1 --figure-dir $OUT/figures
```

What each writes, and where it goes in the manuscript:

| Output | Manuscript |
|---|---|
| `table2.tex`, `group_means.csv` | `tables/main_results_table.tex` (Table 2) |
| `stats.txt` | win counts, Wilcoxon tests and means quoted in Results |
| `table_pertask.tex` | `tables/pertask_table.tex` (all 25 datasets, including Tox21 and MUV) |
| `table_bootstrap.tex`, `bootstrap_cis.csv` | `tables/table_bootstrap.tex`; `source_data/` |
| `figures/bootstrap_ci_forest.pdf` | `figures/bootstrap_ci_forest.pdf` |
| `figures/Fig_baselines.pdf` | `figures/Fig_baselines.pdf` |
| `figures/Fig_task_group_distributions.pdf`, `source_data/Fig_task_group_distributions.csv` | `figures/Fig5_task_group_distributions.pdf`; `source_data/` |
| `figures/Supplementary_2.pdf` | `figures/Supplementary_2.pdf` (loss panel only: the revised trainer does not log masked-token accuracy) |

`Fig_2` is not written, because it needs the released small, base and span
columns. Remove it and its text from the manuscript. Fill in by hand, from
existing audits rather than a script: the tokenizer table (use
`tokenizer_population_audit_corpus_v1.json` for the 588-token vocabulary) and
the new model's parameter count in the embedders table.
