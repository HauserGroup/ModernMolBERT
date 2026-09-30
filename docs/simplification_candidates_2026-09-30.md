# Simplification candidates (2026-09-30)

Measured at `5058425` plus the working tree; sizes below are from that date. Companion to
`docs/code_audit_2026-09-29.md` (R-numbers refer to it).

**Done (30 September):** T4 (unreferenced files) and the dead-code list, in `a589bff`; B10 in
part (`config/default.yaml`, the stale tooling entries); and the documentation part of B11
(the audit reduced to open items, the one-model guides folded into
[revision_run.md](revision_run.md) and [revision_run_record.md](revision_run_record.md), the
harness `readme.md` reduced to a pointer). Everything else is still open.

## Where the size is

| Area | Files | Python lines |
| --- | ---: | ---: |
| `src/modernmolbert` | 54 | 14,423 |
| `scripts` | 24 | 5,576 |
| `analysis` | 7 (+3 notebooks) | 2,036 |
| `tests` | 55 | 7,193 |

Other counts: 51 files define an `argparse` CLI; the trainer has 60 flags, of
which the production launcher sets 41; `docs/` plus the READMEs held about
4,400 lines, 1,744 of them the audit (`docs/` is now 1,837 lines).

Three causes account for most of the sprawl:

1. **Two generations of the pipeline live side by side.** The preprint-v1 path
   (sweep, symbol injection, test-selected heads, PaCMAP) and the revision
   path (frozen factorial, CV-selected heads, common rows) are both complete.
2. **Each question got its own script.** There are 9 `audit_*` scripts, 6
   tokenizer-coverage checkers and 3 model-card generators.
3. **The same fact is defined in several files**: task groups, model labels,
   the 25-task exclusion, run arguments, scoring resume state.

## Timing constraints

- Production runs must stay on one clean commit, and the launcher pins the
  `uv.lock` hash. Do the work on a branch and do not move the Helios checkout
  until all five runs finish.
- Trainer, `utils.py` and dependencies: after G6.
- Scorer and paper generators: between scoring campaigns, not during G7.
  `library_hash` covers the head grids (including the regression grids), so
  removing those changes resume identity.
- Historical code: the plan (C5) still names `build_benchmark_results_frames.py`,
  `R/collect_sweep_results.R` and the `--include-hetero-span` flags as
  provenance tools. Removing them needs a tag (for example `preprint-v1`) and a
  one-line plan change. That decision is yours.

## Tier 1 — retire the preprint-v1 generation (tag, then delete)

About 6,200 lines of code and 1,500 lines of tests, roughly a quarter of the
Python. None of it is imported by the revision path except one function.

| # | Group | Code | Tests | Why it is no longer on the path |
| --- | --- | ---: | ---: | --- |
| S1 | Sweep tooling: `scripts/sweeps/run_sweep.py`, `select_pretraining_run.py`, `analysis/sweep/*` (2), `R/*` (2), `results/*.csv`, `figures/FigX*` | 2,209 | 183 | The campaign has no sweep and selects the terminal step. Removing `R/` also removes `renv.lock`, `.Rprofile`, `air.toml` and the 328 MB local `renv/`; R is used for nothing else. The plan already says `FigX.R` is not a verified source for the retained figure. |
| S2 | Symbol injection: `export_benchmark_corpus.py`, `filter_missing_selfies_symbols.py`, `patch_tokenizer_vocab.py`, `audit_injected_symbols.py`, `tokenizer/extra_symbols/` | 948 | 456 | C2: benchmark symbols are never injected. Also removes `--extra_vocab_symbols_path`, `--extra_vocab_selfies_path` and `add_tokens_to_vocabulary`. |
| S3 | Preprint result path: `build_benchmark_results_frames.py`, `compare_praski_tables.py` (+ `benchmarks/praski/`, 5 files), `audit_published_head_selection.py`, `audit_saved_predictions.py`, `prediction_export.py`, `plot_predictions.py` | 1,664 | ~460 | Superseded by `build_common_row_benchmark.py` → `build_paper_results.py --task-matrix`. Keep `collapse_best_head` (67 lines) by moving it into the common-row builder. The default (archived-matrix) mode and hetero-span flags in three generators go with it. |
| S4 | PaCMAP and property regression: `visualize/embed_selfies_for_pacmap.py`, `load_chembl_for_pacmap.py`, `compute_property_regression.py`, `arrange_panes.py`, two notebooks | 988 | 279 | C5: `Fig_4` is historical and may be dropped. It carries a second embedding implementation and the `pacmap`, `colorcet` and `datashader` dependencies. |
| S5 | Static model cards: `model_cards.py` | 416 | 112 | Hard-codes the four preprint checkpoints. See B1. |

## Tier 2 — one source of truth

| # | Duplication | Evidence | Simplification |
| --- | --- | --- | --- |
| B1 | Three model-card generators | Trainer `write_run_metadata` (about 110 template lines), `upload_model.build_readme` (145), `model_cards.card` (164). They have already drifted: C5 records a 0.20 versus 0.15 mask rate between two of them. | Keep the one in the upload path. The trainer writes JSON only. |
| B2 | Run arguments written three times | `run_identity.json`, `run_args.json`, `run_metadata.json`, plus `README.checkpoint.md` and `final_model/README.md`. | One `run_identity.json`; append final metrics to it. After G6, because resume compares it. |
| B3 | Three layers of scoring resume state | Result-CSV rows, per-head JSON checkpoints, per-dataset CSV checkpoints, steered by `--cache`, `--resume`, `--safe`. | Result rows keyed on the full identity as the only state. About 250 lines, and it removes the R6 class of stale-skip bugs. G7 needs resume safeguards anyway; build them here instead of adding a fourth mechanism. |
| B4 | Two on-disk formats for prepared data | `download.py` writes joblib and "legacy" JSON. Eight readers use the JSON, which is the frozen source of truth; the joblib pickles break on module moves (hence `migrate_prepared_legacy_cache.py`). | JSON only for prepared data, joblib only for embeddings. Removes three fallback branches, the migration script, R20 and R34. |
| B5 | `missing_labels` default at five levels, two values | `as-negative` in `score.py` (2), `observed` in `procedure.py` (2) and `train.py` (2). | Required keyword below the CLI (R101). |
| B6 | Six tokenizer-coverage tools | `validate_tokenizer.py` (336, 0% covered), the trainer gate, `audit_selfies_inputs.py`, `audit_factorial_tokenizers.py`, `audit_benchmark_inputs.py`, `check_tokenized_lengths.py` (410), plus two in `analysis/validation/`. Each loops over molecules for unknown, truncation and length counts. | One audit command over a corpus Parquet or prepared benchmark, built on `compute_tokenization_stats`, which the trainer gate also calls. About 700 lines. |
| B7 | SMILES → model input in seven places | `sf.encoder` call sites in the featurizer, `chembl36.py`, three audit scripts, `export_benchmark_corpus.py` and `check_tokenized_lengths.py`, each with its own failure handling. | One helper returning the text or a rejection reason. This is also what R57 (per-row rejection reasons) needs. |
| B8 | Paper constants | Task groups in 4 files, model labels in 5, the ToxCast exclusion in 5, baseline lists in 7. `datasets.yaml` already has `source.group`. | One small module for labels, order and exclusions; groups read from the registry. Do it as the first step of C3, which otherwise edits five files for the five new names. |
| B9 | Paper generators as import-time scripts | `build_paper_results.py` (335 lines, no functions to test), `make_paper_figures.py`, `make_appendix_table.py`, `make_loss_curves.py` parse arguments at import. | `main()` and a single `--task-matrix` input, together with S3. |
| B10 | Inert configuration | `config/default.yaml` is read by nothing; the `defaults:` keys are Hydra leftovers; `score.yaml` defaults to HIV only (R35); `EmbeddingConfig` has three unused directories; `as_list` exists twice. | Keep `datasets.yaml` and `task_families.yaml`; paths become CLI defaults. |
| B11 | Evaluation described in four documents | `README.md`, `docs/evaluation.md`, the harness `readme.md`, `docs/revision_run.md`. | Partly done (see above). After C6, fold `revision_run.md` into one reproduction guide with the command sequence C3 requires. |
| B12 | Paper outputs in two repositories | `paper/tables`, `paper/figures`, `paper/source_data`, `results/` and `tokenizer/alternative` also exist in the manuscript repository. | Generators write to `outputs/`; the manuscript repository holds the canonical copies. |

Small helpers, only when touching the file: five SHA-256 functions (the
launcher's own copy is justified, it runs before the environment exists), four
ways to find the repository root, three JSON readers.

## Tier 3 — trim the surface

- **T1. Trainer data paths (after G6).** Three training-data paths exist:
  streaming with a hash-bucket split, global shuffle without an order file, and
  the frozen order. A pretokenized `input_ids` branch has no producer in the
  repository. Keeping only local Parquet + frozen order + frozen validation IDs
  removes six flags (`--data_dir`, `--selfies_column`, `--shuffle_buffer_size`,
  `--val_split_mod`, `--val_split_bucket`, `--hf_login`), about 150 trainer
  lines and about 300 lines of `utils.py` (`get_streaming_dataset`,
  `find_local_dataset`, the ZINC/PubChem routing). R1 disappears with the paths. Then set the trainer
  defaults to the G3 recipe (R117) and the launcher shrinks to its gate checks.
- **T2. Scorer options.** Subsampling (4 flags, about 170 lines) is mentioned in
  no document, plan or analysis. No configured dataset is a regression task, so
  the regression grids, scorers and plotting are unexercised. The `local`
  dataset source (55 lines, plus `CONTRIBUTING.md`) has no configured dataset.
- **T3. Dependencies (after G6; `uv.lock` is pinned).** `deepchem` is imported
  nowhere. `jupyterlab`, `ipykernel` and `iprogress` are runtime dependencies.
  The `eval`, `eval-prep` and `pretrain-data` groups repeat the main list except
  for `pytdc` and `pyarrow`; `pyarrow` is imported by the trainer but not
  declared there. Stale configuration: Ruff and Pyright entries for a
  `notebooks/` directory that does not exist, a Pyright comment about Hydra, a
  `pyupgrade` hook that duplicates Ruff `UP`.
- **T5. Generated files in two roots.** Predictions, plots, checkpoints and a
  results CSV sit under `data/`; evaluation and audit outputs under `outputs/`.
  For G7, write everything generated under `outputs/revision_factorial_v1/` and
  leave `data/` for inputs. Do not move old outputs.
- **T6. Campaign one-offs.** `freeze_training_order.py`,
  `audit_factorial_tokenizers.py`, `build_revision_coverage_manifest.py` and the
  launcher are needed for C6 reproduction. Group them in one campaign folder so
  `scripts/` shows only the pipeline.

## Considered, not recommended

- A unified CLI framework or a config system. The count falls from 51 to about
  27 through Tiers 1–2; a framework adds a layer without removing a script.
- Splitting the 1,598-line trainer into modules. Remove the unused paths first
  (T1); what remains is about 1,100 lines with one data path.
- Replacing the custom collator. Span masking backs published checkpoints.
- Moving `tokenization_ape.py` into the `tokenization/` package. It must stay a
  single file because it ships as remote code with each APE model.
- Renaming `train_selfies_ape_modernbert.py` and `ModernMolBERTSelfiesFeaturizer`
  (both now handle SMILES and BPE). The names mislead, but the rename touches
  the launcher, documents and tests for no functional gain. Do it only with T1.
- Replacing the vendored harness with the upstream package. It has diverged
  (finite-label heads, row provenance).

## Suggested order

1. Done: dead code, T4.
2. With G7: B3, B5, T5.
3. First step of C3: B8, B9.
4. After G6: T1, B2, B6, T3.
5. After C3 results are frozen: tag, then Tier 1, B1, B4, B10, T2.
6. After C6: B11, B12, T6.

Estimated end state: non-test Python from about 22,000 to about 13,000 lines,
tests from 7,200 to about 5,500, CLIs from 51 to about 27, one language.
These are estimates from the line counts above, not a measured result.
