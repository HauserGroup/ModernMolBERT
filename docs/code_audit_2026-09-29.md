# Code audit: open findings (audited 2026-09-29, consolidated 2026-09-30)

The audit covered all tracked Python and R in `src/`, `scripts/`, `analysis/`, `R/` and
`tests/`; the vendored benchmark harness only in part. The full first version (about 1,900
lines, including every fixed finding and each verification step) is in git history: see
`git log -- docs/code_audit_2026-09-29.md`. This file keeps what is still open, so
finding numbers (`R…`, plain numbers, `N…`) match commit messages and older notes.

Closed and not listed here: 1, 3, 5, 6, 8–13, 15–17, 20, 22–23, 32, 36–37 and N1–N8;
R1, R5, R7, R8, R18, R19, R23, R25, R26, R30, R34, R35, R39, R40, R44–R46, R49, R50, R58–R60,
R62, R63, R72, R73, R75–R77, R83, R89, R90, R93, R95–R97, R99, R101–R105, R109–R113.
Notes on how these were verified are in the commit messages of `c35f1fa`, `97a13de`,
`30adf4f`, `1fc70b3`, `dfebfd2` and `a589bff`.

**Decisions that hold for the five-model campaign.** The G1–G7 gates in
`MASTER_REVISION_PLAN.md` apply. "Deferred" below means: real, but it does not change the
frozen inputs, the training recipe or the reported scores, or it needs outputs that do not
exist yet. Do not refactor trainer, tokenizer or scorer code between the first launch and
the last accepted result.

## Priority shortlist

Ordered by likely effect on the revised results.

1. **Stale scoring results (R6, R20, R32, R51).** Resume and cache skips key on
   dataset/embedder/head only. They ignore the missing-label mode, the embedding's model
   identity and the prepared-data hash, so a regenerated embedding or a switch to
   `--missing-labels as-negative` keeps old rows. `build_common_row_benchmark` neither
   checks nor records the mode. The pilots used `--no-cache --no-resume`; production
   scoring must too until this is closed. Cheap fix: compare `missing_labels` and
   `prepared_data_sha256` before skipping, include the kNN grid and mode in the version
   hash, record a model-weights hash in embedding metadata, and require one
   `missing_labels` value in the common-row builder.
2. **Manuscript version claim (R47).** `selfies.__version__` reports `2.1.1` in the 2.2.0
   release, and `uv.lock` pinned 2.2.0 before the corpus was prepared, so the corpus
   `metadata.json` and the plan's "selfies 2.1.1" are probably wrong. Confirm the
   preparation environment; record versions with `importlib.metadata.version()`.
   **3 October follow-up:** The accepted Helios environment reports distribution
   version 2.2.0 through `importlib.metadata.version("selfies")` while the module
   still reports 2.1.1. The manuscript now uses the distribution version. The
   historical corpus-preparation environment cannot be reconstructed from its
   module-version field alone; retain that provenance limit rather than claim a
   separately verified preparation version.
3. **Common supervised rows and folds (R114).** Each model drops only its own failed rows
   before `GridSearchCV(cv=5)`, so training rows and folds differ. G7.1 needs one
   eligibility intersection. Lowest-risk place: a scoring-time filter that drops the union
   of every model's `failed_source_row_indices` before `fit_and_eval_embedding` and records
   it in the manifest.
4. **Run provenance (R54–R56).** The attention backend, optimizer and code revision are not
   recorded in `run_identity.json`. The launcher's clean-commit check and the Helios
   manifest cover the revision in practice; the attention implementation and optimizer
   still resolve implicitly from the environment and library defaults.
5. **Paper generators are not ready for five models (R10–R13, R42, R66, R67, R78).** They
   break or mislead with nine columns; see below. Wait until five score matrices exist.
6. **Coverage records (R57, R64).** Rejection reasons are not kept per row and the
   configured 50-row invalid limit is unenforced. The campaign uses measured per-task
   counts instead.
7. **Test gaps on gatekeeping code (R94, R115).**

## Training

- **R3.** With a frozen order there is no reshuffle between passes, so all ~3.2 passes
  present identical batches in the same order (only masks differ). Defensible; state it in
  Methods.
- **R4.** The batch sequence depends on `--num_workers` and microbatch size (verified:
  workers 0, 2 and 4 give three different sequences). Pinned to four workers and 32 × 8 in
  the launcher. The 7.68M presentation count must allow for the uneven end-of-pass batches.
- **R43.** The frozen-order test iterates the dataset, not `get_train_dataloader()` with
  workers, so it does not prove the effective batch sequence. One small test would.
- **R53.** Validation masks are redrawn at every evaluation, so the curve mixes model
  change and mask noise. Worth fixing only if validation curves are reported.
- **R54.** `build_modernbert_config` picks `flash_attention_2` whenever `flash_attn`
  imports. Add `--attn_implementation` (default `sdpa`) and record it, the GPU, CUDA
  version and package versions.
- **R55.** The optimizer is the library default (`adamw_torch_fused`, 0.9/0.999, 1e-8).
  Correct today, but pinned only through `uv.lock`; pass and record it.
- **R56.** Add `{"commit", "dirty"}` to the run identity (`utils.get_git_revision` exists).
- **R68.** `final_model/README.md` advertises a "Best checkpoint" for terminal-step models
  and reports final metrics from freshly drawn masks. State the selection rule and step
  from `selection.json`.
- **R79.** The trainer's tokenizer gate and `validate_tokenizer.py` sample only the start of
  the ID-ordered corpus. `scripts/audit_factorial_tokenizers.py` covers every row, so treat
  the gate as a smoke check or sample through the frozen order.
- **R117.** Fourteen trainer defaults differ from G3 (for example `max_steps`, learning
  rate, `mlm_probability`, `load_best_model_at_end`, seed). The gated launcher passes every
  G3 value; a direct trainer call does not.
- **18.** `DATASET_NAME` still defaults to PubChem10M in three CLIs; `validate_tokenizer`
  defaults to `--max_seq_length 256` against 128 in training.
- **R100.** The Arrow `load_from_disk` path (`--data_dir`, `find_local_dataset`,
  `_local_dataset_matches_request`) is unused; removing it deletes about 80 lines.

## Scoring and evaluation harness

- **R9.** `make_short_model_name` cuts at the first `.`; `EmbeddedDataset.y_np` is a
  `cached_property` that goes stale if read before `remove_failed_embeddings`;
  `fit_multioutput_finite_label_model` ignores `memory_weight` and `n_jobs`.
- **R20.** `load_embedded_dataset` prefers a legacy `<embedder>.json` over a newer joblib.
- **R22.** `evaluate` always computes ROC-AUC, whatever the task. Latent (no regression
  tasks).
- **R24, R65.** Data paths resolve against the working directory (`os.getcwd()` in nine
  places, `tdc_admet_solver` uses `"data/"`), so launching from elsewhere silently creates
  an empty `data/` tree. Resolve against the project root or require `--data-root`.
- **R27, R28.** `audit_benchmark_inputs.py` counts `<unk>` and length, while the featurizer
  rejects on any lossy round trip, special ID, conversion failure or over-context length,
  so predicted and actual coverage can differ. Its defaults are the historical 631-token
  vocabulary and length 128. Take coverage from embedding metadata.
- **R33, R84.** `compare_praski_tables` selects heads with an unstable sort and
  `groupby().first()` (fields from different rows) and collapses variants by test score.
  Diagnostic only; do not let it feed paper numbers.
- **R36.** Audits use different dataset universes (25 configured versus everything in
  `data/prepared`, which includes ToxCast). Totals depend on which one a script used.
- **R52.** `paired_task_differences` and `common_task_matrix` use different cohort rules;
  add a `cohort_complete` column or apply the fixed cohort to both.
- **R57.** Empty inputs and SELFIES conversion failures are skipped without a counter and
  no per-row reason is stored. Store a reason code per failed row and rename `n_truncated`
  to `n_over_context`.
- **R61.** kNN on integer embeddings standardises before a count-Tanimoto metric. Not used
  by the campaign (ECFP4 is imported).
- **R64.** `max_invalid_embeddings` is declared and never read (the three unused directory keys
  were removed).
- **R70.** Each molecule is tokenised three times in `featurize_smiles`; reuse the
  validated `content_ids`. Behaviour unchanged.
- **R91, R92 (and the R21 remainder).** `prediction_export` stamps archives with the current
  grid hash, has no `cv_metric`, uses trapezoidal PR-AUC and raises on unconfigured datasets; `plot_predictions` pools endpoints into a micro-averaged AUROC.
  Neither fits the CV/common-row pipeline; retire them or label their output.
- **R116.** `eval_metrics._normalize_auc_scores` (complexity 36) guesses array orientation
  from shapes; a square input is ambiguous. Normalise once where predictions are produced.
- **R69.** ChEMBL deduplication would collapse every row lacking an InChIKey into one. The
  frozen corpus is unaffected.
- **R85.** `export_benchmark_corpus --split` exports all rows for the 19 datasets without a
  `split` column. Irrelevant once injection is retired (R31).

## Paper generators (C3)

- **R10.** `make_paper_figures.py` has seven bar colours; four baselines plus five `MMB-*`
  columns raise `IndexError`.
- **R11.** The task-group figure rejects NaN cells (a model may lack MUV) and rounds to one
  decimal; extra `MMB-*` labels share colours.
- **R12.** Table 2 averages over different task sets and the "all 25" caption is wrong when
  cells are missing; print `n` or restrict to common tasks.
- **R13.** The five prespecified contrasts are not generated; `compute_bootstrap_cis.py`
  supports only reference-versus-baselines and writes fixed filenames.
- **R14.** Without `--task-matrix`, `build_paper_results.headline()` picks small or base by
  test mean. Legacy path; do not use it for revised claims.
- **R15.** APE token "frequency" uses non-overlapping `str.count` (`[C][C][C][C]` counts 2).
  Count from `ape_tokenize` output if the table is retained.
- **R42.** `make_loss_curves.py` overwrites `Supplementary_2.pdf` on every call and emits no
  source data; name the output after the run and write the `(step, loss)` pairs.
- **R66, R67.** Bootstrap and per-task captions hard-code "-base", `B=10,000` and "not yet
  run"; labels are not escaped.
- **R78.** Fixed axis ranges (0.45–1.0, `ylim(0.55, 0.90)`) clip data.
- **R81.** Generators default to the historical `paper/` snapshot; default to `outputs/…`.
- **R82.** Four generators run at import; wrap them in `main()`.
- **R98.** Ties are exact float equality, but tables print one decimal.
- **R38.** `compute_property_regression.py` drops molecules silently and fits unscaled
  `Ridge(alpha=1.0)`. Optional/historical figure.

## Upload and release (C6)

- **14.** `upload_model.py` hard-codes `MODEL_MAX_LENGTH=128` and defaults
  `--masking_strategy` to `standard`; it overwrites `vocab_size` instead of checking it and
  prefers the repository's `tokenization_ape.py` over the run's copy. APE–SELFIES only.
- **R41.** `upload_tokenizer` writes the hard-coded 631-token card whatever `repo_id`.
- **R80.** Re-uploading to an existing repo leaves stale files behind; pass
  `delete_patterns` or refuse non-empty repos.
- **24, 25.** Three model-card generators (`model_cards.py`, `upload_model.build_readme`,
  the trainer's card) and duplicated pooling logic. Keep one, reading `config.json` and
  `run_args.json`.
- **26, 27.** Duplicate APE loaders and special-ID dicts in the upload scripts; two repository
  root finders plus `parents[N]` in scripts.
- **31.** `copy_tokenizer_artifacts` writes the vocabulary to five places; trimming changes
  the published layout.
- **34.** The dataset README hard-codes "no test split".
- **R106, R107.** The smoke test is APE–SELFIES-only (`tok.vocabulary`); several defaults
  still point at the 631-token vocabulary.

## Data and tokenization (frozen; only relevant if something is regenerated)

- **19.** Pretraining SMILES use partial sanitisation, benchmark SMILES full sanitisation.
  Sized by the representation-overlap audit; no corpus rewrite.
- **28.** `APEPreTrainedTokenizer.encode_molecule` would drop EOS when truncating; unused.
- **33.** `chembl36.py`: `except Exception: continue` drops rows uncounted, the example TSV
  holds train rows only, row-wise `make_split_key` is slow, `load_dotenv()` runs at import;
  guard rejections are labelled `rdkit_parse_failed`.
- **38.** `make_ape_token_table.py` prints `631 - 256 - 5` instead of deriving counts.
- **R37.** `collect_sweep_results.R` marks best runs `NA` instead of `FALSE` when
  `best_*_run.json` is missing.

## Packaging, CI and repository hygiene

- **R47.** See the shortlist. The trainer's run metadata should also record selfies, rdkit
  and tokenizers versions.
- **R48.** Base dependencies include notebook and evaluation-prep packages (`deepchem` is
  imported nowhere); `pyarrow` and `joblib` are imported but undeclared; `aiohttp`,
  `tornado` and `urllib3` are dependabot floors that belong in
  `[tool.uv] constraint-dependencies`. Changing this alters `uv.lock`, so only before the
  commit is pinned or after the campaign.
- **R74.** `figures/FigX_sweep_metrics.*` has no generator in the repository.
- **R86.** Untracked copies of six paper generators (one lacking the LaTeX escape fix) and a
  `tokenizer/alternative/` sit in the manuscript folder; delete them or use symlinks.
- **R108.** `VENDORED.md` lists the local modifications; the upstream revision of the
  copied code was never recorded and is unknown.
- **7.** The sweep launcher still targets the injected-symbol tokenizer and omits
  `--global_train_shuffle`; historical, so update it before any re-sweep.

## Tests

- **R94.** Still untested: a mixed `missing_labels` input to the common-row builder (R51)
  and the sweep-launcher and selector fixes. The scoring resume skip, `validate_args` and the
  trainer's tokenizer gate now have tests.
- **R115.** Line coverage of `src/modernmolbert` under the CI subset is 58 %, lowest exactly
  where it matters: `validate_tokenizer.py` 0 %, the trainer 37 % (`write_run_metadata`, `main`;
  `validate_args` and the tokenizer gate are now covered), `procedure.py` 25 %,
  `score.py` 43 %, `upload_model.py` 26 %, `model_cards.py` 0 %. Small `tmp_path` tests for
  the pure functions, plus a two-step `--debug --max_steps 2` smoke run, cover most of it.

## Considered, not recommended

- Seeded evaluation masks to cut noise in checkpoint and learning-rate selection: needs
  generator plumbing through the collator and Trainer; only worth it for a re-sweep.
- Vectorising span masking: `num_workers >= 4` hides the CPU cost.
- Splitting the trainer or introducing a CLI framework: see
  `simplification_candidates_2026-09-30.md`.
