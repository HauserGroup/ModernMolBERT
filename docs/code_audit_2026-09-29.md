# Code audit: robustness, simplification, bugs (2026-09-29)

Audited oldest-touched first. Covered all tracked Python and R in `src/`, `scripts/`,
`analysis/`, `R/` and `tests/`; the vendored benchmark harness was only spot-checked and
`renv/activate.R` was skipped. The audit itself edited no code.

## Re-check of the working tree (2026-09-29, uncommitted changes on `second-sweep`)

Each finding below carries a status: **Fixed**, **Partial**, or **Open**. Fixes that
introduced new problems are listed under [New issues from the fixes](#new-issues-from-the-fixes).

| Status | Findings |
| --- | --- |
| Fixed | 1, 3, 5, 6, 15, 16, 17, 20, 23, 36, 37 |
| Partial | 4, 22, 28, 34 |
| Open | 2, 7–14, 18, 19, 21, 24–27, 29–33, 35, 38 |
| New | N1–N8 |

Test status (CI marker subset): **1 failed**, 281 passed, 2 skipped. The failure is N5.

Measurements taken during the re-check:

- The SMILES guard now drops 0 of 81,809 RDKit-valid TDC ADMET rows (was 407).
- `data/embedded/*/*.joblib`: still 133 of 133 unloadable.
- Existing prepared data predate the guard: prepared test sizes equal the raw TDC test
  files (CYP2D6 2626/2626, CYP3A4 2467/2467). Finding 1 therefore never affected the
  historical benchmark rows; it would only have affected a re-download.
- Training molecules longer than 128 tokens (including BOS/EOS), full
  `data/pretrain/chembl36_selfies/train.parquet`: APE SELFIES 6 (max 213), APE SMILES 4
  (max 229), BPE SELFIES 5 (max 178), `_corpus_v1` APE SELFIES 6 (max 213). This matters
  for N1.

## New issues from the fixes

- **N1. Training now crashes on the first over-length molecule.** (High)
  `_encode_without_truncation`
  ([train_selfies_ape_modernbert.py:592](../src/modernmolbert/train_selfies_ape_modernbert.py#L592))
  raises inside the streaming `map`, and `keep_train`
  ([:769](../src/modernmolbert/train_selfies_ape_modernbert.py#L769)) does not filter by
  length. Every revision tokenizer has 4–6 training molecules over 128 tokens, and a
  30k-step run covers the corpus about 3 times, so every run will hit one and abort
  mid-training. The eval set builder raises the same way. Fix: drop over-length rows in
  `keep_train` (and in the eval builder) and record the count in run metadata, rather than
  raising.

- **N2. `run_sweep.py` is now incompatible with the trainer.** The trainer refuses a
  non-empty fresh output directory
  ([:389](../src/modernmolbert/train_selfies_ape_modernbert.py#L389)), but
  [run_sweep.py](../scripts/sweeps/run_sweep.py#L268) creates `output_dir/train.log` before
  launching it. Every sweep run fails immediately. Fix: write the log next to the run
  directory, or let the trainer ignore `train.log`.

- **N3. `selection.json` misreports the selected step.**
  [:1373](../src/modernmolbert/train_selfies_ape_modernbert.py#L1373) records
  `trainer.state.global_step`, which stays at `max_steps` after `load_best_model_at_end`
  loads an earlier checkpoint. With the default settings, `selected_step` claims the
  terminal step while `selection_rule` says `best_validation`. Use `best_global_step` when
  the best model was loaded. The `max_steps` check also raises after `save_model`, leaving
  a `final_model/` without tokenizer files or metadata; check before saving.

- **N4. The pinned ModernBERT config is not committed and is not packaged.**
  `configs/` is untracked, so the trainer fails on any other checkout.
  `_MODERNBERT_BASE_CONFIG`
  ([:852](../src/modernmolbert/train_selfies_ape_modernbert.py#L852)) is resolved via
  `parents[2]`, which only works from a source checkout, not an installed wheel. Commit it;
  optionally ship it as package data.

- **N5. A test fails.**
  `tests/test_corpus_primitive_scan.py::test_corpus_only_scan_rejects_benchmark_symbol_injection`
  builds a `Namespace` without the new `aligned_sample_parquet` attribute, so
  `validate_args` ([train_tokenizer.py:205](../src/modernmolbert/train_tokenizer.py#L205))
  raises `AttributeError`. Update the test's Namespace.

- **N6. The LaTeX fix renders tokens incorrectly.** `\detokenize`
  ([make_ape_token_table.py:114](../scripts/paper/make_ape_token_table.py#L114)) doubles
  `#` and adds a space after control words. Verified with `etex`: `[#Branch1][\C]` renders
  as `[##Branch1][\C ]`. Use an explicit escape map instead (`#`→`\#`,
  `\`→`\textbackslash{}`, `_`→`\_`, `&`→`\&`, `%`→`\%`, `$`→`\$`, `{`/`}`→`\{`/`\}`).
  The existing `outputs/eval/paper/table_ape_tokens.tex` is still the old, unescaped
  version until the script is rerun.

- **N7. Multi-target regression would break in the harness.** The new regression branch
  ravels `y` ([supervised/train.py:102](../src/modernmolbert/eval/benchmarking_molecular_models/supervised/train.py#L102)),
  which flattens multi-target labels. Latent: no regression datasets are configured.

- **N8. Run metadata is not uploaded under its new name.** The trainer now writes
  `run_metadata.json`, but `upload_model.py`
  ([:606](../src/modernmolbert/upload_model.py#L606)) still stages
  `ape_tokenizer_metadata.json` and not `run_metadata.json`, and `utils.py:54` still treats
  `ape_tokenizer_metadata.json` as a tokenizer-metadata alias. New runs lose their run
  metadata on upload; old runs still have it staged as tokenizer metadata. (See 22.)

## High: affects results or breaks the pipeline

1. **Fixed. The SMILES guard dropped valid molecules.**
   [rdkit_safety.py](../src/modernmolbert/common/rdkit_safety.py) now checks bracket atoms
   against the RDKit periodic table plus aromatic symbols and `*`; 0 valid TDC rows are
   dropped. Correction to the original finding: the existing prepared data predate the
   guard, so historical benchmark rows were never affected. Leftover: guard rejections are
   still labelled `rdkit_parse_failed` in
   [chembl36.py:312](../src/modernmolbert/data/chembl36.py#L312).

2. **Open. All 133 cached embeddings in `data/embedded/*/*.joblib` can't be loaded.**
   - They were pickled under the old module path `...benchmarking_molecular_models.src`,
     which the package flattening (cc37d13) removed. Loading raises `ModuleNotFoundError`.
   - The embed step skips any output file that already exists, so the failure only shows
     up later, in scoring.
   - Fix: regenerate them, or add a one-off `sys.modules` alias to migrate. Longer term,
     store the arrays as npz plus json instead of pickled classes.

3. **Fixed. Embeddings used a longer context than training.** The featurizer now
   defaults to `model.config.max_position_embeddings` and rejects larger values; the embed
   CLI defaults to `None`. Leftover: [01A_ideal_masking_probability.py:157](../analysis/sweep/01A_ideal_masking_probability.py#L157)
   still passes `max_seq_length=256` and will now raise (the script is broken anyway, see 11).

4. **Partial. The generated APE token table will not compile in LaTeX.** Tokens now go
   through `_safe`, but `\detokenize` renders them wrong (N6).

5. **Fixed. Sharded training layout broke downstream scripts.** `train_shards` now
   defaults to 1. Scripts still require a single `train.parquet`, so an explicit
   `train_shards>1` will still break them.

6. **Fixed. ChEMBL prep could reuse stale chunks.** The checkpoint directory is now keyed
   by a hash of the config and source columns, and resumed chunks count converted rows the
   same way as fresh ones.

## Medium: wrong defaults and silent misbehaviour

7. **Open. The sweep launcher differs from the revision recipe.**
   [run_sweep.py:29](../scripts/sweeps/run_sweep.py#L29) uses the injected-symbol tokenizer
   rather than `_corpus_v1`, and passes neither `--global_train_shuffle` nor
   `--require_corpus_only_vocab`. A re-sweep would bring back the ChEMBL-ID ordering bias.
   It is now also broken by N2.

8. **Open. Sweep resume never retries failed runs.**
   [run_sweep.py:241](../scripts/sweeps/run_sweep.py#L241) treats any non-empty run
   directory as done, and a failed run leaves `train.log` behind. Check for
   `final_model/model.safetensors` instead. The trainer's new `--resume_from_checkpoint`
   could be used here.

9. **Open. `best_run.json` gets overwritten and mislabelled.**
   - [select_pretraining_run.py](../src/modernmolbert/select_pretraining_run.py) ignores
     `--masking_strategy` when naming the default output, so per-strategy runs overwrite
     each other.
   - The R collector expects `best_{span,standard}_run.json`.
   - The `best_metric` field is trainer-state `best_metric`, not the `selection_metric`
     actually used for ranking.
   - Only change so far: it reads `run_metadata.json`, falling back to the old name.

10. **Open. `FigX.R` no longer matches the collector's output.**
    - [FigX.R:24](../R/FigX.R#L24) pivots on `eval_loss` and `eval_masked_accuracy`, but
      [collect_sweep_results.R:191](../R/collect_sweep_results.R#L191) only writes
      `final_*` and `best_logged_*` columns.
    - The learning-rate label uses float `==` (line 17), so other rates become NA.
    - It writes to `figures/` rather than `paper/figures/`.

11. **Open. The 01A notebook resolves the repository root incorrectly.**
    - [01A_ideal_masking_probability.py:44](../analysis/sweep/01A_ideal_masking_probability.py#L44)
      resolves to `analysis/`, not the repo root, and writes to a `notebooks/` directory
      that doesn't exist. Use `common.paths.find_project_root`.
    - The cached and fresh branches compute ROC-AUC with different functions.

12. **Open. Fixed-eval default path is wrong.**
    - [fixed_eval_best_models.py:121](../analysis/sweep/fixed_eval_best_models.py#L121)
      defaults to `valid/valid.parquet`, which doesn't exist (`valid.parquet`).
    - Line 226 uses `best_model_checkpoint` as a raw relative path, so moved runs are
      silently skipped. Resolve it against the repo root as
      [upload_model.py:152](../src/modernmolbert/upload_model.py#L152) does.

13. **Open. `patch_tokenizer_vocab.py` looks for the wrong metadata file.**
    - [patch_tokenizer_vocab.py:135](../src/modernmolbert/tokenization/patch_tokenizer_vocab.py#L135)
      looks for `<stem>_metadata.json`, but the convention is `<stem>.metadata.json`. A
      patch therefore changes the vocab without updating its SHA, and loading later fails.
    - Duplicate symbols in the input file (line 98) create gaps in the ID sequence.
    - With corpus-only now the main path, deleting this script and
      `filter_missing_selfies_symbols.py` may be simpler than fixing them.

14. **Open. `upload_model.py` uses fixed values instead of reading them from the run.**
    - `MODEL_MAX_LENGTH=128` is hard-coded ([:26](../src/modernmolbert/upload_model.py#L26)).
    - `--masking_strategy` defaults to `standard` ([:104](../src/modernmolbert/upload_model.py#L104)),
      so a span model gets a standard `collator_config.json` unless the flag is remembered.
    - `vocab_size` is overwritten ([:190](../src/modernmolbert/upload_model.py#L190))
      instead of checked, which hides a tokenizer/model mismatch.
    - `find_tokenization_code` prefers the repo's current `tokenization_ape.py` over the
      copy saved with the run ([:355](../src/modernmolbert/upload_model.py#L355)).
    - Read these from `run_args.json` and `config.json`.

15. **Fixed. Staging-directory deletion risk.** `make_staging_dir` now refuses a
    non-empty directory instead of deleting it. The module docstring's "without pulling in
    torch via utils" rationale is still stale (see 26).

16. **Fixed. Evaluation kept all logits in memory.** `preprocess_logits_for_metrics` now
    returns argmax predictions.

17. **Fixed. Dataset auto-detection ran before the explicit path.** An explicit local
    Parquet path is now honoured first.

18. **Open. Stale defaults.**
    - `DATASET_NAME` still defaults to PubChem10M in `train_tokenizer.py:57`,
      `validate_tokenizer.py:41` and `train_selfies_ape_modernbert.py:68`, so omitting
      `--dataset_name` streams PubChem from the Hub.
    - `validate_tokenizer` still uses `--max_seq_length 256`
      ([validate_tokenizer.py:99](../src/modernmolbert/validate_tokenizer.py#L99)) against
      128 in training, so it understates truncation (which now matters more, see N1).
    - The trainer's default `--mlm_probability 0.30` matches no documented run.

19. **Open. Pretraining and benchmark SMILES are canonicalized differently.**
    Pretraining uses partial sanitization
    ([chembl36.py](../src/modernmolbert/data/chembl36.py), `SANITIZE_SYMMRINGS |
    SANITIZE_SETAROMATICITY`); benchmarks use full sanitization
    ([data_v2.py:288](../src/modernmolbert/eval/benchmarking_molecular_models/common/data_v2.py#L288)).
    For SMILES checkpoints this is a shift in input distribution. Size it with the
    existing representation-overlap audit before changing anything.

20. **Fixed. Latent harness bugs.** Regression now uses the `r2` scorer, the solver key
    typo is fixed, the fallback uses `saga`, and non-finite CV scores raise. See N7 for a
    remaining latent issue.

21. **Open. PaCMAP embeddings differ from benchmark embeddings.**
    - [embed_selfies_for_pacmap.py:27](../src/modernmolbert/visualize/embed_selfies_for_pacmap.py#L27)
      includes BOS/EOS in the mean pool, so the PaCMAP figure and the property R² use
      different embeddings than the benchmark and the model card.
    - `subfolder="ape_tokenizer"` is forced even when `--tokenizer-path` is given
      explicitly ([:50](../src/modernmolbert/visualize/embed_selfies_for_pacmap.py#L50)).
    - Reuse `ModernMolBERTSelfiesFeaturizer` or `mean_pool_excluding_token_ids`.

22. **Partial. File-name collision in run metadata.** The trainer now writes
    `run_metadata.json`, but the upload path and the alias list were not updated (N8).

23. **Fixed. Bootstrap RNG was shared across comparisons.** Both
    `compute_bootstrap_cis.py` and `paired_task_differences` now seed each comparison from
    a hash of its labels.

## Low: simplification and cleanup

24. **Open. Three model-card generators can drift apart:**
    [model_cards.py](../src/modernmolbert/model_cards.py) (hard-coded parameter counts and
    learning rates), [upload_model.py:437](../src/modernmolbert/upload_model.py#L437) and
    the card in `write_run_metadata` in `train_selfies_ape_modernbert.py`. Keep one that
    reads its facts from `config.json` and `run_args.json`.

25. **Open. Pooling logic is duplicated** in
    [upload_model.py:291](../src/modernmolbert/upload_model.py#L291) and the card snippet.
    Reuse `mean_pool_excluding_token_ids`.

26. **Open. Duplicate helpers.**
    - The APE loaders in [upload_model.py:209](../src/modernmolbert/upload_model.py#L209)
      and [:224](../src/modernmolbert/upload_model.py#L224) repeat each other.
    - `sha256_file` in `fixed_eval_best_models.py` duplicates `file_sha256`.
    - Special-ID dicts in `upload_model.py` and `upload_tokenizer.py` duplicate
      `utils.EXPECTED_SPECIAL_IDS`. The "without pulling in torch via utils" rationale in
      [hf_upload.py:4](../src/modernmolbert/hf_upload.py#L4) is stale.

27. **Open. Two root finders:** `utils.repo_root` and `common.paths.find_project_root`;
    scripts also use `__file__.parents[2]`, and
    [build_benchmark_results_frames.py:37](../scripts/paper/build_benchmark_results_frames.py#L37)
    depends on the current working directory. The new `_MODERNBERT_BASE_CONFIG` adds
    another `parents[2]` (N4).

28. **Partial. Dead code.**
    - Removed: `_LATEXSAFE` in `make_ape_token_table.py`.
    - Still present: `encode_molecule`
      ([tokenization_ape.py:335](../src/modernmolbert/tokenization_ape.py#L335)), which
      would drop EOS when truncating; `filter_zinc20_chembl36_by_source` and the ZINC
      constants in `utils.py`; the environment-variable juggling in
      [count_hf_params.py:98](../analysis/examples/count_hf_params.py#L98).

29. **Open. Overlapping audit scripts.** `analysis/tokenization/check_tokenized_lengths.py`,
    `scripts/audit_benchmark_inputs.py` and `scripts/audit_selfies_inputs.py` overlap, as
    do `analysis/pretraining_eval_overlap.py` and
    `scripts/paper/audit_pretraining_representation_overlap.py`. Audits are also split
    between `scripts/` and `scripts/paper/`, and `regen_groupfig.py` lives in
    `src/modernmolbert/visualize/`.

30. **Open. Duplicated paper constants.** Task-group maps, `EXCLUDED_DATASETS`, pretty
    names and the 25-task counts are repeated across `build_paper_results.py`,
    `make_appendix_table.py`, `regen_groupfig.py` and `make_paper_figures.py`. Move them
    into one shared module.

31. **Open. Tokenizer artifacts are over-copied.** `copy_tokenizer_artifacts`
    ([utils.py](../src/modernmolbert/utils.py)) writes the vocab to 5 places and metadata
    aliases to 4 directories. Trimming it touches the published layout, so it's low
    priority.

32. **Open. Pattern and comment disagree.** The `SMILES_RE` comment in
    [tokenization_ape.py:26](../src/modernmolbert/tokenization_ape.py#L26) says organic
    subset only, but the pattern also allows bare `K` and `H`.

33. **Open. Small issues in `chembl36.py`.**
    - `except Exception: continue` (line 234) drops rows without counting them.
    - The example TSV only ever contains train rows (`head(n)` after concat).
    - Row-wise `apply(make_split_key)` is slow on 2.4M rows.
    - `load_dotenv()` runs at import time.

34. **Partial. Dataset README issues.** Fixed: the usage snippet now uses `--repo_id`,
    the overlap wording is corrected, and the license is now CC BY-SA 3.0. Still open:
    "no test split" is hard-coded.

35. **Open. Stale usage paths and docstrings** in `arrange_panes.py`,
    `make_ape_token_table.py`, `check_tokenized_lengths.py`, `load_chembl_for_pacmap.py`
    and `fixed_eval_best_models.py` (which also lists `fixed_eval_valid_full.pt` twice).
    Also: `check_tokenizer_model_compatibility.py` defaults to a run directory that doesn't
    exist; the trainer's preset comment has the wrong special-ID order; the featurizer
    docstring doesn't mention the over-length drop.

36. **Fixed. Training needed the network.** The base config is now pinned in
    `configs/modernbert_base_config.json` with a SHA check; see N4 before relying on it.

37. **Fixed. Inefficient tokenization stats.** `compute_tokenization_stats` now reuses
    the untruncated encoding for lengths and unknown-token counts.

38. **Open. Magic numbers.** `make_ape_token_table.py` prints `631 - 256 - 5` instead of
    deriving counts from the vocab.

## Considered, not recommended

- **Seeded eval masking to reduce noise in checkpoint and learning-rate selection.** The
  gain is real, but it needs generator plumbing through the collator and Trainer, and
  `analysis/sweep/fixed_eval_best_models.py` already covers apples-to-apples comparison.
  Only worth it if you re-sweep.
- **Vectorizing span masking** ([collator.py:215](../src/modernmolbert/collator.py#L215)).
  `num_workers >= 4` hides the CPU cost, so the added complexity isn't worth it.

## Implementation disposition for the five-model revision (29 September 2026)

This section records the follow-up to the audit. "Deferred" means the finding
is real but outside the frozen five-model experiment, or depends on outputs that
do not exist yet. These are explicit boundaries, not a claim that the old code
or artifact is correct. The G1–G7 gates in `MASTER_REVISION_PLAN.md` still apply.

| Finding | Disposition and reason |
|---|---|
| 1 | **Fixed guard, frozen old cohort.** The bracket-element guard accepts RDKit elements and `*`. Rebuilding all prepared data changed CYP1A2 train/test assignments by ~2,015 molecules despite identical molecules, because the TDC scaffold split depends on raw row order. The pre-existing 26 legacy JSON datasets were migrated to current-module joblib without recomputing splits. That preserves G7 and the imported baseline context. A separate rebuilt cohort and row-change audit are retained under ignored `data/` and `outputs/audit/`; they are not used for revised scores. |
| 2 | **Migration path fixed; old embeddings deferred.** The 26 frozen prepared datasets are loadable again. Reuse of an existing embedding now checks that it unpickles and matches the prepared-source SHA. The 133 historical embedding caches remain incompatible; they belong to old model names, and the five new embeddings will be generated under new names after training. A silent skip is no longer accepted. |
| 3 | **Fixed.** Embedding context defaults to the trained model context, rejects larger overrides, and counts over-context molecules instead of shortening them. |
| 4 | **Fixed generator.** APE tokens are emitted using LaTeX detokenization. Existing generated tables will be regenerated when C5 assembles final artifacts. |
| 5 | **Fixed.** New ChEMBL preparation defaults to one `train.parquet`. The frozen existing corpus is unchanged. |
| 6 | **Fixed.** ChEMBL chunk checkpoints are keyed by preparation configuration and source content; resumed conversion counts are consistent. |
| 7–8 | **Deferred.** The old sweep and its launcher are excluded by G1. Changing that launcher would not affect any of the five direct, explicit training commands. If a sweep is resumed, update its tokenizer/corpus-only flags and completion predicate first. |
| 9–12 | **Deferred.** Selector, R sweep plot, 01A notebook and fixed-eval sweep analysis consume historical sweep runs, not the terminal-step factorial runs. They must be corrected before any historical sweep reanalysis is published. |
| 13 | **Fixed.** Vocab patching deduplicates requested symbols and updates the input's `.metadata.json` convention, writing matching metadata for a new output stem. The factorial tokenizers are trained afresh and are not patched. |
| 14 | **Partially fixed; C6 gate remains.** Upload checks saved vocab/context, reads collator settings from `run_args.json`, and prefers saved tokenizer code. The existing uploader and card remain APE–SELFIES-specific and must be generalized and dry-run verified for all five before public checkpoint release. |
| 15–17 | **Fixed.** Staging refuses nonempty directories; eval logits are reduced before accumulation; explicit dataset paths win over name autodetection. |
| 18 | **Deferred.** Historical CLI defaults are still accepted for old scripts. The factorial launcher must supply dataset, context and masking probability explicitly and pin them in `run_identity.json`; no run may rely on those defaults. |
| 19 | **Measured, no corpus rewrite.** Canonicalization differences are recorded by the representation-overlap audit referenced in the master plan. Rewriting the frozen 2,390,314-row pretraining corpus would change the scientific input and all tokenizer hashes. Both SMILES and SELFIES models use the same saved molecule pairs; the distribution difference is a stated limitation. |
| 20 | **Fixed.** Regression CV uses R², invalid logistic fallback options were removed, and nonfinite CV selection fails. No regression tasks enter the 25-task factorial matrix. |
| 21 | **Deferred.** Historical PaCMAP uses an old checkpoint and is optional in C5. If a new PaCMAP is retained, its embeddings must come from the final featurizer and match benchmark pooling; old coordinates cannot stand in for new model evidence. |
| 22–23 | **Fixed.** Run metadata has a distinct filename; each bootstrap comparison has an independent stable RNG stream. |
| 24–25 | **Deferred to C6.** Historical model-card generators and duplicated pooling logic should be consolidated while making the public upload path representation-neutral. Existing cards must not be reused as five-model cards. This cleanup has no effect on local training or benchmark predictions. |
| 26–31 | **Deferred cleanup.** Duplicate helper/root/constant/audit code and extra artifact copies do not change the frozen data, train/eval recipe or scores. Broad refactoring during the five-run campaign would increase migration risk without resolving a launch gate. |
| 32 | **Fixed.** The tokenizer regex comment now states its accepted legacy bare `K` and `H`. |
| 33 | **Deferred.** The ChEMBL source corpus is immutable for this revision, so preparation speed, example TSV balance and import-time dotenv are outside G1–G7. The uncounted exception should be classified if preparing a new corpus; it is not a reason to silently rerun the frozen one. |
| 34 | **Partially fixed.** Generated README uses the supplied repo ID, corrects the canonical-SMILES overlap explanation, and uses ChEMBL CC BY-SA 3.0. The current frozen source has no test split; a future configurable split upload should generate that text dynamically. |
| 35 | **Partially fixed.** The trainer special-ID comment and featurizer over-context docstring were corrected. Remaining historical usage paths are not used by the explicit G1–G7 commands and can be refreshed with the final documentation pass. |
| 36–38 | **Fixed.** ModernBERT config is pinned locally with upstream provenance and hash; tokenization statistics use full lengths and fewer passes; APE merged-token display count derives from the vocabulary. |

The separate four-tokenizer population audit found zero unknown/lossy rows,
zero paired-identity failures after same-version RDKit canonicalization, and
a maximum of 349 tokens, fixing the common context at 384. Its first direct
string comparison had flagged 164 older-RDKit canonical SMILES spellings;
these represented the same isomeric molecules. See
`docs/revision_factorial_v1_handoff.md` for the input hashes and the retained
original benchmark cohort.

**Additional integration finding (Helios smoke test):** Passing the explicit
training `--data_files train.parquet` also overrode `--validation_split valid`
in the fallback validation loader. The two-step smoke run on commit `e2dcaa6`
therefore evaluated training molecules and is diagnostic only. The loader now
selects `valid.parquet` explicitly in this case; the frozen 4,096-row
validation-ID path was already independent of `--data_files`. A regression
test covers the fallback path. Rerun pilots after this correction; no pilot or
full-model result from `e2dcaa6` is accepted.
