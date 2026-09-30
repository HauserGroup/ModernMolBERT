# Code audit: robustness, simplification, bugs (2026-09-29)

Audited oldest-touched first. Covered all tracked Python and R in `src/`, `scripts/`,
`analysis/`, `R/` and `tests/`; the vendored benchmark harness was only spot-checked and
`renv/activate.R` was skipped. The audit itself edited no code.

> **Latest status:** the status table below is from the first re-check. The current
> state of findings 1–38 and N1–N8 is in
> [Verification pass 2](#verification-pass-2-after-e2dcaa6-and-de3dd25), and the
> second audit round (vendored harness, paper generators, new campaign code) is
> [Round 2 findings](#round-2-findings-list-only-no-code-edited) (R1–R117, still
> being extended; "Checked and not a problem" entries record verified non-issues).

**Campaign implementation note (30 September):** R1 and R5 were fixed in
`2b23d86`; R2 is corrected in the manuscript master-plan command and the
`scripts/run_revision_factorial_v1.py` launcher (`4d9e2a8`) pins both row-ID
files. R4 is pinned at four workers and 32 × 8 for all five pilots; a traced
200-step interrupted/resumed run consumed exactly the same 51,200 source-row
IDs and produced byte-identical terminal weights and optimizer/RNG state.
R56's ignored outer CV worker limit was fixed in `5058425`. R117 is addressed
by the gated launcher rather than changing historical trainer defaults. R57
is partly addressed by the 25-task coverage and common-row manifests in
`docs/revision_factorial_v1_preflight.md`; all 125 pilot embeddings were
aligned on identical train/validation/test rows. R6 remains open because
checkpoint and CSV cache identity still omit scoring mode and input hashes;
the pilot scoring smoke used fresh outputs with `--no-cache --no-resume`, and
production scoring must not use resume until R6 is closed. R64's fixed 50-row
invalid limit is deferred: it would reject larger HIV/SIDER cohorts even when
common-row coverage and endpoint viability are explicitly checked; the
campaign uses the measured per-task counts and stops if an endpoint becomes
unevaluable. Paper-generator findings R10–R13, R42, R66–R67 and R78 remain
deferred until five final-model score matrices exist, because pilot scores are
excluded from manuscript figures and tables.

## Priority shortlist for the five-model campaign (G1–G7)

Ordered by likely impact on the revised results; IDs refer to the entries below.

1. **Scoring can silently keep stale or wrong-mode results** — R6 (verified), R51,
   R101: resume skips ignore `--missing-labels`; the default mode is the one C2 rules
   out; the common-row builder never checks it.
2. **Results CSV can be overwritten** — R18 (verified), R19.
3. **Manuscript version claim** — R47: selfies was most likely 2.2.0, not 2.1.1.
4. **Training-order semantics and recipe defaults** — R1, R2, R4 (verified), R5, R117: frozen order ignored
   without `--global_train_shuffle`, missing from the plan template, and only
   reproducible for a fixed worker count.
5. **Reproducibility records** — R54–R56, R60 (verified), R89, R109: backend,
   optimizer, git commit, RF seeds and token exposure are not recorded or fixed.
6. **Paper generators not ready for five models** — R10–R13, R42, R66, R67, R78.
7. **Evaluation coverage records** — R57, R32, R64: per-row rejection reasons,
   model identity on embedding reuse, unenforced invalid-embedding limit.
8. **kNN on HIV** — R7: policy enforced only procedurally.
9. **Tests that never run** — R25, R105, R106, R94.
10. **Docs that now contradict the code** — R102.

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

## Verification pass 2 (after `e2dcaa6` and `de3dd25`)

Checked every finding against the committed tree, then fixed the remaining items
that are safe and do not touch the G1–G7 campaign inputs (trainer data path,
tokenizers, featurizer, frozen corpus/row IDs, G7 evaluation policy). Test status
after these fixes: 300 passed, 4 skipped (CI marker subset); `ruff check`,
`ruff format --check` and project `pyright` clean (0 errors, 0 warnings).

### Claimed fixes re-verified

- **Confirmed fixed:** 1 (0 of 81,809 TDC rows dropped), 2 (reuse now fails closed;
  historical caches deferred), 3, 5, 6, 13, 15, 16, 17, 20, 22, 23, 32, 36, 37, 38.
- **4 was not fixed by `e2dcaa6`:** `\detokenize` still rendered `[##Branch1]` /
  `[\C ]` (N6). Fixed in this pass (below).
- **34 remainder is not a bug:** the uploader stages only `train.parquet` and
  `valid.parquet`, so "no test split" is accurate for the uploaded dataset.
- **N1 resolved by design:** the frozen context is 384 tokens, above the 349-token
  population maximum (`docs/revision_factorial_v1_handoff.md`). Failing on an
  over-length row matches G2 ("do not silently truncate"). The trainer default
  `DEFAULT_MAX_SEQ_LENGTH = 128` would still abort mid-run; every launch must pass
  `--max_seq_length`.
- **N4 resolved:** `configs/` is committed. Packaging (`parents[2]`) remains a
  source-checkout assumption.
- **N5 resolved:** the test suite passes.

### Fixed in this pass (uncommitted)

| Finding | Change |
|---|---|
| N2, 8 | `scripts/sweeps/run_sweep.py` writes `<run-name>.train.log` beside the run directory (the trainer requires an empty fresh directory), skips only runs with `final_model/model.safetensors` and `all_results.json` (written after the tokenizer files and final evaluation), and stops with a resume/remove message on an incomplete directory instead of treating it as done. Sweep tokenizer/flag defaults (7) left unchanged: they record the historical sweep. |
| N3 | `selection.json` records `best_global_step` when the best checkpoint is loaded, and the terminal-step check runs before `save_model`. (This edit was swept into commit `de3dd25`.) |
| 4 / N6 | `make_ape_token_table._safe` uses an explicit escape map; verified with `pdflatex` that `[Ring1][#Branch1]` and `[C][\C]` render literally. Test updated; its comment wrongly claimed `str.count` counts overlapping matches (it does not). |
| N7 | Regression keeps 2-D targets for multi-target regression instead of flattening them. |
| N8 | `upload_model.py` stages `run_metadata.json`. The `ape_tokenizer_metadata.json` alias for historical runs stays with the C6 uploader rework (14). |
| 9 | `select_pretraining_run.py` defaults to `best_{strategy}_run.json` with `--masking_strategy` (the names `R/collect_sweep_results.R` reads) and records the `selection_metric` actually used for ranking next to trainer-state `best_metric`. |
| 10 | `R/FigX.R` pivots the collector's `final_eval_*` columns and matches learning rates with `near()`. Not run (R rule: edit only). |
| 11, 3 leftover | `01A_ideal_masking_probability.py` finds the repo root with `find_project_root`, writes figures to `outputs/analysis/sweep/`, embeds at the checkpoint's trained context, and scores fresh and cached heads from the same archive with the same metric. |
| 12 | `fixed_eval_best_models.py` defaults to `valid.parquet` and resolves relative `best_model_checkpoint` against the repo root, then the run directory. Docstring paths corrected. |
| 26 (part) | `hf_upload.py` states the real constraint: `utils` re-exports from it, so it must not import `utils`. |
| 28 (part) | `count_hf_params.py` drops the no-op `HF_HOME`/`TRANSFORMERS_CACHE` juggling. |
| 35 (part) | Usage paths corrected in `arrange_panes.py`, `compute_property_regression.py`, `make_ape_token_table.py`, `compute_bootstrap_cis.py`, `check_tokenized_lengths.py`, `patch_tokenizer_vocab.py`, both PaCMAP scripts and `check_tokenizer_model_compatibility.py`. |
| CI | `tests/test_fixed_eval_best_models.py` was unformatted at HEAD, so CI's `ruff format --check .` failed; reformatted. |

### Straightforward Round 2 fixes (uncommitted on `second-sweep`)

| Finding | Change |
|---|---|
| R1 | `train_selfies_ape_modernbert.py`: `validate_args` checks `--train_order_path` requires `--global_train_shuffle` and `--validation_row_ids_path` requires `--use_validation_split`. Unit test added in `tests/test_training_cli.py`. |
| R5 | `train_selfies_ape_modernbert.py`: `_run_input_hashes` hashes `valid.parquet` whenever `--use_validation_split` is set. |
| R7, R101 | `score.py`: `get_disabled_reason` disables kNN on HIV (`clf_ogbg-molhiv`) as well as MUV structurally; `--missing-labels` default changed from `"observed"` to `"as-negative"`. Unit tests added in `tests/eval/benchmarking_molecular_models/test_score.py`. |
| R8 | `supervised/train.py`: `fit_multioutput_finite_label_model` fails closed (`ValueError`) when all candidate CV scores are non-finite. |
| R18 | `praski_export.py`: `read_results_csv` raises on read/parser errors and only treats empty files or missing files as empty. Unit tests added. |
| R19 | `praski_export.py`: Added `"prepared_data_sha256"` to `PRASKI_COLUMNS` and verified export to results CSV. |
| R21, R44 | Removed dead `datasplit.load_embedding` and `datasplit.get_data`; deleted uncalled `common/utils.py`; removed unused `SystemConfig`, `Dataset.filter_out_problematic_molecules`, and `Embedder`/`SmilesEmbedder`/`GraphEmbedder` classes from `common/types.py`. |
| R23 | `supervised/eval_metrics.py`: `log_predictions` stopped writing legacy `.npy` prediction files; removed unused `_object_array`. |
| R25 / addendum | `tests/conftest.py`: Added `MODERNMOLBERT_TEST_MODEL_DIR` env var and `runs/debug_selfies/final_model` to fixture candidate models; added `ROOT` to `sys.path`. |
| R30 | `scripts/paper/make_dataset_summary.py`: Cleaned up unnecessary `int(str(...))` calls. |
| R34 | Ignored `migration_manifest.json` and `*.manifest.json` in all dataset `glob("*.json")` loops across audit and analysis scripts. |
| R39 | `analysis/sweep/fixed_eval_best_models.py`: Added `mps` support and auto-detection in `select_device` and `--device` CLI choices. Unit test added. |
| R40 | `upload_tokenizer.py`: Switched from `clean_tmp` (`shutil.rmtree`) to `make_staging_dir` from `modernmolbert.hf_upload`. |
| R45 | `supervised/models.py`: Pinned `solver="auto"` for `RIDGE_REG`. |
| R46 | `train_selfies_ape_modernbert.py`: `preview_dataset_and_tokenizer` resolves local Parquet path correctly. |
| R58 | `common/types.py`: `Dataset.labels` matches exact identifier names `{"drug_id", "mol_id", "id", "split"}` instead of substring. Unit test added in `tests/test_benchmarking_molecular_models.py`. |
| R59 | `common/types.py`: Removed dead `EmbeddedDataset.serialize_legacy` and cleaned up `from typing import Literal` fallback. |
| R60 | `supervised/models.py`: Passed `random_state=0` to `RandomForestClassifier` and `RandomForestRegressor`. |
| R62 / addendum | `embed_modernmolbert.py`: Removed `_warn_if_not_best_model`, eliminating all pytest warnings in the suite. |
| R63 | `eval/featurizers/modernmolbert_selfies.py`: Metadata reports `"backend": "modernmolbert"`. Assertion updated in test. |
| R73 | `.pre-commit-config.yaml`: Pinned Ruff hook to tag `v0.15.12` to match `uv.lock` and excluded `configs/` from `pretty-format-json`. |
| R76 | Removed unused `score.checkpoint_exists`, `ModernMolBERTSelfiesFeaturizer.featurize` alias, and `model_cards.TOKENIZER_MAX_LENGTH`. |
| R77 | `train_selfies_ape_modernbert.py`: Moved matmul precision and dynamo static config from import time to `main()`. |
| R83 | Fixed 6 of the 7 sub-items (the `audit_saved_predictions.py` one is still open): `select_pretraining_run.copy_best_model` raises `ValueError` if `final_model` missing; `run_sweep.py --dry-run` does not create directory; `audit_injected_symbols.load_symbols` skips any `"#"` comments; `fixed_eval_best_models.assert_compatible_runs` checks `tokenizer_sha256`; `MolecularMLMCollator._build_token_start_weights` bounds token IDs; `load_chembl_selfies` cleans up error message. |
| R89 | `train_selfies_ape_modernbert.py`: Added `include_num_input_tokens_seen="non_padding"` to `TrainingArguments`. |
| R90 | `analysis/validation/check_hf_tokenizer_matches_local.py`: Defaulted `model_max_length` to 128 and added disconnected `[C].[O]` example. |
| R95 | `README.md`: Added context length guard in quickstart example. |
| R96 | `supervised/eval_metrics.py`: Narrowed `except Exception:` to `except ValueError:` in `get_skfp_roc_auc`. |
| R102 | `docs/evaluation.md`, harness `readme.md`, `docs/upload.md`: Removed obsolete `--max-seq-length 256` and clarified staging rules. |
| R103 | `datasets.yaml`: Removed unused `ranking_metric` from `clf_CYP1A2_Veith` and documented metadata fields. |
| R105 | `tests/test_smoke_training.py`: Added committed factorial tokenizer paths to `_find_existing_tokenizer_vocab` and fixed tensor shape assertion; now runs and passes. |
| R110 | `train_selfies_ape_modernbert.py`: In `read_training_tokenizer`, required `tokenizer_sha256` in metadata. |
| R111 | `CLAUDE.md`: Updated to reference current factorial tokenizers and output locations. |
| R112 | `train_tokenizer.py`: Compared resolved paths for `--corpus_primitive_parquet` and `--aligned_sample_parquet`. |

### Verification pass 3 (2026-09-30): the fixes above, checked before commit

Every row of the two tables above was checked against the code. Test status for
the committed state, run in a clean checkout without other uncommitted work:
CI marker subset passes; `ruff check`, `ruff format --check` and `pyright` clean.

**Confirmed by running the code:**

- R7: the imported Praski table has no kNN rows for exactly `ogbg-molhiv` and
  `ogbg-molmuv`; no other configured dataset name contains `hiv` or `muv`.
- R8: an all-missing multi-output target raises instead of fitting default parameters.
- R18: appending to a corrupt results CSV raises `ParserError` and leaves the file
  byte-identical; header-only and missing files still read as empty.
- R19: `prepared_data_sha256` round-trips through the results CSV; a results file
  written before the column existed still loads (value missing).
- R25: with `MODERNMOLBERT_TEST_MODEL_DIR` set to a released checkpoint, the model
  encoding tests run and pass (8 passed).
- R45, N7: ridge regression fits without the failing first attempt; multi-target
  regression predicts two columns.
- R58: label columns are unchanged for all 25 paper datasets; only ToxCast gains
  the two `APR_HepG2_OxidativeStress_*` endpoints (615 → 617).
- R60: two random-forest fits on the same data give identical CV scores and predictions.
- R105: `test_local_tokenizer_encode_selfies_examples` now runs and passes.
- R21, R44, R59, R62, R76: no remaining reference to any removed name.
- R1, R5, R46, R77, R89, R110 were already committed (`2b23d86`…`4d9e2a8`) and are present.

**Corrected in this pass:**

- R83 was marked fully fixed; one of its seven sub-items is not (see R83).
- R83 (`fixed_eval_best_models`): the manifest stored the tokenizer hash under
  `tokenizer_metadata_sha256`. The key is now `tokenizer_metadata_recorded_sha256`
  and the error message names what is compared.
- R101, R19, R7 left the documents behind: `docs/evaluation.md`, the harness
  `readme.md` and `docs/revision_run.md` still gave `observed` as the default, the
  old column list and "kNN is skipped on MUV". Updated.
- R18: the comment in `procedure.check_if_already_evaluated` still said a corrupt
  file is overwritten. Corrected. Residual: that function still swallows the read
  error, so a head is fitted before the append fails. No data is lost, only time.
- R40: `docs/upload.md` now says the tokenizer staging directory must be empty.
  A dry run leaves it populated, so the next run stops until it is removed.

**Consequences to know about:**

- R45 changes `library_hash` from `dac273c48c206b92` to `7e4a44c97f93c74b`, because
  the hash covers the regression grids. Head checkpoints written before it are
  re-run under `--resume`. No configured dataset is a regression task.
- R60 changes random-forest scores relative to unseeded runs and is **not**
  reflected in `library_hash`, which covers grids only. Do not mix rows scored
  before and after it.
- R101 changed only the CLI default. `eval_procedure`, `eval_embedding`,
  `fit_and_eval_embedding` and `fit_model` still default to `observed`;
  `01A_ideal_masking_probability.py` calls them without the argument.
- R63 changes only the `backend` metadata value; nothing compares it.

**Not verified:** R90 needs a Hub download. With the added `[C].[O]` example the
script is expected to report an `input_ids` mismatch if the published tokenizer
predates component-dot handling; that is the intended detection. Finding 10 (`R/FigX.R`)
was not run.

**Not in this commit:** R111 (`CLAUDE.md` is local). Later edits to
`train_tokenizer.py` and `tests/test_smoke_training.py` beyond R112 and R105
belong to work still in progress and were left uncommitted.

### Still open by decision

7 (historical sweep defaults), 18 (CLI defaults; campaign commands pass explicit
values), 19 (frozen corpus), 21 (historical PaCMAP must stay reproducible), 24–25
and 14 remainder (C6 uploader rework), 27, 29–31, 33, the `rdkit_parse_failed`
label for guard rejections in `chembl36.py` (preparation is frozen), and the
N8 alias for historical runs.

## Round 2 findings (list only; no code edited)

Audit of code added since the first pass and of areas the first pass only
spot-checked (scoring harness, paper generators). Severity is relative to the
G1–G7 campaign.

### Training inputs

- **R1. (Fixed) `--train_order_path` is silently ignored without `--global_train_shuffle`.**
  (Medium) The frozen order is only applied inside the global-shuffle branch of
  `make_train_iterable_dataset`, yet `_run_input_hashes` pins its hash in
  `run_identity.json` either way. A launch that omits `--global_train_shuffle`
  streams in buffer-shuffled order while the manifest claims the frozen order.
  Reject `--train_order_path` (and `--validation_row_ids_path` without
  `--use_validation_split`, already checked) in `validate_args`.
  Fixed: `validate_args` checks `--train_order_path` requires `--global_train_shuffle`
  and `--validation_row_ids_path` requires `--use_validation_split`. Unit test added.
- **R2. The master-plan launch template omits the frozen-row flags.** The G6
  command has neither `--train_order_path` nor `--validation_row_ids_path`, so a
  copy-paste launch falls back to seeded shuffles and the streaming validation
  sample. Seeds keep runs mutually consistent, but the frozen artifacts in the
  handoff would go unused. Update the template (manuscript repo) when pinning G4.
- **R3. Every pass uses the identical molecule order.** With a frozen order no
  reshuffle is applied between passes, so all ~3.21 passes present the same
  batches in the same sequence (only masks differ). This is a defensible design
  but should be stated in Methods; per-pass reshuffling would need per-pass frozen
  orders (modest complexity; not required by G2).
- **R4. "Identical molecule order" also depends on `num_workers` and microbatch
  size.** `to_iterable_dataset(num_shards=64)` shards are split across DataLoader
  workers and each worker yields whole microbatches, so the effective batch
  sequence changes with `--num_workers` or `--per_device_train_batch_size`. Both
  are pinned per run in `run_identity.json`, but G3's "reduce microbatch
  uniformly" must also keep `num_workers` identical across all five runs.
  End-of-pass partial batches (one per worker) also make per-step molecule counts
  slightly uneven; include that in the 7.68M presentation accounting.
- **R5. (Fixed) Validation input is not hashed when `--data_files` is set.**
  `_run_input_hashes` hashes `valid.parquet` only when `--data_files` is absent,
  while `de3dd25` now reads `<dataset_name>/valid.parquet` in exactly that case.
  Hash the validation file whenever `--use_validation_split` is set.
  Fixed: `_run_input_hashes` hashes `valid.parquet` whenever `--use_validation_split` is set.

### Scoring harness (G7)

- **R6. Resumed scoring can silently keep stale rows.** (Medium) Two skip layers
  key only on dataset/embedder/head: the head checkpoint's `version_hash` covers
  the RF and ridge grids but not the kNN grid, `--missing-labels`, `CV_SPLITS` or
  the embedding/prepared-data hash; `eval_procedure` also skips whenever a matching
  row exists in the accumulated results CSV (`--cache` defaults on). Re-scoring
  after regenerating an embedding under the same name, or switching
  `--missing-labels observed` → `as-negative` (required by C2), keeps the old
  rows. The rows record `missing_labels` and `prepared_data_sha256`, so a cheap
  fix is to compare those fields before skipping, and to include kNN and the
  missing-label mode in the version hash.
- **R7. (Fixed) kNN is not disabled on HIV.** `score.get_disabled_reason` disables kNN
  only for MUV; C2 requires "no kNN on HIV or MUV". `build_common_row_benchmark`
  enforces it only when the Praski table is passed (the table has no kNN rows for
  HIV/MUV), so an internal-only selection would include kNN on HIV. One-line policy
  fix, left to the G7 owner.
  Fixed: `score.get_disabled_reason` disables kNN for HIV (`clf_ogbg-molhiv`) as well as MUV.
- **R8. (Fixed) The finite-label multi-output path bypasses the non-finite-CV guard.**
  `fit_multioutput_finite_label_model` returns early; when every candidate's CV is
  NaN it silently fits the estimator's default parameters (not a grid value). Used
  only with `--missing-labels observed` on sparse multi-endpoint tasks, not the C2
  primary path. Fail closed like `fit_model`.
  Fixed: `fit_multioutput_finite_label_model` raises `ValueError` when all candidate CV scores are non-finite, failing closed like `fit_model`.
- **R9. Minor harness latent issues.** `make_short_model_name` truncates names at
  the first `.` (dotted run IDs collide); `EmbeddedDataset.y_np` is a
  `cached_property` that would go stale if read before `remove_failed_embeddings`;
  `fit_multioutput_finite_label_model` ignores `memory_weight`/`n_jobs`.

### Paper generators (C3)

- **R10. `make_paper_figures.py` breaks with five internal models.** (Medium, AFTER)
  `bar_colors` has seven entries, so four baselines plus five `MMB-*` columns raise
  `IndexError`; bar width `0.13` also overlaps at nine bars. Only columns starting
  `MMB-` enter the bars, so new labels must keep that prefix.
- **R11. The task-group figure rejects valid missing tasks.** `make_paper_figures`
  writes NaN cells into the group CSV, and `regen_groupfig.validate_group_distribution_data`
  rejects non-numeric values and requires every model to cover every task. C2/C3
  allow a model/endpoint to be missing (e.g. MUV), so the last figure step would
  raise. The CSV is also rounded to one decimal although C3 asks for unrounded
  plotted values. `model_color` cycles four extra colours, so five new `MMB-*`
  labels would share colours.
- **R12. Table 2 averages over different task sets.** `build_paper_results.py`
  computes group and overall means "over available tasks" and bolds the best mean
  per column, but the caption states the mean across all 25 tasks and the
  `Overall_n` counts are not shown. With any missing cell, models are compared on
  different denominators. Either restrict to common tasks or print `n` and drop
  the "all 25" wording when counts differ.
- **R13. Prespecified internal contrasts are not generated.** `build_paper_results`
  and `make_paper_figures` hard-code the historical small/base/span pairs and
  silently skip them when absent; `compute_bootstrap_cis.py` supports only
  reference-versus-baselines and writes fixed filenames, so the five G7.4
  contrasts need separate invocations that overwrite each other. Known pending in
  G7; noted so the generators are not assumed ready.
- **R14. Historical headline selection uses test scores.** Without
  `--task-matrix`, `build_paper_results.headline()` picks small or base by the
  higher overall test mean. Harmless for the new path (explicit `--reference`),
  but the legacy path should not be used for revised claims.
- **R15. APE token "frequency" is neither pair nor token counts.**
  `make_ape_token_table.count_token_frequencies` uses non-overlapping
  `str.count` on the raw string, which undercounts repeated runs
  (`[C][C][C][C]` → 2) and does not match greedy tokenisation. If the table is
  retained, count tokens from `ape_tokenize` output instead and describe it as
  token frequency.

### New scripts

- **R16. Row-ID dtype conventions differ.** `freeze_training_order.py` saves
  `uint32`; `train_tokenizer.collect_aligned_sample` requires little-endian
  `int64`. Both are validated where read, so this is harmless, but one convention
  would simplify the manifest.
- **R17. Migrated prepared joblibs repeat the pickle fragility of finding 2.**
  `migrate_prepared_legacy_cache.py` re-pickles current-module `Dataset` objects;
  the JSONs remain the source of truth and `load_prepared_dataset` prefers them, so
  only direct `joblib.load` consumers (e.g. `check_tokenized_lengths.py`) depend on
  the pickles.

### Results CSV and harness I/O

- **R18. (Fixed) An unreadable results CSV is replaced by a single row.** (Medium-High)
  `praski_export.read_results_csv` returns an empty frame on *any* read error, and
  `append_result_row` / `delete_result_rows` then write that frame back. A
  truncated or malformed accumulated `results.csv` (for example after an
  interrupted write) therefore loses every other row without warning. Verified:
  writing one row after corrupting a CSV leaves a one-row file. The write is also
  a non-atomic read-modify-write, so two scoring processes sharing
  `--output-csv` race and drop rows. Raise on read errors, write via a temporary
  file plus rename, and use one results CSV per embedder when scoring in parallel.
  Fixed: `read_results_csv` catches only `pd.errors.EmptyDataError` (and non-existent files),
  raising on malformed/corrupt CSVs rather than silently returning an empty frame.
- **R19. (Fixed) `prepared_data_sha256` never reaches the results CSV.** `eval_procedure`
  adds it to the row, but `to_praski_schema` keeps only `PRASKI_COLUMNS`, which
  lacks it. Verified with `append_result_row`. The prediction archives still
  carry the hash (and `build_common_row_benchmark` checks it there), so G7 is not
  blocked, but the CSV silently lacks a field the code intends to record. Add it
  to `PRASKI_COLUMNS`. (This also limits the R6 fix to `missing_labels` until
  then.)
  Fixed: Added `"prepared_data_sha256"` to `PRASKI_COLUMNS` and verified export to results CSV.
- **R20. Scoring silently prefers a legacy embedding JSON over a newer joblib.**
  `procedure.load_embedded_dataset` loads `<embedder>.json` whenever it exists,
  even if `embed_modernmolbert` has since written a fresh `<embedder>.joblib`.
  Latent for the five new names; a stale JSON under a reused name would be scored
  instead of the new embedding.
- **R21. (Fixed - part) Duplicate and dead harness helpers.** `datasplit.load_embedding` is
  unused and duplicates `procedure.load_embedded_dataset`. ROC-AUC is computed by
  `get_skfp_roc_auc`, `multioutput_auroc_score`, `prediction_export`, and
  `build_common_row_benchmark.rank_roc_auc`; they agree today, but
  `prediction_export`'s PR-AUC is trapezoidal while the paper's AP is stepwise.
  `prediction_export` also produces rows without `cv_metric`, so its output
  cannot feed CV head selection; retire it or document it as legacy.
  Fixed: Removed unused `datasplit.load_embedding` and `datasplit.get_data` and unused imports.
- **R22. `evaluate` always computes ROC-AUC.** The test metric ignores the task
  and `dataset_config.metric`, so a regression dataset would crash (continuous
  targets) or be mislabelled. Latent (no regression tasks configured); pair with
  finding 20.
- **R23. (Fixed) Legacy `.npy` prediction files are still written.** `log_predictions`
  writes both `.npy` (object arrays, pickle) and `.npz`; every consumer now reads
  the `.npz`. Dropping the `.npy` removes a pickle artifact and halves prediction
  I/O.
  Fixed: Stopped writing legacy `.npy` prediction files; removed unused `_object_array`.
- **R24. `tdc_admet_solver` ignores its `root`.** It builds `admet_group(path="data/")`
  relative to the current directory. Only matters for a re-download, which the
  frozen cohort avoids.

### Tests

- **R25. (Fixed) Opt-in model tests can never run.** `tests/conftest.py`'s
  `find_existing_minimal_model` looks only for `runs/mps_base_minimal_pubchem10m`,
  `mps_debug`, `mps_base_smoke_*` and `zinc20_debug`; none exist (current runs are
  `chembl36_small_mask_mlm_lr_sweep`, `revision_clean_small_v1`). The checkpoint
  reload/encoding tests therefore always skip, even when opted in. G5 relies on
  reload checks; point the fixture at an environment variable or the pilot run
  directory.
  Fixed: In `tests/conftest.py`, added `MODERNMOLBERT_TEST_MODEL_DIR` env var and
  `runs/debug_selfies/final_model` to candidate model paths.
- **R26. (Resolved by finding 20.)** `test_regression_path_returns_1d_predictions`
  used to pass with every CV score NaN. `fit_model` now raises on a non-finite
  best CV score, so the test covers the guard implicitly; no change needed.

### Audit and summary scripts

- **R27. Coverage audits do not use the featurizer's rejection rule.**
  `scripts/audit_benchmark_inputs.py` counts `<unk>` tokens and length, while
  `ModernMolBERTSelfiesFeaturizer` rejects on any lossy round trip
  (`"".join(tokens) != text`), any special ID in the content, SELFIES conversion
  failure, or over-context length. Predicted coverage from the audit can
  therefore differ from actual embedding coverage for the new tokenizers. C2
  coverage should come from embedding metadata (`failed_source_row_indices`); if
  the audit is kept as a pre-check, call one shared validity helper from both.
- **R28. Audit defaults describe the historical model.** `audit_benchmark_inputs.py`
  defaults to the shipped 631-token vocabulary and `--max-length 128`, and loads it
  with `load_checkpoint_tokenizer` (no SHA check). For G2 artifacts pass the
  factorial tokenizer and 384 explicitly, or use `load_verified_tokenizer`.
- **R29. `pretraining_eval_overlap.py` swallows per-dataset errors.** Any load or
  RDKit error becomes an `error` column entry, and if every dataset failed,
  `sort_values(["overlap_pct", ...])` raises `KeyError` instead of a clear
  message. Any non-dataset JSON in the prepared directory (for example a
  `migration_manifest.json` from `migrate_prepared_legacy_cache.py`) appears as
  an error row. The default output also writes into the source tree
  (`analysis/pretraining_eval_overlap.csv`).
- **R30. (Fixed) Small cleanups.** `make_dataset_summary.py` round-trips integers through
  `int(str(...))` for typing only; the published-head audit and paper generators
  repeat the task-name maps again (see 30).
  Fixed: `scripts/paper/make_dataset_summary.py` cleans up `int(str(...))` conversions.

### Evaluation provenance and duplicate selection logic

- **R31. The benchmark-symbol injection toolchain is now dead weight.**
  `export_benchmark_corpus.py` (reads pickled `.joblib` prepared data),
  `filter_missing_selfies_symbols.py`, `patch_tokenizer_vocab.py` and
  `train_tokenizer --extra_vocab_*` exist only to add benchmark-derived symbols,
  which C2 forbids for every new model. The committed symbol lists document the
  preprint vocabulary; the code can be retired (or confined to one clearly
  historical module) to remove about 700 lines and a leakage-prone path.
  `--require_corpus_only_vocab` already blocks such vocabularies at training time.
- **R32. Embedding reuse checks the data hash but not the model.**
  `embed_modernmolbert.assert_reusable_embedding` accepts an existing
  `<embedder>.joblib` if its `prepared_data_sha256` matches, but the embedding
  metadata records only the model *directory*, not a weights hash. Reusing an
  embedder name after replacing the checkpoint (for example a pilot and a final
  model in the same directory) silently skips re-embedding. Record
  `model.safetensors`/`config.json` SHA-256 in the featurizer metadata and compare
  it on reuse; C2 asks for embedding identity anyway.
- **R33. A second head-selection implementation behaves differently.**
  `compare_praski_tables.select_best_head_per_dataset_embedder` sorts by
  `cv_metric` with the default (unstable) sort and takes `groupby(...).first()`,
  which returns the first non-null value *per column*, so fields can come from
  different rows when any is missing, and ties have no deterministic break.
  `build_benchmark_results_frames.collapse_best_head` (stable sort, head-name
  tie-break, provenance checks) is the one the paper uses. Reuse it here, or mark
  `compare_praski_tables` as a diagnostic that must not feed paper numbers.
  (`rank_within_dataset` also ignores its `group_cols` argument.)

### Prepared-data directory and dataset selection

- **R34. (Fixed) `data/prepared/migration_manifest.json` breaks every `*.json` consumer.**
  (Medium) The migration wrote its manifest into the prepared directory.
  `audit_benchmark_inputs.py`, `audit_injected_symbols.py` and
  `paper/audit_saved_predictions.py` iterate `prepared_dir.glob("*.json")` and call
  `Dataset.deserialize_legacy`, which raises
  `TypeError: Dataset.__init__() got an unexpected keyword argument 'source'` on the
  manifest (verified). `pretraining_eval_overlap.py` records it as an error row.
  Move the manifest beside the directory (or name it `*.manifest.json` and skip it),
  and prefer the dataset config over a directory glob.
  Fixed: Ignored `migration_manifest.json` and `*.manifest.json` across all dataset
  `glob("*.json")` loops in audit and analysis scripts.
- **R35. Scoring defaults to one dataset.** `config/score.yaml` sets
  `datasets: [clf_ogbg-molhiv]`, so `score.py` without `--datasets` scores only
  HIV. Downstream, `build_common_row_benchmark` would leave the other rows missing
  rather than fail loudly. Default to the configured 25 (`all`) or require
  `--datasets`.
- **R36. Audits use different dataset universes.** Config-driven scripts
  (`make_dataset_summary`, `audit_split_overlap`, `audit_task_overlap`, scoring)
  see the 25 configured tasks; glob-driven scripts (R34 list,
  `pretraining_eval_overlap.py`) also include the prepared but unconfigured
  `ogbg-moltoxcast`. Totals such as "33,562 test rows" depend on which universe a
  script used; state it in outputs or select datasets from the config everywhere.

### Leftovers from the first pass (not previously listed)

- **R37. `collect_sweep_results.R` marks best runs as `NA`, not `FALSE`.**
  `is_best_span = run == best$span` yields `NA` for every row when
  `best_span_run.json` is absent (`get_chr` returns `NA_character_`). Use
  `run %in% best$span` (and likewise for standard) so filters on these columns do
  not silently drop rows. With finding 9 fixed, `select_pretraining_run
  --masking_strategy span` now writes the file this script reads.
- **R38. `compute_property_regression.py` silently drops molecules.** It inner-joins
  the 100k PaCMAP sample to `train.parquet` on `chembl_id`, so any sampled
  validation-split molecule disappears without a count, and it fits unscaled
  `Ridge(alpha=1.0)` with no CV over alpha. The script is optional/historical (§4 of
  the plan); if its table is retained, print the joined/dropped counts and state the
  fixed alpha in the caption.
- **R39. (Fixed) `fixed_eval_best_models.py --device` has no `mps`.** Choices are
  `auto|cuda|cpu`, and `auto` never selects MPS, so the historical sweep
  re-evaluation runs on CPU on the Mac where the sweep outputs live.
  Fixed: Added `mps` to `--device` choices and added automatic MPS detection in `select_device`.

### Tokenizer upload (C6)

- **R40. (Fixed) `upload_tokenizer.py` still deletes an arbitrary staging directory.**
  (Safety) `clean_tmp` calls `shutil.rmtree(--staging_dir)` before staging; the
  finding-15 fix covered only `hf_upload.make_staging_dir`, which this script
  does not use. `--staging_dir .` would delete the checkout. Switch it to
  `make_staging_dir` (refuses non-empty directories). Trivial, but it is a new
  finding, so it is listed rather than changed in this pass.
  Fixed: Replaced `clean_tmp` (`shutil.rmtree`) with `make_staging_dir` from `modernmolbert.hf_upload`.
- **R41. The tokenizer card and defaults are fixed to the historical tokenizer.**
  `_write_readme` ignores `repo_id` and writes `model_cards.tokenizer_card()`,
  which hard-codes the 631-token vocabulary, 128 context and repo name;
  `DEFAULT_MODEL_MAX_LENGTH = 128` and the default vocabulary are historical, and
  only APE tokenizers can be staged. Publishing any factorial tokenizer with this
  script would ship a wrong card. Part of the C6 uploader rework (see 14, 24).

### Paper generators (continued)

- **R42. Loss curves overwrite each other across runs.** C3 calls
  `make_loss_curves.py --run-dir` once per completed run, but the script always
  writes `Supplementary_2.pdf` into `--figure-dir` and emits no source data, so
  five calls leave only the last run's figure and no logged values (C6 asks for
  curve values). Name the output after the run (or plot all runs in one figure)
  and write the plotted `(step, loss)` pairs as CSV. It also writes into
  `paper/figures`, the historical snapshot, by default.

### Tests (continued)

- **R43. The frozen-order test checks the dataset, not the batches the model sees.**
  `test_frozen_row_order_is_identical_across_representations` iterates the
  `IterableDataset` in-process, so it proves the source-row order but not the
  effective batch sequence after DataLoader worker sharding (R4) or after resume
  skipping. G5 asks for "the same effective batches" and matching consumed row
  IDs; a small test that iterates `trainer.get_train_dataloader()` with
  `num_workers=2` for two representations (and once more after a simulated resume)
  would cover it without touching the training code.

### Dead or redundant harness code

- **R44. (Fixed) Unused harness modules and classes.** `common/utils.py` (`batch`,
  `cuda_available`, `get_least_utilized_gpu`, `get_device`, with an import-time
  `try/except Exception` that only logs), `Dataset.filter_out_problematic_molecules`,
  the `Embedder`/`SmilesEmbedder`/`GraphEmbedder` classes in `common/types.py`, and
  `datasplit.load_embedding` (R21) have no callers. Removing them shrinks the
  vendored surface without touching splits or metrics.
  Fixed: Removed unused `common/utils.py`, `Dataset.filter_out_problematic_molecules`,
  `SystemConfig`, and `Embedder`/`SmilesEmbedder`/`GraphEmbedder` classes.
- **R45. (Fixed) Regression ridge always fails once before its fallback.** `RIDGE_REG`
  pins `solver="lbfgs"`, which scikit-learn's `Ridge` accepts only with
  `positive=True`, so every regression ridge fit errors and then reruns with the
  `saga` fallback. Latent (no regression tasks); use `solver="auto"` if regression
  is ever added.
  Fixed: Set `clf__solver: ["auto"]` for `RIDGE_REG`.

### Trainer logging

- **R46. (Fixed) The trainer log misreports local Parquet input as Hub streaming.**
  `preview_dataset_and_tokenizer` decides the "Dataset mode" line with
  `find_local_dataset`, which only recognises Arrow directories with
  `dataset_info.json`. The G6 command (`--dataset_name data/pretrain/chembl36_selfies`
  without `--data_files`) therefore logs "streaming from HuggingFace Hub" while
  `get_streaming_dataset` actually reads the local Parquet. Misleading in run logs
  that serve as provenance; reuse `_resolve_dataset_name_as_local_path` for the
  message.
  Fixed: Resolved local dataset path in `preview_dataset_and_tokenizer` so local directories
  are reported accurately.

### Provenance and packaging

- **R47. Recorded selfies version is likely wrong (manuscript-facing).** (High for
  C4 text) `chembl36.collect_preparation_versions` records `module.__version__`.
  The installed selfies distribution is 2.2.0 (`importlib.metadata.version`), but
  `selfies.__version__` still reports `"2.1.1"` in that release. `uv.lock` pinned
  selfies 2.2.0 on 15 May (commit `189789b`), before the corpus was prepared, and
  every dependency group requires `selfies>=2.2.0`. The corpus
  `metadata.json` therefore says `selfies: 2.1.1`, and the plan (§5, C4) cites
  "selfies 2.1.1" in Methods. Confirm which environment prepared the corpus; if it
  was this lockfile, the correct version is 2.2.0. Use
  `importlib.metadata.version()` for every recorded package version (the trainer's
  run metadata should record selfies/rdkit/tokenizers too; it currently records
  only torch and transformers).
- **R48. Base dependencies include notebook and evaluation-prep packages.**
  `jupyterlab`, `ipykernel`, `iprogress`, `tornado`, `aiohttp`, `urllib3`,
  `deepchem`, `ogb` and `pacmap` are core runtime dependencies, although only
  notebooks import `pacmap` and nothing in `src/`, `scripts/` or `tests/` imports
  `deepchem`; the `eval`/`eval-prep` groups repeat several base packages.
  `pyarrow`, imported directly by core modules, arrives only transitively via
  `datasets`. Trimming the base set shortens the Helios `uv sync` and reduces the
  surface of the pinned environment; changing it now would alter `uv.lock`, so do
  it before G4 pins the commit or leave it.

### Notebooks

- **R49. `analysis/benchmark/visualizations.ipynb` selects heads by test score.**
  Every summary in it takes `groupby(["embedder", "dataset"])["test_metric"].max()`,
  the test-maximum rule the plan's §5 audit showed to be optimistic, and it writes
  PNGs into the current directory. It is exploratory and does not feed the paper
  scripts; mark it as historical (or move it to an archive folder) so its win-rate
  and ranking outputs are not reused for revised claims.
- **R50. Committed notebooks keep their outputs despite the nbstripout hook.** All
  three `.ipynb` files contain outputs (47 in the benchmark notebook), so the
  pre-commit `nbstripout` hook was bypassed or predates them. If the PaCMAP outputs
  are kept deliberately as Fig_4 provenance, exclude that notebook in the hook
  config; otherwise strip them. The PaCMAP notebooks also use relative paths
  (`../../outputs/...`) and write to `figures/pacmap`, so they only work when run
  from their own directory.

### Common-row builder (G7), second look

- **R51. The missing-label policy is neither checked nor recorded.** C2 requires
  `--missing-labels as-negative` for every internal model (table compatibility),
  and results rows carry a `missing_labels` column, but
  `build_common_row_benchmark.py` never reads it: `collapse_best_head`'s
  provenance check matches columns containing `split`, `seed`, `checkpoint`, … but
  not `missing_labels`, and the manifest does not record the mode. A result CSV
  scored with the default `observed` mode would be selected and compared silently
  (R6 makes this more likely on resume). Require a single `missing_labels` value
  across archive-backed rows (optionally a CLI-declared expected value) and write
  it to the manifest.
- **R52. Paired per-dataset differences use a different cohort rule from the
  common matrix.** `paired_task_differences` intersects rows over whichever models
  verified for that dataset, while `common_task_matrix` blanks the whole dataset
  unless the fixed cohort is complete. On a dataset where one of the five models
  failed, `paired_task_differences.csv` still reports pairs on a larger row set
  than any other dataset's pairs. Either apply the same fixed-cohort rule or add a
  `cohort_complete` column so the interaction summary (G7.4) can filter.

### Validation curves (G3)

- **R53. Evaluation masks are redrawn at every evaluation.** G3 asks for
  "fixed/independently seeded evaluation masks per tokenizer ... where supported".
  The trainer evaluates the fixed 4,096 molecules through the same
  `MolecularMLMCollator` as training, which draws masks from the global torch RNG,
  so each 5,000-step validation point uses different masks and the curve mixes
  model change with mask noise. The trainer has no option for fixed masks today.
  The lowest-complexity route reuses `fixed_eval_best_models.build_fixed_masked_dataset`:
  pre-mask the validation set once with a seeded collator and give the Trainer an
  eval collator that only pads pre-masked rows. Worth doing only if validation
  curves are reported; G3 already treats validation loss as diagnostic.

### Environment and backend recording (G4)

- **R54. The attention backend is implicit and unrecorded.**
  `build_modernbert_config` switches to `flash_attention_2` whenever `flash_attn`
  imports, with no CLI switch, and neither `run_identity.json` nor
  `run_metadata.json` records the resolved implementation. G4 requires recording
  the backend and using the same one for all five runs; an environment where the
  import succeeds on one run and not another would change kernels silently. It
  would also select FlashAttention on a non-CUDA backend if the package happened to
  be installed. Add an explicit `--attn_implementation` (default `sdpa`), store it
  in the identity manifest, and record GPU name, CUDA/driver, `torch.version.cuda`
  and package versions (see R47) in `run_metadata.json`; currently only
  `platform.platform()`, torch and transformers versions are stored.
- **R55. The optimizer comes from the library default.** The trainer never sets
  `optim` or the Adam betas/epsilon; under the locked transformers they resolve to
  `adamw_torch_fused`, 0.9/0.999, 1e-8, with Trainer's default no-decay groups
  (bias and LayerNorm weights). G3 asks for an explicit AdamW and a recorded
  parameter-group policy. The values match G3 today, but they are pinned only
  indirectly by `uv.lock` and are not in `run_identity.json`. Pass
  `optim="adamw_torch"` (or fused, chosen once in preflight) and the betas/epsilon
  explicitly, and record them.
- **R56. Runs do not record the code revision.** `run_identity.json` pins argument
  values and input hashes (tokenizer, corpus, row IDs, base config, `uv.lock`) but
  not the git commit or a dirty flag, and `run_metadata.json` has neither. G4
  requires a pinned clean commit per run; `build_common_row_benchmark.git_revision`
  already implements the two-line check. Adding `{"commit", "dirty"}` to the
  identity (and refusing a dirty tree for non-debug runs) makes the per-run
  provenance self-contained instead of relying on the launch log.

### Embedding coverage records (C2)

- **R57. Rejections are counted by reason only partially, and never per row.** C2
  asks for per-dataset rejection reasons. `featurize_smiles` counts lossy/unknown
  tokenisation (`n_tokenization_failures`) and over-context inputs (still named
  `n_truncated`, although they are now rejected, not truncated), but empty inputs
  and SELFIES conversion failures are skipped without any counter.
  `embed_dataset` stores only `failed_source_row_indices`, so a failed row's reason
  cannot be recovered. Recording a per-row reason code (`empty`,
  `selfies_encoder`, `lossy_or_unknown`, `over_context`) in the metadata, and
  renaming `n_truncated` to `n_over_context`, would give C2 its table directly.
  (Note also that `embed_dataset` calls `.astype(str)`, so a missing SMILES
  reaches the featurizer as the string `"nan"` rather than `None`.)

### Dataset types (`common/types.py`)

- **R58. (Fixed) `Dataset.labels` drops any column whose name contains "id".** Label
  columns are everything except `smiles`, `graph` and columns whose lower-cased
  name contains `"id"` or `"split"`. Verified on the prepared data: the 25 paper
  datasets are unaffected, but `ogbg-moltoxcast` silently loses
  `APR_HepG2_OxidativeStress_24h_up` and `_72h_up` (the "id" inside "Oxidative"). Any future
  endpoint name with that substring (e.g. "…Lipid…", "…acid…") would vanish
  from training, scoring and the dataset summary alike. Match identifier columns
  exactly (`Drug_ID`, `mol_id`, `id`, `split`) instead of by substring.
  Fixed: `Dataset.labels` matches exact identifier names `{"drug_id", "mol_id", "id", "split"}`.
- **R59. (Fixed) Unused `EmbeddedDataset.serialize_legacy` hard-codes provenance.** It
  writes `pooling="mean"`, `special_tokens_excluded=True` and `max_seq_length=128`
  regardless of the embedding. It has no callers; delete it before anyone uses it
  for the 384-context embeddings. The duplicated `try/except ImportError` around
  `from typing import Literal` is also dead.
  Fixed: Deleted `EmbeddedDataset.serialize_legacy` and cleaned up `Literal` import.

### Downstream heads (`supervised/models.py`)

- **R60. (Fixed) Random-forest heads are unseeded.** (Medium, C6 reproducibility)
  `RandomForestClassifier`/`Regressor` are built without `random_state`, and
  nothing seeds NumPy globally before fitting, so every scoring run gives slightly
  different RF CV scores, selected heads and test scores. The common-row builder
  verifies archives against their own recorded scores, so G7 checks pass, but
  "regenerate the reported comparison" (C6) cannot reproduce RF numbers exactly,
  and a near-tie between RF and logistic CV scores can flip the selected head on a
  rerun. Setting `random_state` (for example 0) changes no grid or metric and keeps
  the Praski protocol.
  Fixed: Passed `random_state=0` to `RandomForestClassifier` and `RandomForestRegressor`.
- **R61. kNN on integer embeddings standardises before a count-Tanimoto metric.**
  `get_knn_distance` chooses `tanimoto_count_distance` for integer embeddings, but
  the pipeline first applies `StandardScaler`, so the metric receives centred
  floats with negative values and the "Tanimoto" distance is meaningless. It is
  also a pure-Python metric (very slow). Only integer (fingerprint) embeddings take
  this path; the paper's ECFP4 numbers are imported, so the five-model campaign is
  unaffected. Drop the scaler for integer inputs if ECFP is ever re-embedded here.

### Embedding CLI

- **R62. (Fixed) The embed CLI nudges toward "best" checkpoints.**
  `_warn_if_not_best_model` warns whenever the model path lacks the substring
  `best` ("Pass runs/best_<name> to use the designated best checkpoint"). Under
  G3 the primary models are terminal-step checkpoints in
  `runs/revision_factorial_v1/<run-id>/seed42/final_model`, so every production
  embedding would emit a warning recommending the opposite policy. Remove it, or
  check `final_model/selection.json` instead of the path string.
  Fixed: Removed `_warn_if_not_best_model` from `embed_modernmolbert.py`.
- **R63. (Fixed) Featurizer metadata labels every model "modernmolbert_selfies".** The
  `backend` field and the class name say SELFIES although the same class now
  embeds SMILES and BPE checkpoints (the `representation` field is correct). Rename
  the metadata value (e.g. `modernmolbert`) so provenance files for the SMILES
  models are not misleading.
  Fixed: `ModernMolBERTSelfiesFeaturizer` metadata records `"backend": "modernmolbert"`.

### Harness configuration

- **R64. `max_invalid_embeddings: 50` is declared but never enforced.**
  `config/embedding/default.yaml` sets it and `EmbeddingConfig` carries it, but no
  code reads it: `embed_dataset` drops any number of failed rows silently apart
  from the metadata list. Either enforce it per dataset (with the plan's "stop and
  investigate if filtering makes an endpoint unevaluable") or delete it so the
  config does not suggest a guard that does not exist. `clock_directory`,
  `svd_directory` and `data_directory` are likewise unused.
- **R65. Download, embed and scoring resolve data paths against the working
  directory.** Nine `os.getcwd()` joins (plus `tdc_admet_solver`'s `"data/"`, R24)
  make every stage write to `./data/...` wherever it is launched, while configs are
  resolved relative to the package. Launching from another directory on Helios
  silently creates a fresh empty `data/` tree and re-downloads or fails with
  "embedding not found". Resolve against `find_project_root()` or require an
  explicit `--data-root`.

### Bootstrap table text

- **R66. Bootstrap captions hard-code the reference model and resample count.**
  `compute_bootstrap_cis._emit_task_latex` always says "\model{}-base" and
  "$B=10{,}000$" regardless of `--reference` and `--n_boot`; the family-table
  caption uses the actual reference name but also fixes the resample count. With the
  plan's reference (small APE–SELFIES) the generated caption would name the wrong
  model. Interpolate both values, as the family branch already does for the name.
- **R67. The per-task appendix caption misstates missing cells.**
  `make_appendix_table.py` explains "--" as "evaluations not yet run", but in the
  revised matrices a missing cell also means an incomplete common-row cohort, an
  undefined ROC-AUC or a rejected endpoint (R11, R12). It also hard-codes "25-task
  benchmark" and prints `--models` labels unescaped inside `\textbf{}` and the
  legend, so a label containing `_` (e.g. a run ID) breaks compilation. Take the
  missing-cell reasons from `common_task_matrix_status.csv` or reword generically,
  and escape labels with the same helper as R15/N6.

### Checkpoint README

- **R68. `final_model/README.md` advertises a "Best checkpoint" for terminal-step
  models.** `write_run_metadata` always adds a "Best checkpoint" section (path,
  metric, step from trainer state) when trainer state exists. Under G3
  (`--no-load_best_model_at_end`) the shipped weights are the step-30,000 weights,
  yet the card names an earlier validation-best checkpoint, and the "Final
  evaluation metrics" come from a fresh evaluation with new random masks (R53). State
  the selection rule and selected step from `selection.json` instead, and label the
  best-validation checkpoint as diagnostic.

### ChEMBL preparation (latent)

- **R69. Deduplication would collapse every row lacking an InChIKey into one.**
  `prepare_chembl36_frame` calls `drop_duplicates(dedupe_column)`, which treats all
  missing keys as equal. The frozen corpus is unaffected (metadata shows
  `rows_after_dedupe == input_rows == 2,854,815`, and no prepared row lacks an
  InChIKey), but a source with missing keys would silently keep only the first such
  molecule. Deduplicate only non-null keys and fall back to the canonical SMILES
  for the rest, mirroring `make_split_key`.

### Featurizer efficiency

- **R70. Each molecule is tokenised three times.** `featurize_smiles` calls
  `tokenize()` (lossless check) and `encode()` (special-ID and length check), then
  `_tokenize_batch` tokenises the accepted strings again for the forward pass. For
  the pure-Python APE tokenizer this triples the dominant CPU cost of embedding
  (the plan's extraction-cost diagnostic measured tokenisation at 9–18 % of the
  total). Keep the validated `content_ids`, add BOS/EOS and pad them directly;
  behaviour is unchanged and the validation and the model input can no longer
  diverge.

### Tokenizer settings (verified, no defect)

- **R71. `max_vocab_size` means different things for APE and BPE.** For APE it
  excludes the five special tokens, for BPE it includes them
  (`train_tokenizer --help`). The four factorial artifacts all record
  `max_vocab_size=2000`, and none reached it (APE 600/1,376; BPE 1,690/1,602,
  stopped by `min_freq_for_merge=3000`), so the difference changed nothing. The
  handoff calls it "the prespecified ceiling"; if Methods quote it, say it was
  non-binding. Verified the rest of the handoff: file SHA-256s match, all four share
  one sample-order hash, no benchmark symbols were injected, and the APE tokenizers
  appended 9 corpus primitives after merge learning.

### Checked and not a problem

- **Frozen-order data path throughput.** `indexed.select(order)` leaves an indices
  mapping, which the `datasets` docs warn slows reads. Measured on the full
  2,390,314-row training Parquet: about 169k rows/s through the frozen-order
  `to_iterable_dataset(num_shards=64)` versus about 203k rows/s after
  `flatten_indices()` (7 s one-off). Training consumes a few thousand molecules per
  second, so no change is warranted.
- **APE merge engine.** `tokenization/ape.py` is covered by a randomised comparison
  against an independent reference implementation
  (`test_ape_engine_matches_independent_reference`); no issue found.
- **BPE and SMILES checkpoint reload.** `load_checkpoint_tokenizer` round trips for
  APE and BPE are covered in `tests/test_tokenizer_variants.py`.

### Continuous integration

- **R72. CI does not guard the working branch or the lockfile.**
  `.github/workflows/ci.yml` runs only on pull requests to `main`, so the many
  commits on `second-sweep` are never tested; the unformatted test file fixed in
  this pass went unnoticed that way. The jobs call `uv sync` without `--locked`
  (G4 requires `--locked`, so CI can pass on a re-resolved environment that Helios
  would reject), and there is no `pyright` step although AGENTS.md and G4 treat it
  as a gate. The lint job also installs the full torch stack just to run Ruff. Add a
  `push` trigger for working branches, `--locked`, a `pyright` step, and run Ruff
  via `uvx`.

### Pre-commit hooks

- **R73. (Fixed) Hook versions and exclusions do not match the pinned tooling.**
  - The pre-commit Ruff hook is `v0.16.9`, while `uv.lock` (used by CI's
    `ruff format --check`) pins Ruff 0.15.12. Formatting can differ between the
    two versions, so a commit formatted by the hook can fail CI or vice versa; pin
    the hook to the locked version (or run Ruff from the project environment via a
    `local` hook).
  - `pretty-format-json --autofix` excludes only `tokenizer/*.json`, yet
    `configs/modernbert_base_config.json` is also hash-pinned
    (`.provenance.json`, checked by the trainer). The file is currently already in
    the hook's format (verified: autofix leaves the SHA unchanged), so this is
    luck rather than protection; add `configs/` to the exclusion. The same applies
    to any JSON manifest whose hash is recorded elsewhere.
  Fixed: Pinned Ruff hook to `0.15.12` and excluded `configs/` from `pretty-format-json`.

### Tracked artefacts without a generator

- **R74. `figures/FigX_sweep_metrics.{pdf,png}` has no generator in the repo.**
  The tracked files date from 21 May; `R/FigX.R` writes `FigX_sweep_all.*`
  instead, and nothing references `FigX_sweep_metrics`. The plan already says
  `FigX.R` is not a verified source for `Fig_3`/`Supplementary_1`; delete the
  orphaned figure or record its origin so it is not mistaken for regenerated
  evidence. (`imgs/mmbert_text.png` is only the README banner.)

### Dead code scan (`vulture --min-confidence 60`, reviewed by hand)

- **R75. Two upload validators are never called.** `upload_model.validate_tokenizer_config`
  (checks `auto_map`, `use_fast=False`, `model_max_length`, absence of
  `tokenizer_class`) and `validate_direct_ape_tokenizer` are defined but unused;
  `upload_model_to_hub` runs only `validate_staged_files` and
  `validate_staged_model`. The tokenizer-config checks therefore never run before a
  public upload. Call them from `upload_model_to_hub` (cheap) or delete them; do not
  leave validators that look active.
- **R76. (Fixed - part) Other unused code confirmed by hand.** `score.checkpoint_exists`,
  `compare_praski_tables.best_head_per_dataset` (alias),
  `ModernMolBERTSelfiesFeaturizer.featurize` (alias), `model_cards.TOKENIZER_MAX_LENGTH`,
  `SystemConfig`, plus the items in R44/R59. Removing them is behaviour-neutral.
  Fixed: Removed unused `score.checkpoint_exists`, `ModernMolBERTSelfiesFeaturizer.featurize`,
  and `model_cards.TOKENIZER_MAX_LENGTH`.
- **R77. (Fixed) Importing the trainer changes global torch state.**
  `train_selfies_ape_modernbert` sets `torch.set_float32_matmul_precision("high")`
  and `torch._dynamo.config.assume_static_by_default = False` at import time, so
  tests or scripts that merely import a helper from it (e.g.
  `tests/test_frozen_molecule_order.py`) run with modified matmul precision. Move
  both into `main()`.
  Fixed: Moved `torch.set_float32_matmul_precision("high")` and dynamo configuration into `main()`.

### Figure scales

- **R78. Paper figures clip data with fixed axis ranges.** `make_paper_figures.paired_panel`
  fixes both axes to 0.45–1.0, so any task scored below 0.45 is silently
  outside the panel, and the group bar chart fixes `ylim(0.55, 0.90)`, so a lower
  group mean disappears and bars that start at 0.55 exaggerate the differences
  between models. Derive limits from the data (with a floor at 0.5 marked as chance
  for ROC-AUC) or annotate points outside the range; C3 asks for figures that match
  the source data exactly.

### Addendum to R62

- **R62 addendum. (Fixed)** All four warnings in the current CI test run are this
  "model-dir does not contain 'best'" warning from
  `test_embed_modernmolbert_cli_skips_existing_and_overwrites`; removing it leaves
  the suite warning-free.
  Fixed: Removing `_warn_if_not_best_model` left the test suite completely warning-free.

### Training-time tokenizer gate

- **R79. The trainer's tokenizer gate samples only the start of the corpus.**
  `_sample_train_partition_sequences` streams the ChEMBL-ID-ordered training
  Parquet through a 100,000-row shuffle buffer and takes the first
  `--tokenizer_validation_samples` (1,000) molecules, so every gate statistic
  (unknown rate, silent loss, truncation) comes from roughly the first 100k source
  rows — the same bias §5 of the plan found in the old "held-out" length sample.
  The full-population audit (`scripts/audit_factorial_tokenizers.py`) now covers
  all rows, so the gate is a sanity check only; either sample through the frozen
  training order (or random row IDs) or state in the log that it is a head-of-file
  smoke check.
  The same head-of-file sampling applies to `validate_tokenizer.py` (`--n 1000`
  through the same streaming loader).

### Hub upload (C6, continued)

- **R80. Re-uploading to an existing repo leaves stale files behind.**
  `hf_upload.push_folder_to_hub` calls `create_repo(exist_ok=True)` and
  `upload_folder` without `delete_patterns`, so files from an earlier upload that the
  new staging folder lacks (an old `selfies_vocab.json`, `collator_config.json`,
  `ape_tokenizer_metadata.json`, a previous README variant) stay in the repository
  next to the new model. If any of the five new checkpoints reuse an existing repo
  name, pass `delete_patterns="*"` (the staging folder is complete by construction)
  or refuse non-empty repos.

### Output locations

- **R81. Figure generators write into the historical `paper/` snapshot by default.**
  `paper/README.md` says the code repo's `paper/` is a historical snapshot, to be
  synchronised only as an intentional release step. Yet `make_paper_figures.py`
  (`--figure-dir paper/figures`), `make_loss_curves.py` (R42) and the installed
  `modernmolbert-regen-groupfig` command (`paper/figures/Fig_task_group_distributions.pdf`)
  default to overwriting it, while the table scripts default to `outputs/eval/paper`.
  Default every generator to `outputs/…` and copy into `paper/` explicitly at
  release.
- **R82. Four paper generators run on import.** `build_paper_results.py`,
  `make_paper_figures.py`, `make_appendix_table.py` and `make_loss_curves.py` parse
  arguments and do their work at module level, so their constants (task groups,
  labels) cannot be imported by other scripts (one reason R30's duplication exists)
  and tests must drive them through subprocesses. Wrapping the body in `main()` is a
  mechanical change that also enables R30.

### Minor robustness leftovers (first-pass notes, low priority)

- **R83. (Fixed, except the `audit_saved_predictions.py` sub-item)** Each is a one-line fix; none affects the five-model campaign.
  - `select_pretraining_run.copy_best_model` raises `TypeError` (`Path(None)`) when
    the winning run has no `final_model`, instead of a clear message.
  - `run_sweep.py --dry-run` still creates the run root directory.
  - `audit_injected_symbols.load_symbols` skips only lines starting with `"# "`,
    while every other symbol reader skips any line starting with `"#"`.
  - `audit_saved_predictions.py` scores every `.npz` with ROC-AUC (a regression
    archive would crash it) and counts datasets without prepared labels as "size
    mismatches".
  - `fixed_eval_best_models.assert_compatible_runs` requires byte-identical
    `tokenizer_metadata.json` across runs; timestamps or paths in that file would
    raise a false "metadata SHA mismatch". Comparing `tokenizer_sha256` suffices.
  - `MolecularMLMCollator._build_token_start_weights` indexes a `vocab_size` tensor
    with every `ids_to_tokens` key and would raise `IndexError` if the mapping held
    an ID ≥ `vocab_size` (hetero-span only).
  - `load_chembl_for_pacmap.load_chembl_selfies` raises a message about the
    `--sample-size` CLI flag from a library function.
  Fixed: six sub-items implemented and verified. Still open: `audit_saved_predictions.py`
  scores every archive with ROC-AUC and labels missing prepared labels a size mismatch.
- **R84. `compare_praski_tables.make_table1_like` selects variants by test score.**
  With `collapse_names=True` (the default) it merges model variants (e.g.
  `ChemBERTa_[10M][MTR]`, `…[77M][MLM]`) by keeping, per dataset, the variant with
  the best *test* metric — the optimistic rule the plan's §5 audit documented.
  Heads inside each variant are CV-selected (R33 caveats aside). Diagnostic only; if
  any of its output reaches the manuscript, collapse by CV score or not at all.

### Benchmark corpus export

- **R85. `export_benchmark_corpus --split` ignores the real splits for most datasets.**
  `iter_smiles_for_split` filters on a `split` *column* of the prepared DataFrame and
  falls back to every row when that column is absent. Verified: 19 of the 25
  configured prepared datasets have no `split` column (the splits live in
  `Dataset.splits`), so `--split test` or `--split train` silently exports all
  molecules for them; the TDC datasets that do have the column label validation rows
  `train`, so `--split valid` returns nothing. The historical
  `benchmark_selfies_symbol_counts.tsv`/injection lists may therefore count
  training-split molecules even if a test-only export was intended. It also reads
  pickled `.joblib` files rather than the frozen JSON. Low impact now that injection
  is retired (R31), but any reuse should select rows via `Dataset.splits`.

### Manuscript-repository copies

- **R86. Untracked copies of paper scripts live in the manuscript folder.** The
  manuscript directory contains untracked `scripts/paper/` copies of six generators
  (`build_paper_results`, `build_benchmark_results_frames`, `audit_saved_predictions`,
  `audit_published_head_selection`, `compute_property_regression`,
  `make_ape_token_table`) and an untracked `tokenizer/alternative/`. Two already
  differ from the code repository (`make_ape_token_table.py` lacks the escape fix
  from this pass). The plan names the code root as the only place to run generators;
  delete the copies (or make them symlinks) so a stale generator cannot rebuild a
  manuscript table.

### Checked and not a problem (frozen inputs)

- **Frozen inputs match the handoff.** Local `train.parquet`, `valid.parquet`,
  `tokenizer_sample_seed42_2m.npy`, `train_order_seed42.npy` and
  `validation_rows_seed42_4096.npy` all have the SHA-256 values recorded in
  `docs/revision_factorial_v1_handoff.md` (verified 29 September).

### Data-loader verification

- **R4 verified empirically.** Using `make_train_iterable_dataset` with a 256-row
  toy corpus and a frozen permutation: with `num_workers=0` the DataLoader yields
  exactly the frozen order; with 2 and with 4 workers it yields two further,
  different sequences of the same molecules. The frozen order therefore defines the
  per-worker shard contents, not the global presentation order; the latter is
  reproducible only for a fixed worker count (and microbatch size). State that in
  the run manifest and Methods, or document the frozen order as "shared shard
  assignment".
- **R87. Training workers require the `fork` start method.** The same experiment
  under macOS's default `spawn` fails with
  `AttributeError: Can't get local object 'make_train_iterable_dataset.<locals>.keep_train'`:
  the filter and map functions are closures, which cannot be pickled for
  `spawn`/`forkserver` workers. Linux on Python 3.13 defaults to `fork`, and MPS/CPU
  runs force `num_workers=0`, so the campaign is unaffected; but Python 3.14 makes
  `forkserver` the Linux default, which would break every multi-worker run. Moving
  the two functions to module level (taking the column/length via
  `functools.partial`) removes the dependency.
- **R88. Expect masks to differ after a resume (G5 test design).** Masking happens
  in the collator inside DataLoader workers, whose torch seeds are drawn from the
  main-process RNG when the iterator is created. On resume, Trainer restores the
  main RNG and skips the consumed batches, so the molecule sequence matches (for the
  same worker count, R4) but the worker seeds, and hence the masks, need not match
  an uninterrupted run. The G5 interrupted-versus-uninterrupted check should compare
  consumed row IDs, step/LR state and loss *trajectories within tolerance*, not
  exact per-step losses; otherwise it will fail for a benign reason.

### Exposure accounting (G6)

- **R89. Presentations are estimated, and tokens are not counted.** G6 acceptance
  asks for "7.68M accounted molecule presentations" and records of "token
  presentations". The trainer writes only `train_samples_streaming =
  max_steps × batch × accumulation × world_size`, an estimate that ignores the
  short end-of-pass batches (R4). Setting
  `TrainingArguments(include_num_input_tokens_seen="non_padding")` makes Trainer log
  `num_input_tokens_seen` (non-padding tokens) at every logging step, at negligible
  cost on one GPU; with the per-step batch sizes in `trainer_state.json` this
  gives both token and molecule accounting without new code paths. This matters
  because equal *molecule* exposure means unequal *token* exposure across
  tokenizers, which the plan asks to report.

### Validation scripts (analysis/validation)

- **R90. `check_hf_tokenizer_matches_local.py` fails with its own defaults.** It
  builds the local tokenizer with `--model-max-length 256` and then requires
  `model_max_length` to equal the Hub tokenizer's, which was published with 128
  (`upload_tokenizer.DEFAULT_MODEL_MAX_LENGTH`, `model_cards.TOKENIZER_MAX_LENGTH`),
  so a default run stops at the first check. It also compares vocabularies and
  encodings only for three dot-free SELFIES, so it cannot detect the historical
  component-dot loss the plan documents. Default to 128 (or read the Hub value) and
  add a disconnected example such as `[C].[O]`.
- **R91. `prediction_export` stamps archives with the current grid hash.** Rows
  exported from `.npz` archives get `library_hash = get_model_version_hash()` at
  export time, so predictions produced under an older grid are labelled as if
  produced by the current one, and the archive's own `prepared_data_sha256` and
  row IDs are ignored. It also raises on any archive whose dataset is not in
  `datasets.yaml` rather than skipping it (e.g. predictions for the unconfigured
  `ogbg-moltoxcast`). Together with R21 (no `cv_metric`, trapezoidal PR-AUC), this
  module no longer fits the CV/common-row pipeline; retire it or mark its output
  as a test-score export without provenance.
  (`data/predictions/` currently contains `ogbg-moltoxcast`, so a default
  `prediction_export` run raises on it.)
- **R92. `plot_predictions` shows AUROC values that differ from the benchmark's.**
  For multi-endpoint datasets it pools all endpoints into one micro-averaged curve
  and prints that AUROC in the legend, whereas the benchmark reports the mean of
  per-endpoint (macro) AUROCs; the PR legend uses trapezoidal AUPRC (R21). It also
  plots every head, including non-selected ones, and `_load` leaves each `.npz`
  open. Diagnostic only; if a plot is ever used as a figure, label the averaging
  and restrict to CV-selected heads.

### APE tokenizer files

- **R93. A representation-specific vocab alias silently overrides `vocab.json`.**
  `APEPreTrainedTokenizer` declares `vocab.json`, `selfies_vocab.json` and
  `smiles_vocab.json`, and `_select_vocab_file` prefers the representation-specific
  file whenever it is present. `copy_tokenizer_artifacts` and `upload_model` write
  the alias as a byte copy, so they agree today, but any directory where
  `vocab.json` is later replaced (a patched or re-saved vocabulary) keeps loading
  the stale alias without warning. Either load only `vocab.json`, or fail when both
  exist and differ. The alias adds nothing a single file does not provide; keep it
  only for already-published repos.

### Checked and not a problem (continued)

- **Type-checking `scripts/` and `analysis/`.** Running `pyright` over the two
  directories (outside the configured `include`) gives 52 errors; reviewed by
  hand, all are pandas/NumPy stub unions (`Series | ndarray | …`) rather than
  defects. Adding these folders to the type-checked surface would cost many casts
  (which the conventions discourage) for no bug-finding gain; leave them out.

### Test gaps for the riskier paths

- **R94. Several behaviours above have no test.** No test covers: the scoring
  resume skip and its version hash (R6); `read_results_csv`/`append_result_row` on
  an unreadable file (R18); column retention in `to_praski_schema` (R19); a mixed
  `missing_labels` input to `build_common_row_benchmark` (R51); or
  `--train_order_path` without `--global_train_shuffle` (R1). The fixes made in this
  pass to `scripts/sweeps/run_sweep.py` (completion predicate, log location) and to
  `select_pretraining_run.py` (strategy-specific default filename,
  `selection_metric` field) are also untested; each needs one small `tmp_path` test
  when those files are next touched.

### README example

- **R95. The README embedding example has no context-length check.** It rejects
  lossy and unknown tokenisation and pools over non-special tokens, matching the
  featurizer, but it calls the tokenizer without truncation or a length test. The
  published checkpoints were trained at 128 positions, so a longer molecule is
  embedded at positions the model never saw, which the featurizer now rejects
  (finding 3). Add
  `if inputs["input_ids"].shape[1] > model.config.max_position_embeddings: raise …`
  so users follow the same policy.
  The model-card quickstarts generated by `upload_model.build_readme` and
  `model_cards.card` lack the same length check and, unlike the README, also the
  lossless-tokenisation and unknown-token checks; fold this into the single card
  generator proposed in finding 24.

### Broad exception handling

- **R96. The test metric silently switches definition on any error.**
  `eval_metrics.get_skfp_roc_auc` calls `roc_auc_score` inside `try/except Exception`
  and falls back to `multioutput_auroc_score` (per-endpoint, skipping single-class
  endpoints). The intended trigger is "an endpoint has one class", but any error —
  a shape bug, a dtype problem — also lands in the fallback, which may then return
  a score computed under a different rule without a trace. Catch only the
  single-class `ValueError`, or call the per-endpoint function directly for
  multi-output targets and let other errors surface. A scan of the remaining
  `except Exception` handlers found only the cases already listed (SELFIES
  conversion in the featurizer, R57; ChEMBL chunk rows, 33; canonicalisation in
  `data_v2`, intended).

### Hard-coded path scan

- **Hard-coded paths.** No tracked `.py`, `.R`, `.toml` or `.yaml` file contains an
  absolute user path. The only one in the repository is inside a committed
  notebook output (`pacmap_visualization.ipynb` shows `/Users/<user>/…/.venv`),
  another reason to strip outputs (R50).

### Duplicated helpers (systematic count)

- **R97. Same-name helpers defined in several files.** A count of top-level
  function names finds, beyond findings 26/30:
  - `load_tokenizer` in `tokenization/load.py`, `analysis/tokenization/check_tokenized_lengths.py`
    and `analysis/sweep/fixed_eval_best_models.py` (the first copy skips the SHA check
    in `load_verified_tokenizer`; the second reimplements `load_checkpoint_tokenizer`);
  - `_apply_paper_style` in `compute_bootstrap_cis.py`, `compute_property_regression.py`
    and `make_ape_token_table.py` (identical rcParams; figures drift if one changes);
  - `sha256_file` in `fixed_eval_best_models.py` and
    `audit_pretraining_representation_overlap.py` (duplicates `file_sha256`);
  - `load_symbols` in `audit_injected_symbols.py` and `patch_tokenizer_vocab.py`,
    with different comment rules (R83);
  - `as_list` in `score.py` and `common/config.py`; `resolve_metadata_path` in
    `check_tokenized_lengths.py` and `upload_tokenizer.py`.
  A shared `scripts/paper/_style.py` and reuse of the package's `file_sha256` and
  `load_verified_tokenizer` remove most of these with no behaviour change except the
  added SHA check.

### Win/tie/loss counts

- **R98. Ties are counted by exact float equality.** `compute_bootstrap_cis` and
  `build_paper_results` count a tie only when two unrounded ROC-AUCs are exactly
  equal, which essentially never happens, while tables print the values at one
  decimal of ×100. A reader can therefore see equal printed values reported as a
  win or loss. Either define ties at the printed precision (|Δ| < 0.0005) or state
  in captions that W/T/L use unrounded scores. No change in method is needed, only a
  consistent definition.

### Hub token handling

- **R99. Two token-resolution rules.** The trainer's `--hf_login` reads only
  `HF_TOKEN`, while the upload scripts use `hf_upload.resolve_hf_token`
  (`HF_TOKEN_ORG`, then `HF_TOKEN`). With only the organisation token set, upload
  works but the trainer's login fails. Reuse `resolve_hf_token`. (`.env` is
  git-ignored and a `detect-private-key` hook is configured; no secrets are tracked.)
  Since the base configuration is now pinned locally and the corpus is local, the
  campaign trainer needs no Hub access at all; dropping `--hf_login` (and the
  module-level `huggingface_hub.login` import) from the trainer is simpler than
  aligning it.

### Simplification candidates (streamlining)

- **R100. The Arrow `load_from_disk` input path is unused.** `--data_dir` in
  `train_tokenizer.py`, `validate_tokenizer.py` and the trainer, together with
  `utils.find_local_dataset`, `_local_dataset_matches_request` (loose substring
  name matching, finding 17) and the Arrow branch of `get_streaming_dataset`,
  serve saved Arrow datasets with a `dataset_info.json`. None exist under `data/`;
  every current recipe streams local Parquet or the Hub, and the trainer even
  has to guard against the auto-detection in `corpus_only_training_parquet`.
  Removing the Arrow path deletes about 80 lines and the last name-guessing
  behaviour; `--data_files` already covers explicit files.

### Scoring defaults versus the documented recipes

- **R101. The default missing-label mode is the one C2 forbids.** `score.py`
  defaults to `--missing-labels observed`; C2 requires `as-negative` for every
  internal model. `docs/revision_run.md` passes `as-negative` explicitly, but the
  generic recipes in `docs/evaluation.md` (and `config/score.yaml`) do not, so a run
  that follows the main evaluation guide is silently incompatible with the imported
  table and, with R6/R51, can be mixed into the comparison unnoticed. Make the mode a
  required argument (or default to `as-negative`), which costs nothing and removes
  the trap.
- **R7 addendum.** The revision recipe avoids kNN on HIV/MUV procedurally, by
  scoring those two datasets in a separate invocation with `--heads rf ridge`.
  That works but depends on the operator remembering it; the one-line
  `get_disabled_reason` change in R7 makes the policy structural.

### Documentation drift caused by code changes

- **R102. Embedding recipes now fail against the featurizer's new context rule.**
  `docs/evaluation.md` (example and option table: "`--max-seq-length` default
  `256`, truncation length") and the harness `readme.md` (three examples) pass
  `--max-seq-length 256`. Since finding 3 the featurizer rejects any value above the
  checkpoint's `max_position_embeddings` (128 for all existing models, 384 for the
  factorial runs) and no longer truncates, so these commands raise for every current
  checkpoint and describe behaviour that no longer exists. Drop the flag from the
  examples (the default now follows the model) and update the option table.
  `docs/revision_run.md` already uses 128.
  Similarly, `docs/upload.md` describes `--keep_staging_dir` as "keep staged files at
  this path", but since finding 15 the path must be empty or absent, and
  `docs/revision_run_record.md`'s logs already live outside run directories, which
  is the pattern the fresh-run check (N2) now requires everywhere.

### Addenda and empirical confirmations

- **R25 addendum.** `docs/tests.md` tells users to create the debug run at
  `runs/debug_selfies`, which is not among the fixture's candidate directories, so
  following the documented recipe still leaves the model tests skipped. Adding
  `runs/debug_selfies/final_model` to the candidate list (or reading an env var) is
  the one-line fix.
- **R6 verified empirically.** On a toy two-endpoint dataset with missing labels,
  `eval_procedure(..., missing_labels="observed")` followed by the same call with
  `missing_labels="as-negative"` (default `override=False`, i.e. `--cache` on)
  leaves exactly one row in the results CSV, recorded as `observed`: the
  `as-negative` run was skipped silently. This is the path C2's required re-score
  would take on any results file that already holds default-mode rows.
- **R60 verified empirically.** Three identical `fit_model(..., model_head="rf")`
  calls on the same 200-row toy data returned CV ROC-AUCs 0.6077, 0.6062 and
  0.6057. Differences of this size can reorder RF against logistic regression in
  CV head selection on small benchmark tasks.

### Dataset configuration

- **R103. Descriptive fields in `datasets.yaml` are unused and partly stale.**
  `n_tasks`, `pct_positive` and a lone `ranking_metric` (only on `clf_CYP1A2_Veith`)
  are read by no code; `n_samples` is read only by `make_dataset_summary.py` for
  comparison. Checked against the prepared data: all 25 entries match except the
  three differences the plan already records (CYP2C19 12,665 vs 12,663, HIV 41,084
  vs 41,120, Tox21 7,831 vs 7,823). Remove `ranking_metric` (it implies a choice
  the code never makes) and label the rest as source-publication metadata, so
  nobody reads them as the prepared cohort's counts.

### Test markers

- **R104. The CI marker filter excludes markers no test carries.** CI deselects
  `model` and `cuda`, but no test is marked with either; model-dependent tests
  (`test_model_encoding.py`, checkpoint reload) rely on the `existing_minimal_model`
  fixture skipping instead, which never finds a model (R25). Conversely `network`
  is used (one test) but not deselected in CI. Without `--strict-markers`, a marker
  with a misspelt name would pass silently. Mark the model-dependent tests
  `model`, add `network` to the CI filter (or rely on its skip), and enable
  `addopts = "--strict-markers"`.

### Ignore rules (checked)

- **Ignore rules.** `git status --ignored` shows no ignored files under `src/`,
  `scripts/`, `tests/`, `configs/`, `analysis/`, `R/`, `docs/` or `tokenizer/`
  apart from caches, so `.gitignore` hides no source.

### Skipped tests

- **R105. Both CI skips are caused by stale file names, not by opt-in markers.**
  The two skipped tests in the CI subset are `test_model_encoding.py` (R25: no
  fixture candidate exists) and `test_smoke_training.py::test_local_tokenizer_encode_selfies_examples`,
  a fast, unmarked encode check that looks only for
  `tokenizer/selfies_symbol_tokenizer.json`, `selfies_ape_tokenizer.json` and
  `selfies_ape_tokenizer_1m.json`, none of which exist. The committed
  `tokenizer/revision_factorial_v1/*.json` (and the corpus-only vocabulary) would let
  it run in CI with no extra data. Point it at a committed tokenizer so the check
  actually executes.
- **R106. The smoke checks are APE–SELFIES-only.** The opt-in smoke test encodes
  only SELFIES examples and reads `tok.vocabulary`, an `APEPreTrainedTokenizer`
  attribute, so on a BPE checkpoint it fails with `AttributeError` and on a SMILES
  checkpoint every example is unknown. G5 asks for reload/encode checks on all five
  pilots; parametrise the check by the checkpoint's representation (via
  `load_checkpoint_tokenizer`) and use `get_vocab()`/`unk_token_id`, which both
  tokenizer classes provide.

### Tokenizer directory layout

- **R107. Defaults still point at historical vocabularies next to the frozen set.**
  `tokenizer/` holds the historical 631-token injected vocabulary, the 588-token
  corpus-only one and an older APE–SMILES vocabulary beside
  `revision_factorial_v1/`; several defaults (`run_sweep.py`, `audit_benchmark_inputs.py`,
  `upload_tokenizer.py`, `check_hf_tokenizer_matches_local.py`) point at the 631-token
  file. Hash pinning in `run_identity.json` prevents a silent mix-up during
  training, but a misplaced default in an audit or upload would not be caught.
  Moving superseded vocabularies under `tokenizer/historical/` (as was done for
  `alternative/`) and removing tokenizer defaults from campaign-facing scripts makes
  the choice explicit.

### Vendored-harness provenance

- **R108. The vendored harness records no upstream revision or local diff.** The
  project rule is that splits and metrics "must stay unchanged to remain comparable
  with the Praski tables", yet neither the harness `readme.md` nor the docs name the
  upstream repository commit it was copied from, and the local changes are many:
  the SMILES guard (finding 1), the TDC scaffold-split reimplementation,
  missing-label modes, finite-label multi-output fitting, regression scoring
  (finding 20), row-ID provenance and prediction archives. A reviewer cannot tell
  which of these alter scores. A short `VENDORED.md` with the upstream commit
  (the plan's §5 already identifies public commit `17d2aa1` for the imported CSV)
  and a list of behaviour-changing local modifications would make the comparability
  claim checkable.

### Verification of this pass's fixes

- **N7 fix and R45 verified.** `fit_model(task="regression")` on a two-target toy
  problem now fits ridge, RF and kNN heads with finite R² and `(n, 2)` predictions.
  The same run logs `'lbfgs' solver can be used only when positive=True` before the
  ridge fallback succeeds, confirming R45 (every regression ridge fit fails once).
- **N2/8 fix verified by dry run.** With a scratch run root, `run_sweep.py --dry-run`
  skipped a run whose `final_model/model.safetensors` and `all_results.json` exist, planned the missing
  one, and, once an incomplete directory (only `train.log`) was added, stopped with
  the resume-or-remove message. (A unit test is still missing; R94.)
- **Finding 9 fix verified.** On three fake runs, `select_pretraining_run
  --masking_strategy span` wrote `best_span_run.json` for the lower-loss span run
  and recorded `selection_metric: 0.3` next to trainer-state `best_metric: 0.9`,
  which shows why the old record (best_metric only) could misstate the ranking
  value. The console line still says "wrote best_run.json" for any file name
  (cosmetic).
- **Follow-up on this pass's finding-12 fix.** `fixed_eval_best_models.py` resolves a
  relative `best_model_checkpoint` with `find_project_root()`, which starts from the
  working directory; launched outside the repository it raises `FileNotFoundError`
  before the run-directory fallback is tried. `01A` anchors the same helper on
  `Path(__file__)`. Fixed in this pass (my own regression): the call now passes
  `Path(__file__)`.

### Reproduction commands

- **R109. Artefact metadata records a fixed command string, not the invocation.**
  `train_tokenizer.py` writes `"creation_command": "python -m modernmolbert.train_tokenizer"`
  and `chembl36.py` a similar constant; the factorial handoff lists hashes and
  settings but not the four exact `train_tokenizer` command lines. The individual
  settings are in the metadata, so the commands can be reconstructed, but C6 asks
  for exact reproduction commands. Recording `sys.argv` (and the git commit, R56) in
  both metadata files is a two-line change that makes each artefact
  self-describing.

### Tokenizer integrity check

- **R110. A tokenizer without a recorded hash is accepted for training.**
  `load_verified_tokenizer` fails on a hash *mismatch* but only warns when the
  metadata has no `tokenizer_sha256`; the trainer uses it as its sole integrity
  check. Every current tokenizer records the hash, so requiring it in the trainer
  (keeping the warning for historical audit scripts) costs nothing and closes the
  gap where hand-edited metadata could drop the field.

### Agent instructions (local, git-ignored)

- **R111. `CLAUDE.md` describes the historical pipeline as current.** It states
  "The production vocab has 631 tokens and includes the 42 injected symbols" and
  that `scripts/paper` "writes `paper/tables` and `paper/figures`". Under the plan
  the production inputs are the four `revision_factorial_v1` tokenizers (no
  injection, 384 context), and `paper/` is a historical snapshot (R81). Because
  coding agents treat this file as authoritative, update those two lines so they do
  not steer work back to the 631-token vocabulary or into `paper/`.

### Tokenizer training CLI

- **R112. The aligned-sample check compares paths textually.** `validate_args`
  requires `--corpus_primitive_parquet` to equal `--aligned_sample_parquet` with
  `!=` on `Path` objects, so the same file given once relative and once absolute (or
  via a symlink) is rejected as a different file. Compare `.resolve()` paths, or
  better the file hashes that are recorded anyway.

### Interrupted writes

- **R113. Evaluation artefacts are written in place, not atomically.** Only the
  ChEMBL chunk and shard writers use a temporary file plus `rename`. The embedding
  `joblib.dump` (`embed_modernmolbert.py`), the prediction `.npy`/`.npz`
  (`log_predictions`), the prepared-data `joblib`/JSON (`download.py`) and the
  results CSV (R18) are written directly, so an interrupted job leaves a truncated
  file under the final name. For embeddings the next run then fails inside
  `assert_reusable_embedding` with a raw `EOFError`/`UnpicklingError` (it only maps
  `ModuleNotFoundError`/`AttributeError` to the "regenerate with --overwrite"
  message); a truncated `.npz` makes `build_common_row_benchmark` crash rather than
  report a status. Writing to `<name>.tmp` and `os.replace` costs two lines per
  writer. (`assert_reusable_embedding` also loads the whole embedding to read its
  metadata; `mmap_mode="r"` avoids that.)

### G7.1 integration point (pending in the plan)

- **R114. Per-model failure removal makes training rows and CV folds differ.**
  `embed_dataset` calls `EmbeddedDataset.remove_failed_embeddings`, so each model's
  train/valid rows exclude only its own failures; `GridSearchCV(cv=5)` then builds
  unshuffled folds from those differing row sets. G7.1 requires one eligibility
  intersection for training, validation and test across the five models. The
  least-invasive place is a scoring-time filter: read every model's
  `failed_source_row_indices` for the dataset, drop their union from each
  `EmbeddedDataset` before `fit_and_eval_embedding`, and record the union in the
  predictions/manifest. The plan already lists this as open; noted here only to
  pin the code location, not as new scope.

### Measured test coverage

- **R115. The campaign's gatekeeping code is largely untested.** Line coverage of
  `src/modernmolbert` under the CI test subset is 58 % (`pytest-cov`, measured
  30 September). The low spots are exactly the code that guards G1–G7:
  - `validate_tokenizer.py` 0 % — the tokenizer gate the plan requires before
    training;
  - `train_selfies_ape_modernbert.py` 37 % — untested are `validate_args`,
    `validate_tokenizer_for_training` (unknown/silent-loss/truncation gate),
    `write_run_metadata` and the whole `main` (selection record, `max_steps`
    check, `TrainingArguments`);
  - `supervised/procedure.py` 25 % and `score.py` 43 % — the result-row skip logic
    and the head-checkpoint resume logic behind R6 are not executed by any test;
  - `upload_model.py` 26 %, `model_cards.py` 0 % (C6), `data/prepare_chembl36_selfies.py`
    0 % (frozen, acceptable).
  Pure functions such as `validate_args`, `validate_tokenizer_for_training` (with a
  toy tokenizer and a tiny Parquet), `head_checkpoint_is_success` and
  `check_if_already_evaluated` can be covered with small `tmp_path` tests; a
  two-step CPU run (`--debug --max_steps 2` on a small Parquet, marked `smoke`)
  would exercise `main` end to end, as the opt-in smoke test partly does. This is where test effort pays off most before G5.

### Complexity hotspots (radon cyclomatic complexity)

- **R116. A few functions carry most of the branching; two are worth simplifying.**
  `radon cc` grades `eval_metrics._normalize_auc_scores` E (36), and
  `make_eval_dataset`, `validate_args` and `main` in the trainer, `score.main`,
  `export_benchmark_corpus.main` and `supervised/train.fit_model` D (22–29).
  - `_normalize_auc_scores` guesses the orientation of lists, object arrays, 2-D
    and 3-D arrays by comparing shapes with `n_outputs`/`n_samples`; a square
    `(n, n)` input is ambiguous, and every metric path depends on it. Normalising
    once where predictions are produced (`fit_and_eval_embedding` knows whether the
    model is multi-output) and storing a fixed `(n_samples, n_outputs)` positive-class
    matrix would let the metrics drop the heuristics. Moderate change, high value
    because every reported ROC-AUC flows through it.
  - `make_eval_dataset` mixes three sources (frozen row IDs, validation split,
    hash bucket) and two row formats; splitting the frozen-ID branch into its own
    function (it already returns early) would make the campaign path readable and
    testable (R115).
  The remaining D-graded CLIs (`score.main`, `export_benchmark_corpus.main`) are
  long but linear; refactoring them is not worth the risk now.

### Dependency declarations (deptry)

- **R48 refinement.** A `deptry src` run (its `dotenv`/`yaml`/`sklearn` reports are
  import-name mapping false positives) adds two concrete points to R48:
  - `joblib` is imported directly by six harness modules and `pyarrow` by core
    modules, but neither is declared; both arrive transitively (via scikit-learn and
    datasets). Declare them, since a transitive change would break imports.
  - `aiohttp`, `tornado` and `urllib3` are declared but never imported. Commit
    `046f0e0` ("silence dependabot") added them as minimum-version floors for
    transitive packages;
    move them to `[tool.uv] constraint-dependencies` so they constrain resolution
    without becoming runtime dependencies of the package. `deepchem`, `pacmap`,
    `jupyterlab`, `ipykernel` and `iprogress` are genuinely unused by `src/`.

### Trainer defaults versus the G3 recipe

- **R117. Fourteen trainer defaults differ from the frozen G3 recipe.** Parsing the
  trainer with only the two required flags and comparing against G3:
  `max_steps` 150,000 (G3 30,000), `learning_rate` 1e-4 (4e-4), `warmup_steps`
  1,000 (1,500), `mlm_probability` 0.30 (0.15), `load_best_model_at_end` True
  (False), `eval_size` 100,000 (4,096), `seed` 13 (42), microbatch 128 × accumulation
  2 (32 × 8), `global_train_shuffle` off (on), `require_corpus_only_vocab` off (on),
  `unk_rate_threshold` 0.001 (0), `max_seq_length` 128 via `None` (384),
  `report_to` none (tensorboard); plus `--train_order_path` and
  `--validation_row_ids_path` unset (R2). The G6 template passes all of these today,
  but a single omitted flag silently changes the recipe, and `run_identity.json`
  only records what was run, not whether it matches G3. Rather than changing
  historical defaults, commit a small launcher (or a JSON recipe file read by the
  trainer) that sets every G3 value and fails on unset manifest fields, as G6
  already asks ("a production launcher should refuse unset manifest fields"). This
  subsumes R2 and part of 18.

### Checked and not a problem (scans)

- **Runtime `assert`s.** The 13 `assert` statements in package and script code are
  type-narrowing invariants (for example `assert proc.stdout is not None`), not
  input validation, so running under `python -O` would not remove a safety check.
- **Test randomness.** Every test that exercises random masking or sampling seeds
  its RNG (`torch.manual_seed`, `default_rng`, `random.Random`); no unseeded random
  call was found in `tests/`.

- **Execution and deserialisation.** No `shell=True`, `eval`/`exec` of strings,
  `yaml.load` or `allow_pickle=True` in tracked Python. Pickle-based loads are
  `joblib.load` of locally produced prepared/embedded data and
  `torch_load_with_legacy_ogb_defaults`, which forces `weights_only=False` while
  reading downloaded OGB files; the latter only runs on a re-download, which the
  frozen cohort avoids. `upload_model` executes the staged `tokenization_ape.py`,
  which it copied from the run or the repository itself.

- **Documented commands parse.** Every `train_selfies_ape_modernbert`,
  `train_tokenizer`, `validate_tokenizer`, `score.py`, `embed_modernmolbert`,
  `build_common_row_benchmark`, `make_dataset_summary` and `audit_split_overlap`
  command in `README.md`, `docs/*.md` and the harness `readme.md` was run through
  the real argument parser: no unknown or missing flags. (Parsing succeeds for the
  `--max-seq-length 256` examples; they fail only at run time, R102.)

- **Extra Ruff rule families.** Running `PLW`, `PLE`, `RUF005` and `RUF012` on
  top of the configured `E,F,UP,B,SIM` gives 21 hits: 15 intentional
  loop-variable reassignments (`PLW2901`, e.g. `group = group.copy()`), 5
  list-concatenation style hints and the Hugging Face convention
  `model_input_names = [...]` as a class attribute. None is a defect; adding these
  rules would be churn.

- **Copy-paste volume is small.** `pylint duplicate-code` (≥10 identical lines,
  imports ignored) finds only four blocks: shared upload CLI options
  (`upload_dataset`/`upload_tokenizer`), the pooling snippet in
  `model_cards`/`upload_model` (finding 25), and data-source/tokenizer argument
  definitions shared by `train_tokenizer`, `validate_tokenizer` and the trainer.
  The duplication worth removing is semantic (R97, findings 24–26, 30), not textual.

- **On-the-fly tokenisation will not starve the GPU.** Single-core
  `encode_sequence` throughput on 5,000 real validation molecules: APE–SELFIES
  ≈13.7k, APE–SMILES ≈20.4k, BPE–SELFIES ≈20.4k, BPE–SMILES ≈41.8k molecules/s. At
  256 molecules per optimiser step even a single worker supplies >50 steps/s, far
  above plausible GPU step rates, so pre-tokenisation (allowed by G6) is not
  needed for throughput.
- **No test pins a flagged behaviour.** Searching the tests for the defaults and
  behaviours flagged above (kNN only disabled for MUV, `observed` mode, the "best"
  path warning, `.npy` side files, `max_invalid_embeddings`) finds only config
  fixtures that repeat the unused `max_invalid_embeddings: 50`; none of the
  recommended fixes would require rewriting an existing assertion.

### Self-corrections to this pass's fixes

- **Self-correction to the N2/8 fix.** The first version treated
  `final_model/model.safetensors` alone as "complete", but the trainer writes the
  weights before the tokenizer files, metrics and run metadata, so a crash in that
  window would have been skipped as finished. The predicate now also requires
  `all_results.json`, which old and new trainers write after the final evaluation;
  re-verified by dry run (weights only → "incomplete"; weights plus
  `all_results.json` → skipped).
