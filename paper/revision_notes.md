# Notes for the revised manuscript

## TODO: re-run benchmark aggregation with CV head selection (2026-09-28)

**Status: not yet re-run. All benchmark numbers in `main.tex` predate this fix.**

`scripts/paper/build_benchmark_results_frames.py` now picks each dataset × embedder's
downstream head by cross-validation ROC-AUC, not test ROC-AUC (critical review
item 1). The old version kept the head with the highest *test* score.

- `outputs/eval/best_metric_by_dataset_embedder.csv` was built by the old
  version. It has no `selection_metric` column, so `build_paper_results.py`
  refuses to run until it is regenerated.
- The new script raises an error on repeated runs, conflicting run provenance,
  or missing or out-of-range CV scores. Existing results may need cleaning
  before it completes.

Re-run from the repo root, in order:

1. `uv run python scripts/paper/build_benchmark_results_frames.py`
2. `uv run python scripts/paper/build_paper_results.py`
3. `uv run python scripts/paper/make_paper_figures.py`
4. `uv run python scripts/paper/make_appendix_table.py`
5. `uv run python scripts/paper/compute_bootstrap_cis.py`

**Manuscript impact:** every headline and per-task number may change. Check
at least:

- Abstract and Conclusion: MMB-base 77.9.
- Results: the ECFP4/MoLFormer/SELFormer/ChemBERTa-2 means and win counts.
- `tab:main-results`, `tab:pertask`, `tab:bootstrap-cis`, `fig:bootstrap-ci`.
- Ablation deltas: size +0.5, span −0.3.
- Baseline rows are also affected: Praski results are re-selected by CV.

Also align the downstream-head wording. The Table 2 caption generator now says
"logistic regression", but Methods and the Fig 1 caption say "ridge".

## TODO: re-run related-task and paired intervals on corrected results (2026-09-29)

**Status: archived diagnostic only. Re-run after the retrained encoder and the
re-scored baselines exist.**

The task-family bootstrap (code commit `0455b61`) currently runs on the archived
matrix, whose scores used test-selected heads and imported baseline rows. Its
numbers in `tab:bootstrap-cis` and Results are provisional, like every other
archived number. The per-dataset paired intervals cannot run at all yet: they
need prediction archives with test-row identities for every compared model.

After the retrain and baseline re-scoring, from the repo root:

1. `uv run python scripts/paper/build_common_row_benchmark.py ...` (see
   `docs/revision_run.md` §5); it writes `paired_task_differences.csv`.
2. `uv run python scripts/paper/compute_bootstrap_cis.py --matrix <corrected matrix>`
   with the archival caption note removed.

Keep `task_families.yaml` unchanged; the families were fixed before corrected
results existed. **Manuscript impact:** replace the archived family intervals
and family win counts in Results and `tab:bootstrap-cis`, and replace raw
per-dataset win counts with wins whose paired interval excludes zero.

## Hetero-span masking moved to the supplement (2026-09-28)

Hetero-span leaves the main text and all claims. It stays as a labelled
exploratory column in `tab:pertask`, with a disclosure sentence, because preprint
v1 reported it and its checkpoint is public. The section-by-section plan (text,
figures, tables, reworded conclusions, disclosure wording) is in
[`hetero_span_removal_plan.md`](hetero_span_removal_plan.md).

## Pretraining/benchmark structure overlap (measured, 2026-09-28)

**Replaces:** the caveat in `main.tex` (Discussion, "Finally, the ChEMBL 36 corpus
and several benchmark datasets derive from overlapping medicinal-chemistry
sources...", ~l. 1061–1068). There the overlap analysis is listed as future work;
it has now been run. It also addresses critical review item 5
(item map in `paper/MASTER_REVISION_PLAN.md`).

**Source data:** `analysis/pretraining_eval_overlap.csv`, produced by
`uv run python analysis/pretraining_eval_overlap.py`.

### Method (for Methods / Supplement)

- Pretraining set: `data/pretrain/chembl36_selfies/train.parquet` (2,390,314
  molecules; the 1% validation split is excluded because the model never trained on it).
- Benchmark molecules: prepared datasets in `data/prepared/`, the same files the
  embedding step reads. SMILES are RDKit-canonicalized there.
- Standardization policy: a benchmark molecule counts as seen in pretraining if its
  RDKit standard InChIKey matches a ChEMBL `standard_inchi_key`, or if its
  canonical SMILES matches the pretraining `smiles_canonical_clean` string.
  Standard InChIKey merges tautomers and mobile-H forms. Salts and counter-ions
  are not stripped: a salt form and its parent count as different structures.
- Counted over unique canonical SMILES, for the full dataset and for its test split.
- Exact-identity overlap only. No similarity- or analogue-level analysis yet.

### Results (25 paper datasets, ToxCast excluded)

- Pooled over all benchmark molecules: **81.1%** were seen in pretraining
  (test splits: **78.7%**).
- Per-dataset test-split overlap: median 76.0%, range 20.4–100%.
  - ≥ 90% overlap on the test split (7 datasets): Bioavailability_Ma (100%),
    the five CYP*_Veith sets (94–96%) and MUV (91%).
  - < 50% overlap on the test split (6 datasets): BACE (20%), molhiv (24%),
    SARSCoV2_3CLPro_Diamond (26%), AMES (37%), ClinTox (40%), hERG (41%).
- Matching on SMILES strings alone would give 76.0% pooled overlap. The
  difference is mostly tautomer and representation variants, which is why the
  InChIKey criterion is used.

### Suggested framing

- Pretraining is self-supervised on structures only, so **no benchmark labels
  are leaked**. Unlabelled exposure to benchmark inputs is common practice:
  PubChem- and ZINC-based encoders such as ChemBERTa-2 and MoLFormer almost
  certainly contain most MoleculeNet/TDC structures as well.
- State the overlap numbers explicitly. Drop the unsupported claims that
  inflation is "modest" and that relative comparisons are "less affected":
  baseline overlap rates were not measured.
- Qualify generalisation claims. Scaffold splits separate supervised training
  from testing, but the encoder has seen most test structures during pretraining.
  Avoid claiming transfer to unfamiliar chemistry.

### Possible follow-ups (not yet done)

1. Seen/unseen breakdown: score each test set separately on molecules seen vs
   unseen in pretraining, for ModernMolBERT and for the baselines. Unseen molecules
   are not a random subset, so the comparison needs baselines on the same subsets.
2. Nearest-neighbour Tanimoto similarity of test molecules to the pretraining set
   (analogue exposure, not just exact identity).
3. Clean-corpus control (critical review item 5): remove benchmark structures
   from the pretraining corpus, retrain, and re-evaluate.

## Tokenizer extra-symbol injection now off by default (2026-09-28)

**Affects:** `main.tex` Methods, "Post-training symbol injection" (~l. 531–544),
and Appendix tokeniser paragraph (~l. 1276–1283).

- Since commit 060f1b4, `train_ape_tokenizer` force-adds no extra symbols unless
  `--extra_vocab_symbols_path` / `--extra_vocab_selfies_path` is passed, and the
  docs recommend leaving it off: benchmark-derived symbol lists put evaluation-set
  vocabulary into the tokenizer.
- The shipped tokenizer (`tokenizer/chembl36_selfies_2m_ape_max2_min3000.json`)
  and all current checkpoints were built **with** injection: 589 learned tokens
  + 42 benchmark-derived primitives (freq ≥ 10) = 631. The current `main.tex`
  text is accurate for them.
- **If the tokenizer and model are retrained with the new default** for the
  revision:
  - Drop or rewrite the "Post-training symbol injection" paragraph.
  - Appendix: vocabulary becomes 589 tokens (verify from the new metadata), not 631.
  - The "zero unknown-token rate on evaluation molecules" claim no longer holds by
    construction. Re-measure benchmark `<unk>` rate and report it.
  - Re-run all benchmark numbers.
- **If not retrained:** keep the current text, but consider stating explicitly that
  the injected primitives were selected using benchmark molecules. Reviewers may
  read this as eval-set leakage into the vocabulary, even without label leakage.
