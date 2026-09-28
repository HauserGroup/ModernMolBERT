# Notes for the revised manuscript

## Hetero-span masking dropped (2026-09-28)

The hetero-span ablation is removed from the next edition. The section-by-section
removal plan (text, figures, tables, reworded conclusions) is in
[`hetero_span_removal_plan.md`](hetero_span_removal_plan.md).

## Pretraining/benchmark structure overlap (measured, 2026-09-28)

**Replaces:** the caveat in `main.tex` (Discussion, "Finally, the ChEMBL 36 corpus
and several benchmark datasets derive from overlapping medicinal-chemistry
sources...", ~l. 1061–1068). There the overlap analysis is listed as future work;
it has now been run. It also addresses critical review item 5
(`plans/ModernMolBERT-critical-review.md`).

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
