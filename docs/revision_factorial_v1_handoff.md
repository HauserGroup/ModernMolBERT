# Five-model factorial input handoff

This is the frozen input contract for `MASTER_REVISION_PLAN.md` G1–G7. It
precedes any downstream score. The five runs share ChEMBL source rows, one
training-row permutation, one 4,096-row validation cohort, seed 42 and a
384-token context (including BOS/EOS). The base encoder reuses APE–SELFIES.

## Corpus and shared row IDs

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `data/pretrain/chembl36_selfies/train.parquet` | 2,390,314 | `5ba76a62d62c7dc628e5af4a6707eb05f4e03f79d563fd895484e4ac6fccdc7e` |
| `data/pretrain/chembl36_selfies/valid.parquet` | 24,228 | `2426bc7f1514507ef17901db51e8c8bdf056e04d645b567e081c8ae96b12a19e` |
| `data/pretrain/chembl36_selfies/tokenizer_sample_seed42_2m.npy` | 2,000,000 | `6df2f3be915cb316c27eec56aac6616a00aea72b32fbcdf680b7e05c896eed6a` |
| `data/pretrain/chembl36_selfies/train_order_seed42.npy` | 2,390,314 | `60069ea515283bf5e7fb4ca28196092c41f849a5eb4b1fa157dba6441de126a1` |
| `data/pretrain/chembl36_selfies/validation_rows_seed42_4096.npy` | 4,096 | `8f5b52b43d88a766b2c1e806eaa0225b0d83c839bd882dfb9d1c14a4eb020040` |

The row-ID files are ignored data artifacts and are staged separately on
Helios. The sample is the first 2 million indices in the seed-42 NumPy
permutation of training rows; its ordered-content hash is
`1d983d91659422f2068fa9b1c5dc6579f0e5dcf1bebc6aa1d04b3d69a97b7f40`.
The four tokenizer metadata files record that same sample and source hash.
The training order is the full seed-42 permutation. Validation IDs are the
first 4,096 indices in a separate seed-42 permutation of validation rows.
The trainer checks both files, rejects non-permutations or invalid validation
IDs, and pins their file hashes in `run_identity.json`.

## Tokenizers and full-population lengths

All four were trained on the paired columns from the same sampled molecules.
The prespecified ceiling was 2,000 tokens, with minimum merge frequency
3,000; APE maximum merge pieces were 2 for SELFIES and 6 for SMILES. Actual
vocabularies differ because candidate merges exhausted at different points.
This is a comparison of complete tokenizer/representation configurations,
not a pure segmentation effect at equal vocabulary size.

| Tokenizer | Vocab | File SHA-256 | Train median / p99 / max | Valid median / p99 / max |
|---|---:|---|---:|---:|
| APE–SELFIES | 600 | `8871e9414362c2f1fb313a2dc844a57f077b8e82c7e99195cfb88aa1d741f0f2` | 27 / 56 / 213 | 27 / 55 / 106 |
| APE–SMILES | 1,376 | `a6f40f48409378a8726bac930ecd68d1a57509851388fefa2aeccb6119a98bc5` | 14 / 31 / 229 | 14 / 31 / 122 |
| BPE–SELFIES | 1,690 | `e33ac36a5f4e4907689b02d2b22efbea6612c5e56819d0d20d8d950c1782f7f6` | 14 / 33 / 178 | 14 / 32 / 78 |
| BPE–SMILES | 1,602 | `b08c4285f505cb94ba522209e499c291dcd6e9c66fb4252bde0f89c43d169d5f` | 12 / 29 / 349 | 12 / 29 / 184 |

The full 2,390,314 training and 24,228 validation rows produced zero unknown
rows and zero tokenization round-trip losses for all four tokenizers. Maximum
length is 349, so 384 is the smallest multiple of 64 that retains every
frozen pretraining row without truncation. The audit uses entire populations,
not a sample; its machine-readable result is
`outputs/audit/revision_factorial_v1/tokenizers.json` and should be copied
with the ignored input artifacts. The old biased 588-token APE vocabulary
and 128-token encoder remain historical only.

Every SELFIES decoded to the same canonical isomeric molecule as its paired
SMILES: zero identity failures in both full populations. An initial direct
text comparison flagged 164 training rows because the stored SMILES had been
canonicalized by an older RDKit version; canonicalizing both sides with the
current RDKit resolved all 164. No corpus row was changed or excluded.

## Benchmark cohort

The existing prepared JSON row order and splits are frozen for the 25 paper
tasks. `scripts/migrate_prepared_legacy_cache.py` recreates current-module
joblib files from those JSONs without resplitting. The corrected RDKit guard
is used for future preparation, but that rebuilt cohort changed some TDC
scaffold assignments and would break the planned comparison. The frozen
`data/prepared/` files and imported Praski CSV were checksum-matched after
transfer to Helios. The rebuilt cohort moved about 2,015 CYP1A2 molecules
between train and test although the molecules were identical, because the TDC
scaffold split depends on raw row order.

## Remaining launch gates

G4–G5 now require a clean pinned Helios commit, environment record, exact batch
order/resume proof, five 200-step pilots, and consistent embedding/reload
checks. No 30,000-step run is accepted until those gates pass.
