# Run record: `revision_clean_small_v1`

This records the frozen inputs and the check results for the clean revision run
in [revision_run.md](revision_run.md). Update it at the end of each step. The
step outputs under `outputs/` and `runs/` are local and ignored by Git; hashes
below identify them.

| Step | Status |
|---|---|
| 1. Corpus-only tokenizer | Trained 2026-09-28 21:48 UTC; verified 2026-09-29 (below) |
| 2. Validate and preflight | Passed 2026-09-29 (below) |
| 3. Full encoder pretraining | Not started; decide the open points below first |
| 4. Re-embed and re-score | Not started |
| 5. CV selection and common rows | Not started |

## Frozen inputs

| Input | Identity |
|---|---|
| Training split | `data/pretrain/chembl36_selfies/train.parquet`, 2,390,314 rows, SHA-256 `5ba76a62d62c7dc628e5af4a6707eb05f4e03f79d563fd895484e4ac6fccdc7e` |
| Validation split | `data/pretrain/chembl36_selfies/valid.parquet`, 24,228 rows, SHA-256 `2426bc7f1514507ef17901db51e8c8bdf056e04d645b567e081c8ae96b12a19e` |
| Data manifest | `data/pretrain/chembl36_selfies/metadata.json`, SHA-256 `7536a95be879e71caa7f6db9721018e2a2cd525a7e702c0bc410e607adaf94ac`; prepared 2026-05-15 with RDKit 2026.03.1 and selfies 2.1.1 |
| Tokenizer code | Commit `c5852c5` (HEAD at training time; tokenizer code unchanged since) |
| Lockfile | `uv.lock`, SHA-256 `6d01c03db898583d8bba44c172d63bbcb9d463b11196f4db4e69de454bebaae1` |
| Environment at verification | Python 3.13.5, datasets 5.0.1, pyarrow 24.0.0, selfies 2.2.0, RDKit 2026.3.6, torch 2.12.1, transformers 5.17.0 |

The tokenizer metadata does not record the code commit or package versions, so
they are recorded here.

The environment's selfies and RDKit versions differ from the data-preparation
versions. Pretraining reads SELFIES strings from Parquet, so only
SMILES-to-SELFIES conversion of benchmark inputs uses the current selfies
version.

## Step 1: corpus-only tokenizer

Command as in [revision_run.md](revision_run.md) §1. Files:

| File | SHA-256 |
|---|---|
| `tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json` | `c097b8d1295343a77b8d60c278a78c3974b3ff85d5a8395697359acbd6a86796` |
| `tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json` | `8f2984cc0d2063ba76fef9d49ca1f2535f19a8226701c858a6c560640c6c2385` |
| `tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1_freq.json` | `a72ca960bd6c71950826409f5f4b2c23e3ec5b724cb4e729a0521a2c2754bea0` |

These are committed byte-for-byte. The trainer writes 4-space JSON without a
trailing newline, which the formatting hooks would rewrite, breaking the
recorded SHA-256. `.pre-commit-config.yaml` therefore excludes
`tokenizer/*.json` from those hooks (commit `a9cf0e1`).

### Metadata check

Every recorded value matches an independent check:

| Field | Recorded | Independent check |
|---|---|---|
| `tokenizer_sha256` | `c097b8d1…` | SHA-256 of the vocabulary file |
| `extra_vocab_symbols_requested` / `_added` | 0 / 0 | None of the 42 benchmark-derived symbols (`tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt`) is in the vocabulary |
| `corpus_primitive_scan.sha256` | `5ba76a62…` | SHA-256 of `train.parquet` |
| `corpus_primitive_scan.n_rows` | 2,390,314 | Row count of `train.parquet` and the data manifest |
| `corpus_primitive_scan.n_distinct_primitives` | 338 | Regex scan of every training string (`\[[^\[\]]*\]` or `.`): 338 distinct primitives, no string lost content, all 338 in the vocabulary |
| `corpus_primitive_scan.n_added_after_merge_learning` | 6 | The last six IDs (582–587) are `[68Ga+3]`, `[=C+1]`, `[=SH0]`, `[CaH2]`, `[Cs]`, `[OH3+1]`, each seen 1–3 times in the training split |
| `vocab_size` | 588 | 5 special + 338 single primitives + 245 merges; IDs contiguous |
| `special_ids` | BOS 0, PAD 1, EOS 2, UNK 3, MASK 4 | Same as the shipped tokenizer |
| `max_merge_pieces`, `min_freq_for_merge`, `max_vocab_size`, `tokenizer_train_size`, `seed` | 2, 3000, 2000, 2,000,000, 42 | Match the recipe; merging stopped on `min_freq_for_merge` |

The merge-learning sample was read from `train.parquet` only. No local Arrow
dataset under `data/` matches `chembl36_selfies`, so the loader took the
local-Parquet `train` split. `valid.parquet` was not read.

### Contents

- **Component separator.** `.` is a token (ID 24). Seven merges include it:
  `.[Cl]`, `[Cl].`, `.[Na+1]`, `.[Br-1]`, `.[Cl-1]`, `.[I-1]` and `[Br].`.
  Tokenisation is lossless for disconnected molecules: `[C].[O]` gives `[C]`,
  `.`, `[O]`, distinct from `[C][O]`.
- **Difference from the shipped 631-token tokenizer** (learned part, excluding
  its 42 injected symbols):
  - 21 tokens appear only here:
    - `.` and its 7 merges;
    - 10 rare primitives that the shipped vocabulary lacked, such as
      `[135I]`, `[Al-3]` and `[Se-1]`;
    - 3 other merges: `[NH1][N]`, `[I][=C]` and `[O-1][=C]`.
  - 22 tokens appear only in the shipped vocabulary: 21 merges near the
    frequency threshold, such as `[C][F]`, `[S][=O]` and `[=C][=O]`, and the
    primitive `[Ar]`.
- **Shipped-tokenizer provenance clue.** The shipped `_freq.json` records one
  `[Ar]` occurrence in its training sample. `[Ar]` occurs in no local split
  (`chembl36_selfies` train/valid, `chembl36_selfies_old`, the 10k subset). So
  the shipped vocabulary was learned from a different corpus revision than the
  current `train.parquet`. This fits, but does not prove, the unresolved
  difference between the preprint's 2,390,317 training molecules and the local
  2,390,314.

### Known limitations

- **Merge sample is not uniform.** It is the first 2,000,000 rows of a
  streaming shuffle with a 100,000-row buffer. `train.parquet` is ordered by
  ChEMBL ID, so the sample under-represents roughly the newest 15% of compounds. The
  exact sample also depends on the `datasets` streaming implementation. The
  vocabulary SHA-256 above, not the command, identifies the artifact.
- **`_freq.json` is diagnostic only and overstates merges.** It debits pair
  counts that include overlapping occurrences, so `[C]` reads 0. Token IDs and
  merges do not depend on these values.

## Step 2: validate and preflight

### Tokenizer validation (`validate_tokenizer`, n = 10,000, threshold 0)

| Split | Unknown rate | Truncation @128 | Mean / p50 / p95 / p99 length |
|---|---|---|---|
| train | 0 | 0 | 25.89 / 24 / 43 / 57 |
| valid | 0 | 0 | 27.70 / 27 / 43 / 56 |

The train sample is biased. `validate_tokenizer` draws through the same
100,000-row buffer, so on the ChEMBL-ID-ordered training file it samples from
the start: older, shorter compounds. Use the full-population audit below for
reported statistics.

### Full-population input audit (`scripts/audit_selfies_inputs.py`)

Lengths include BOS/EOS. The shipped vocabulary is scored with the current
strict parser, which reports dots as unknown; archived checkpoint code instead
silently dropped them.

| Vocabulary | Split | Rows | With `.` | With unknown | > 128 tokens | Mean / p50 / p95 / p99 / max |
|---|---|---|---|---|---|---|
| corpus_v1 | train | 2,390,314 | 106,299 | 0 | 6 | 27.76 / 27 / 43 / 56 / 213 |
| corpus_v1 | valid | 24,228 | 1,069 | 0 | 0 | 27.74 / 27 / 43 / 56 / 106 |
| shipped (631) | train | 2,390,314 | 106,299 | 106,310 | 7 | 27.56 / 27 / 43 / 56 / 210 |
| shipped (631) | valid | 24,228 | 1,069 | 1,069 | 0 | 27.54 / 27 / 43 / 55 / 105 |

The preprint's Table 5 figures (median 24, mean 25.7, p95 43, p99 56, on
"10,000 held-out" molecules) are reproduced by the first 110,000 rows of
`train.parquet` with the shipped vocabulary: mean 25.7, median 24, p95 43,
p99 57. The last 110,000 rows give mean 32.2 and median 32. Those preprint
figures describe older ChEMBL compounds, not the corpus.

### Benchmark inputs (`scripts/audit_benchmark_inputs.py`)

Output: `outputs/audit/revision_clean_small_v1_benchmark_inputs.csv`, 61 split
rows. All 25 configured datasets are present; ToxCast is extra.

Test splits of the 25 paper datasets:

| Vocabulary | Test rows | Conversion failures | With `.` | With unknown | > 128 tokens |
|---|---|---|---|---|---|
| shipped (`source_data/benchmark_input_audit.csv`) | 33,562 | 3 | 964 | 1,004 | 12 |
| corpus_v1 | 33,562 | 3 | 964 | **168** | 12 |

The featurizer rejects unknown tokens, so these 168 test rows (0.50%) will
become failed embeddings in the clean run and must be reported as coverage.
They carry 61 distinct primitives absent from ChEMBL training, 27 of which were
among the 42 injected symbols. Most are metals (`[Cu-3]` in 20 rows, `[Pt-2]`
11, `[Sn]` 11, `[Pt]` 9, `[Cu]` 8), explicit `[H]` (8) and `[Branch3]` (8).

By dataset: HIV 98, CYP2D6 12, CYP2C9 10, SIDER 8, CYP2C19 7, Tox21 7, ClinTox
5, hERG-Karim 5, CYP1A2 4, CYP3A4 4, BACE 2, CYP3A4-substrate 2, and one each
in CYP2C9-substrate, CYP2D6-substrate, 3CLPro and hERG.

### Debug pretraining and reload

- **Command:** as in [revision_run.md](revision_run.md) §2 (`--debug`,
  `--require_corpus_only_vocab`, `--no-bf16`).
- **Environment:** MPS backend, batch 128 × 2 accumulation, 200 steps. The
  command omits `--learning_rate` and `--seed`, so the defaults applied
  (1e-4, seed 13).
- **Gate and runtime:** the corpus-only gate accepted the tokenizer. The run
  took 376 s.
- **Validation loss:** 5.753, 5.032, 4.497 and 4.179 at steps 50–200; the
  final evaluation gave 4.163. Training loss 5.06; 34,127,436 parameters.
- **Reload check (`docs/tests.md` §3):** passed.
  - `AutoModelForMaskedLM` and the `ape_tokenizer/` bundle load.
  - Logits are finite, with shape (1, 4, 588).
  - The bundled vocabulary is identical to `corpus_v1`.
  - `[C].[O]` and `[Na+1].[Cl-1]` tokenise losslessly without unknowns.
- **Log:** `outputs/audit/revision_clean_small_v1/debug_train.log`.

## Open points before step 3

1. **Pretraining data order.** Training streams the ChEMBL-ID-ordered
   `train.parquet` through a 100,000-row shuffle buffer. Each pass therefore
   runs roughly in registration order; the archived runs made about 3.2 passes
   (30,000 × 256 / 2,390,314). The archived small run's training loss drops
   at each pass boundary (±600 steps), more than the within-pass trend:
   0.619 → 0.580, 0.452 → 0.439, 0.377 → 0.354.
   - A global shuffle needs a pre-shuffled training file with a new tokenizer
     scan hash, or a loader change.
   - Decide before the full run and record the choice here.
2. **Special-token IDs in the model config.** The config inherits ModernBERT's
   `cls_token_id` 50281 and `sep_token_id` 50282, outside the 588-token
   vocabulary. transformers warns on load. The archived released small config
   has the same values. Set them to valid IDs or `None` before release.
3. **Precision and hardware.** The guide uses `--no-bf16`; the archived runs
   used bf16. Record the actual choice.
4. **Benchmark coverage.** 168 test rows will fail embedding. Report per-model
   coverage from `build_common_row_benchmark.py` rather than adding benchmark
   symbols.
