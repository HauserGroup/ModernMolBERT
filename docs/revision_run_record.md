# Record: the one-model run `revision_clean_small_v1` (historical)

The first revision attempt trained one small encoder on a corpus-only APE tokenizer
(`tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.*`, 588 tokens). It was superseded
by the five-model factorial campaign (`MASTER_REVISION_PLAN.md` G1–G7, inputs in
[revision_factorial_v1_handoff.md](revision_factorial_v1_handoff.md)). This file keeps only the
facts that are still cited: the frozen corpus, the corpus-only tokenizer, corpus and benchmark
length statistics, and how the run ended. Step-by-step commands and debug-run logs are in git
history (`git log -- docs/revision_run_record.md`). Outputs under `outputs/` and `runs/` are
local and ignored by Git.

## Status

| Step | Outcome |
|---|---|
| Corpus-only tokenizer | Trained 2026-09-28, verified 2026-09-29 |
| Validation and preflight | Passed, including 200-step debug runs with reload (MPS) |
| Full pretraining (30,000 steps, MPS, full precision, code `4b9d1b8`) | Started 2026-09-29 03:11 UTC; stopped at step 16,051. Complete checkpoints at 5,000, 10,000 and 15,000 steps (validation loss 0.4672612 at 15,000); no final model |
| Re-embedding and scoring | Not started |

The trainer could not resume then (`resume_from_checkpoint` was not exposed) and the run was
stopped under the no-retraining scope of the time. The partial checkpoints are diagnostics,
not the submission model.

## Frozen inputs

| Input | Identity |
|---|---|
| Training split | `data/pretrain/chembl36_selfies/train.parquet`, 2,390,314 rows, SHA-256 `5ba76a62d62c7dc628e5af4a6707eb05f4e03f79d563fd895484e4ac6fccdc7e` |
| Validation split | `data/pretrain/chembl36_selfies/valid.parquet`, 24,228 rows, SHA-256 `2426bc7f1514507ef17901db51e8c8bdf056e04d645b567e081c8ae96b12a19e` |
| Data manifest | `data/pretrain/chembl36_selfies/metadata.json`, SHA-256 `7536a95be879e71caa7f6db9721018e2a2cd525a7e702c0bc410e607adaf94ac`; prepared 2026-05-15 with RDKit 2026.03.1; the recorded selfies version `2.1.1` is probably wrong (audit finding R47) |

The corpus is unchanged in the factorial campaign.

## Corpus-only tokenizer

| File | SHA-256 |
|---|---|
| `…_corpus_v1.json` | `c097b8d1295343a77b8d60c278a78c3974b3ff85d5a8395697359acbd6a86796` |
| `…_corpus_v1.metadata.json` | `8f2984cc0d2063ba76fef9d49ca1f2535f19a8226701c858a6c560640c6c2385` |
| `…_corpus_v1_freq.json` | `a72ca960bd6c71950826409f5f4b2c23e3ec5b724cb4e729a0521a2c2754bea0` |

The files are committed byte for byte; the formatting hooks skip `tokenizer/*.json` because
the trainer writes 4-space JSON without a trailing newline and the recorded hashes would
change. Verified independently:

- 588 IDs: 5 special tokens, all 338 primitives in the full training split (6 of them added
  after merge learning, each seen 1–3 times), and 245 merges. No benchmark-derived symbol is
  present (0 of the 42 injected ones).
- `.` (component separator) is a token and seven merges contain it, so disconnected SELFIES
  tokenise losslessly (`[C].[O]` is `[C]`, `.`, `[O]`).
- Versus the published 631-token tokenizer (learned part): 21 tokens only here (`.`, its 7
  merges, 10 rare primitives, 3 merges) and 22 only there (21 merges near the frequency
  threshold, and `[Ar]`, which occurs in no local split; the published vocabulary was
  probably learned from a different corpus revision, which fits the unresolved 2,390,317
  versus 2,390,314 training-molecule difference).
- Limitations: the merge-learning sample is the first 2,000,000 rows of a streaming shuffle
  with a 100,000-row buffer over an ID-ordered file, so it under-represents roughly the newest
  15 % of compounds; `_freq.json` overstates merges (it debits overlapping pair counts) and is
  diagnostic only.

## Length statistics (full population; lengths include BOS/EOS)

| Vocabulary | Split | Rows | With `.` | With unknown | > 128 tokens | Mean / p50 / p95 / p99 / max |
|---|---|---|---|---|---|---|
| corpus_v1 | train | 2,390,314 | 106,299 | 0 | 6 | 27.76 / 27 / 43 / 56 / 213 |
| corpus_v1 | valid | 24,228 | 1,069 | 0 | 0 | 27.74 / 27 / 43 / 56 / 106 |
| published (631) | train | 2,390,314 | 106,299 | 106,310 | 7 | 27.56 / 27 / 43 / 56 / 210 |
| published (631) | valid | 24,228 | 1,069 | 1,069 | 0 | 27.54 / 27 / 43 / 55 / 105 |

The published tokenizer is scored with the current strict parser, which reports dots as
unknown; archived checkpoint code silently dropped them. Sample-based validation
(`validate_tokenizer`, 10,000 rows) reads the head of the ID-ordered file and understates
lengths; use the full-population audits. The preprint's Table 5 lengths (median 24, mean
25.7) are reproduced by the first 110,000 rows of `train.parquet`, i.e. older compounds; the
last 110,000 rows give mean 32.2.

## Benchmark input coverage (25 paper datasets, test splits)

| Vocabulary | Test rows | Conversion failures | With `.` | With unknown | > 128 tokens |
|---|---|---|---|---|---|
| published (631) | 33,562 | 3 | 964 | 1,004 | 12 |
| corpus_v1 | 33,562 | 3 | 964 | **168** | 12 |

The featurizer rejects unknown tokens, so these 168 rows (0.50 %) become failed embeddings and
are reported as coverage, not repaired by adding benchmark symbols. They carry 61 distinct
primitives absent from ChEMBL training (27 were among the 42 injected symbols), mostly metals
(`[Cu-3]`, `[Pt-2]`, `[Sn]`), explicit `[H]` and `[Branch3]`. By dataset: HIV 98, CYP2D6 12,
CYP2C9 10, SIDER 8, CYP2C19 7, Tox21 7, ClinTox 5, hERG-Karim 5, CYP1A2 4, CYP3A4 4, BACE 2,
CYP3A4-substrate 2 and one each in CYP2C9-substrate, CYP2D6-substrate, 3CLPro and hERG. The
factorial tokenizers are audited separately in the handoff.

## Findings that shaped the campaign

- **Data order.** The archived loader streams the ID-ordered file through a 100,000-row
  buffer, so passes follow registration order; the archived runs made about 3.2 passes with
  loss drops at pass boundaries. `--global_train_shuffle` (and later the frozen order file)
  replaced it. A 200-step run with global shuffle reached validation loss 3.3161 in 259 s on
  MPS; these debug losses are readiness checks only.
- **Special-token IDs.** Archived small and base configs inherited ModernBERT's CLS/SEP IDs
  50281/50282, outside their 631-token vocabulary. The public revisions `d734b00` and
  `9b13a24` correct them to 0/2 and add MASK/UNK 4/3, with byte-identical weights and
  tokenizer files.
- **Precision.** The one-model run used full precision on MPS; the archived runs used bf16.
