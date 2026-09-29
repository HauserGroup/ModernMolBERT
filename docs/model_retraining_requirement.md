# Exact model training still required for the submission revision

Status: 29 September 2026. No model training is running. This document defines
the remaining encoder experiment; it is not a request to launch it.

## Decision

**Complete one corpus-only ModernMolBERT-small masked-language-model run, then
re-embed and re-evaluate the 25 paper datasets.** The required encoder is the
34,127,436-parameter, 8-layer, 512-hidden, 8-head ModernBERT preset with a
2,048-unit feed-forward layer, a 128-token input cap, global attention every
third layer, and a 128-token local-attention setting. Train it from random
initialisation on SELFIES with standard independent token masking. This is a
new submission checkpoint, not a fine-tune of the published small/base weights.

No ModernMolBERT-base retrain, span/hetero-span retrain, multi-seed sweep, or
new molecular pretraining corpus is required for the paper's present claim:
a released SELFIES encoder compared as a frozen representation. A matched
comparator would become necessary only if the paper restores a causal claim
about APE, SELFIES, or the ModernBERT backbone; that would be a separate,
explicitly justified experiment.

## Already complete: the tokenizer and preflight

Use the committed training-only APE vocabulary
`tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json` (SHA-256
`c097b8d1295343a77b8d60c278a78c3974b3ff85d5a8395697359acbd6a86796`).
It has 588 IDs: 5 special tokens, all 338 primitives in the full local training
split, and 245 merges. It preserves disconnected-component dots and contains
none of the 42 benchmark-derived symbols appended to the published tokenizer.
The tokenizer need not be fitted again unless the frozen corpus or tokenisation
policy changes. Its merge-learning sample was biased toward earlier ChEMBL IDs;
that known limitation is documented in [revision_run_record.md](revision_run_record.md).

The 200-step MPS preflight with global row shuffle passed, including a saved
model reload, finite logits, lossless disconnected SELFIES, and valid special
IDs. It is a software readiness check, not the submission model.

## Frozen input and recipe for the one encoder

| Item | Required value |
|---|---|
| Train split | `data/pretrain/chembl36_selfies/train.parquet`; 2,390,314 rows; SHA-256 `5ba76a62d62c7dc628e5af4a6707eb05f4e03f79d563fd895484e4ac6fccdc7e` |
| Validation split | `data/pretrain/chembl36_selfies/valid.parquet`; 24,228 rows; SHA-256 `2426bc7f1514507ef17901db51e8c8bdf056e04d645b567e081c8ae96b12a19e` |
| Encoder code | Training path in code commit `4b9d1b8`; subsequent paper-script commit `2fc2001` did not change the trainer |
| Architecture and tokenizer | `--model_size small`, 588-token corpus-only vocabulary, BOS/CLS 0, PAD 1, EOS/SEP 2, 128-token maximum |
| Objective | Standard MLM, mask probability 0.15; no benchmark-derived token injection |
| Schedule | 30,000 optimiser steps, learning rate `4e-4`, cosine decay, 1,500 warmup steps, weight decay 0.01, seed 42 |
| Batch and order | 128 molecules per device × 2 accumulation on one device; `--global_train_shuffle` over the frozen Parquet before buffer shuffle |
| Validation and selection | 4,096 validation examples; evaluate and save every 5,000 steps; choose lowest validation loss with `load_best_model_at_end` |
| Precision in interrupted run | MPS, full precision (`--no-bf16`); record any hardware or precision change before continuing |

The full command is in [revision_run.md](revision_run.md) §3. Its output name is
`runs/revision_clean_small_v1`; do not overwrite the published checkpoints.

## Interrupted run and what remains

The full run began at 03:11 UTC on 29 September 2026 and was stopped at step
16,051/30,000 under the current instruction to proceed without model
retraining. Its last complete checkpoint is
`runs/revision_clean_small_v1/checkpoint-15000` (model SHA-256
`e00d049df7246972748112034127aabf274d42ca340f9ed50cc1ab0ba341773f`).
That checkpoint includes optimiser, scheduler, RNG and trainer-state files;
its validation loss was 0.4672612. The 5,000- and 10,000-step checkpoints
also remain. Work after step 15,000 was not saved. There is **no final model**
and no submission benchmark from this run.

The current CLI calls `Trainer.train()` without exposing
`resume_from_checkpoint`. Before using the 15,000-step checkpoint to finish
the run, implement and verify checkpoint resumption, including model,
optimiser, scheduler, RNG, and the position/order of the globally shuffled
iterable data. If that exact continuation cannot be shown, start a fresh
30,000-step run with the frozen recipe. Do not treat a 15,000-step checkpoint
or a new 15,000-step run as equivalent to the planned 30,000-step experiment.
Keep the partial checkpoints and log as labelled diagnostics.

## Evidence required after the encoder is complete

1. A `final_model/` selected by validation loss, with the 588-token vocabulary
   and valid saved token IDs, plus a successful reload and finite-logit check.
2. A run record with the exact code revision, input/tokenizer hashes,
   hyperparameters, hardware, precision, elapsed time, checkpoint selected,
   and validation metrics. Distinguish this recipe from the published model.
3. New embeddings and downstream scores for all 25 paper datasets, recording
   conversion/unknown/truncation failures and prepared-row IDs. The 168
   benchmark test rows with primitives absent from ChEMBL training are an
   input-coverage warning, not a reason to add benchmark symbols.
4. Select downstream RF, L2-logistic (historically named `ridge`), and kNN
   heads using training-side CV from a common candidate set per dataset;
   evaluate the selected head on held-out test rows. Head fitting is downstream
   evaluation, not encoder retraining.
5. Verify baseline row identity and compare models on shared evaluable test
   rows. Use the prepared-row split-overlap audit for a paired sensitivity that
   excludes shared InChIKey/stereo-insensitive matches while retaining the
   configured splits for the primary analysis. Rebuild the manuscript's numeric
   tables, intervals, plots, and claims from the same versioned result matrix.

Until these checks pass, the revised manuscript must label existing scores as
archival and avoid treating the partial checkpoint as the final model.
