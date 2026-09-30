# Provenance simplification: implementation checklist

30 September 2026. This expands the [decision plan](provenance_simplification_plan.md)
into concrete edits and acceptance checks for the five-model revision campaign.
It does not change the scientific design: one frozen ChEMBL corpus, four
tokenizers, five seed-42 models, 25 prepared benchmark tasks, CV-selected heads,
common test rows, an imported baseline table, figures, and model uploads.

## Target state

| Record | Written when | Owns | Referenced by |
| --- | --- | --- | --- |
| `configs/revision_factorial_v1.json` | Before staging; committed | Five run recipes, shared training arguments, frozen input expectations, scoring policy | Stage and launch scripts |
| `outputs/revision_factorial_v1/campaign_manifest.json` | Once per clean staged checkout | Code commit, lockfile, corpus/order/tokenizer hashes, 25 prepared JSON hashes, imported baseline hash, shared environment | Each run record and the evaluation record |
| `runs/revision_factorial_v1/<run>/seed42/run_identity.json` | At run start; completed after training | Exact run arguments, campaign-manifest hash, seed, tokenizer choice, final step, selected checkpoint, terminal weights hash and final metrics | Embedding, upload, result interpretation |
| `outputs/revision_factorial_v1/evaluation_manifest.json` | Once after five final embedding sets exist | Common row/label/split identities, model/embedding identities, coverage, CV and missing-label policy | Scoring and paper analysis |
| Result CSV and prediction `.npz` | Once per task/model/head, resumable | Score, selected hyperparameters, full scoring identity, test predictions and row IDs | Head selection, tables, figures |

The committed config is a recipe, not another runtime status ledger. The
manifests under `outputs/` are generated artifacts. Preserve historical files
for old runs; new runs should have the records above as their authoritative
path.

## 1. Stage and launch the campaign

**Edit:** `configs/revision_factorial_v1.json`,
`scripts/stage_revision_factorial_v1.py`,
`scripts/run_revision_factorial_v1.py`.

1. Put the five run IDs, representation/tokenizer pair, model size, shared
   training flags, task count, excluded task, scoring policy, and frozen file
   expectations in one committed config. Keep the command builder small.
2. Stage from a clean Git checkout. Hash the ignored train/validation Parquet,
   row-order arrays, and four tokenizer JSONs once; verify tokenizer metadata,
   the 25 prepared JSONs, and imported baseline CSV. Write one immutable
   campaign manifest; refuse to overwrite a different one.
3. Replace `production_gate.json` and its preflight-document/evidence hashes
   with a manifest-to-commit/config match. Check that the destination is empty
   for a fresh run. Immediately before any CUDA launch on Helios, inspect
   compute-process occupancy and stop if another job owns the GPU.
4. Let `--dry_run` print all five exact commands without requiring a GPU or
   writing output. Before deployment, verify all five commands and staging on
   the intended clean commit. Do not move an active Helios checkout or start a
   GPU job while the shared GPU is occupied.

**Remove:** the separate gate file, hashes of prose/evidence files, duplicated
hard-coded input hashes in the launcher, and readiness status copied among
documents. **Keep:** frozen-input verification at staging, commit/config
identity at launch, fresh-output protection, and the immediate GPU check.

## 2. Make one run record

**Edit:** `src/modernmolbert/train_selfies_ape_modernbert.py`,
`src/modernmolbert/utils.py`, `src/modernmolbert/select_pretraining_run.py`,
`src/modernmolbert/upload_model.py`, and their focused tests.

1. Write run arguments and the campaign-manifest hash to `run_identity.json`.
   Resolve the model's tokenizer through the campaign config, rather than
   copying all shared file hashes into each run directory.
2. On resume, require the original identity, matching frozen arguments and
   campaign reference, and an intact Trainer checkpoint. Keep Trainer's own
   optimizer, scheduler, RNG, and sampler state as the recovery mechanism.
3. At successful completion, append the terminal step, selected checkpoint,
   final weights hash, tokenizer hash, parameter count, and final metrics to
   the same run record. Do not call a partial checkpoint a final model.
4. Stop writing new `run_args.json`, `run_metadata.json`, checkpoint README,
   and training-time model cards. Generate the public model card at upload.
   Historical run readers may fall back to old files; new writes must not.
5. Retire production `--trace_source_rows` and its row-by-row trace output.
   Keep the existing interrupted-versus-continuous proof as an archived
   diagnostic, plus a focused resume regression test.

**Acceptance:** a fresh tiny CPU run and an interrupted/resumed tiny CPU run
reload successfully; mismatched recipe or checkpoint state fails. A new run
directory contains one run record plus Trainer's actual training artifacts.

## 3. Make one evaluation record

**Edit:** `scripts/materialize_revision_common_embeddings.py`,
`scripts/run_revision_common_scoring.py`, and embedding/scoring readers.

1. Require five completed final models. For each, verify the final weights
   against its run record and identify the exact source embedding files.
2. For each of the 25 tasks, compare the five prepared source-row IDs, labels,
   train/validation/test membership, finite embeddings, and training-side CV
   folds. Reject a task on disagreement; do not quietly narrow it to whichever
   rows happened to survive one model.
3. Record one task-level row/label/split identity, per-model source and common
   embedding content identity, coverage/rejection counts, and the five final
   run references in `evaluation_manifest.json`. Keep only the row IDs and
   other data required to score/reconstruct the common cohort inside joblib.
4. Validate the evaluation manifest against its campaign reference before a
   scoring batch. Do not rehash the same 125 embedding files in each layer of
   the wrapper and scorer. The scorer still identifies the actual file it
   loads when making a result row.

**Acceptance:** all five models for a task yield identical common row order,
labels, splits, and CV folds; a changed input file, model weights, or
missing-label policy stops scoring. The final manifest reports 25 tasks and
explicit sparse-endpoint viability.

## 4. Use the result row as the scoring recovery state

**Edit:** `src/modernmolbert/eval/benchmarking_molecular_models/score.py`,
`supervised/procedure.py`, `supervised/eval_metrics.py`, `supervised/utils.py`,
and focused tests.

1. Define one scoring identity from task, model, head, exact embedding and
   prepared-data content, common-cohort/policy reference, head-grid/CV version,
   missing-label mode, and code revision. Store it in the result CSV and
   prediction archive. Make the key unambiguous for subsampling if that
   generic scorer option remains.
2. A score is complete only if exactly one matching result row and its
   matching prediction archive exist. `--resume` skips that pair. A missing
   archive reruns the pair; a duplicate or conflicting identity is an error.
3. Remove per-head JSON checkpoints, per-dataset CSV checkpoints, and the
   separate `--cache` state. Remove the second skip/delete decision in
   `supervised/procedure.py` for the revision path so the top-level scorer
   owns resumption. Fail visibly on task-level errors; retain counted,
   narrowly handled molecule conversion failures.
4. Score one small task twice. Prove the second invocation skips the complete
   row; change the embedding or missing-label mode and prove old rows cannot
   be reused. Then run all 25 tasks and all eligible heads with one output
   table, preserving the HIV/MUV kNN exclusion.

**Acceptance:** one result table and prediction archives are sufficient to
resume an interrupted scoring batch. No checkpoint side files are needed and
no stale result can pass under a changed cohort, grid, or label rule.

## 5. Reproduction guide and historical material

**Edit:** `docs/evaluation.md`, `docs/revision_run.md`, `docs/tests.md`,
`docs/revision_factorial_v1_preflight.md`, and one current reproduction guide.

1. Give one command sequence for staging, launching/resuming five runs,
   embedding, common-cohort construction, scoring, selecting heads, producing
   figures/tables, and uploading. State where each output and manifest lives.
2. Remove current-looking instructions for `production_gate.json`,
   `--trace_source_rows`, `--checkpoint-dir`, and `--cache`. Label preflight,
   pilot, and audit reports as historical evidence and link them only where a
   scientific decision needs context.
3. Preserve the important limitations: one pretraining seed, tokenizer
   vocabularies of different realized sizes, and imported baseline rows that
   cannot be matched molecule-by-molecule to the internal common cohort.
4. Keep manuscript source data and Hugging Face uploads traceable to the
   final campaign/run/evaluation records. Upload only completed final models.

## Validation and commit sequence

1. Run Ruff, Pyright, and focused launcher/trainer/evaluation/scorer tests.
   Use a tiny CPU training and scoring fixture to test the two recovery paths.
2. Run `uv build` and the relevant full test suite after the code and docs
   agree. Do not run GPU validation while Helios is occupied; inspect the GPU
   immediately before any later CUDA training or embedding.
3. Review the diff for hidden dependencies on removed flags/files, then
   commit the simplification before resuming the Master revision campaign.
   Generated manifests, raw data, checkpoints, and model weights stay out of
   Git. State any test or environment limitation in the commit/report.

Completion means a reader can identify every model's corpus, tokenizer,
recipe, seed, final weights, every score's common cohort and scoring rule,
and every figure's source table from these records without consulting an
audit log or a second readiness gate.
