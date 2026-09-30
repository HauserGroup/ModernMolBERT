# Provenance and guardrail simplification plan

30 September 2026. This plan concerns the revision campaign: one frozen
pretraining corpus, four tokenizer artifacts, five model runs, 25 frozen
supervised tasks, one imported baseline table, figures, and eventual model
upload. It is narrower than [the general simplification inventory](simplification_candidates_2026-09-30.md).
The benchmark fits heads to frozen task splits; it does not obtain model scores
by comparing embeddings directly with the imported CSV.

## Rule for deciding what stays

Keep a check if removing it could silently change **which molecules, labels,
folds, tokenizer, model weights, or scoring rule** produced a reported result.
Keep checkpoint integrity checks that make interruption recovery safe. Prefer
one authoritative record for each fact, written once at the boundary where
the fact is established. A diagnostic proof can be archived after it has
served its purpose; it does not need to become a permanent runtime gate.

The intended records are:

1. **One campaign manifest:** code commit and lockfile identity; frozen
   pretraining train/validation files and row-order files; the four tokenizer
   hashes; benchmark dataset/split identities; imported baseline CSV identity;
   shared recipe and evaluation policy.
2. **One small run record per model:** campaign manifest reference, run ID,
   tokenizer and representation, model size, seed, final step, selected
   checkpoint, and terminal model hash. Keep Trainer's checkpoint state for
   actual resumption.
3. **One evaluation record:** the 25 task cohorts, common source-row and split
   identities, five final model references, head grids/CV rule, missing-label
   rule, coverage, and selected heads. Retain final predictions and scores.

Do not alter `run_identity.json`, checkpoint formats, or the training command
while a production model is running. Finish or safely stop the campaign first;
then migrate the format with a focused resume test. Existing historical
manifests remain readable as archived evidence.

## Specific reductions

| Current mechanism | Simplification | Essential check retained |
| --- | --- | --- |
| `scripts/run_revision_factorial_v1.py` requires `production_gate.json` with a matching commit and four separate evidence hashes, including two hashes of the same preflight document. | Remove the gate file and evidence-hash comparison. Keep the five fixed recipes in one campaign configuration and a short human-readable readiness note. | Refuse a nonempty fresh run directory, wrong tokenizer/representation, or an occupied Helios GPU. Check code commit and frozen inputs against the campaign manifest before the first launch. |
| The launcher hard-codes hashes for train/valid Parquet, both row arrays, four tokenizers, and `uv.lock`; `run_identity.json` then hashes many of them again for each run. | Store these hashes once in the campaign manifest. Each run records its manifest ID, relevant tokenizer, recipe, and seed. The clean code commit already identifies the checked-in config and lockfile; no separate per-run lock/config hash is needed. | Check the ignored data and tokenizer files against the campaign manifest when staging on Helios and before launching the batch. On resume, require the same manifest ID, recipe, checkpoint state, and tokenizer. |
| `run_identity.json`, `run_args.json`, `run_metadata.json`, and model README files repeat arguments and environment facts. | Keep one machine-readable run record; generate a model card when uploading. Put final metrics and selected step in that run record, or link to Trainer's state, instead of copying every argument into several files. | Record the exact command/recipe, code commit, terminal step, final weights, and tokenizer; put shared library versions in the campaign manifest. |
| `--trace_source_rows` writes every consumed training row ID and the preflight uses byte-identical resumed-vs-continuous artifacts as a launch proof. | Keep the completed proof as a diagnostic. Remove tracing from normal production runs and do not require a new trace for every run. | Retain a focused regression test for interrupted resumption and validate checkpoint state before resuming. |
| Pilot coverage JSON, common-embedding JSON, preflight prose, audit status, and the manuscript plan repeat the same row and readiness numbers. | Make one final evaluation manifest authoritative. Give the manuscript concise methods/results and links; keep the pilot report archived, clearly marked diagnostic. Stop updating several status narratives for one change. | Preserve source-row IDs, train/validation/test membership, per-model and common coverage, and endpoint viability. |
| `materialize_revision_common_embeddings.py` stores source and common hashes for each of 125 files and repeats row/split/coverage metadata inside every joblib. `run_revision_common_scoring.py` rehashes each selected file before its task run. | Check each model's row IDs, labels, splits, and finiteness when building common cohorts. Record one task-level row/split fingerprint and one content identity per final embedding in the evaluation manifest. Validate those identities once for a scoring batch, rather than duplicating coverage metadata and checking the same content at several layers. | All five models must use identical supervised rows, labels, splits, and deterministic training-side CV folds; scores must identify the embedding content actually used. |
| `score.py` has result-CSV skipping, per-head JSON checkpoints, per-dataset CSV checkpoints, `--cache`, `--resume`, and `--safe`; the campaign wrapper adds another fresh-output layer. | Use one result table keyed by campaign, task, model, head, scoring policy, and input cohort. Complete rows can resume; incomplete work reruns. Until that replacement is tested, continue the existing fresh-output, `--no-cache --no-resume` campaign path. Remove the redundant checkpoint layers afterward. | Never accept a score or prediction from another cohort, missing-label rule, grid, or code revision. Fail visibly on a scoring error. |
| Several broad catches and fallback readers convert missing/corrupt artifacts into a skipped task or a generic failure. | For the revision path, make missing/corrupt inputs and invalid checkpoint state explicit errors. Retain narrow per-molecule conversion failures as counted rejection reasons so the 25-task coverage remains measurable. | A failed task must not silently appear as a complete result. One bad molecule may be excluded only under the declared common-row rule. |
| Multiple audit documents and historical guides contain live-looking launch instructions and repeated checklists. | Maintain one current reproduction guide and one concise campaign manifest. Mark older audit and pilot notes historical; do not use them as additional launch gates. | Keep scientific decisions and known limitations, including the single pretraining seed and the imported baseline's unmatched rows. |

## Implementation order

1. **Freeze campaign state before edits.** Inventory active Helios jobs and
   commits. Do not change the trainer or remote checkout during a production
   run. Preserve current checkpoint and manifest files.
2. **Define the three small records above.** Give each field a single owner.
   Check that the manifest can identify every reported figure and table value
   without the audit log. Record the five shared recipes as data rather than
   duplicating command arguments in prose.
3. **Collapse launch provenance.** Replace `production_gate.json` and duplicate
   hashes with the campaign manifest, a one-time staging check, a clean commit
   check, and the immediate GPU occupancy check. Run a dry run for all five
   commands before retiring the old gate.
4. **Collapse run output.** Change the trainer only between campaigns. Verify
   one fresh run and one interrupted resume against the new run record. Keep
   legacy readers for existing runs, but stop writing duplicate new files.
5. **Collapse evaluation state.** Build a final common-cohort manifest, verify
   identical rows/folds once, then score a small task twice to prove resume
   cannot reuse a different cohort or missing-label policy. Remove redundant
   cache/checkpoint paths only after this passes. Re-run the final 25-task
   validation, including sparse endpoint counts.
6. **Trim documentation and one-off diagnostics.** Archive the pilots and
   audit; link final source data, commands, predictions, and checkpoints from
   one reproduction guide. Keep the figures and Hugging Face upload tied to
   the final run and evaluation records.

## Acceptance criteria

- A reader can reproduce each of the five runs and identify its tokenizer,
  corpus, code, seed, recipe, and terminal checkpoint from the campaign and
  run records alone.
- A reader can trace every internal score to the same supervised rows/folds,
  its selected head, predictions, and scoring policy. External baseline values
  are labelled as unmatched context.
- A changed tokenizer, corpus, cohort, split, checkpoint, or missing-label
  rule produces a visible failure rather than an apparently valid result.
- The normal launch and scoring paths each have one identity check and one
  recovery mechanism; pilot proofs and audit narratives are no longer runtime
  dependencies.
