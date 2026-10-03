# Reproduce the five-model revision campaign

Run commands from the repository root with Python 3.13 and `uv`. The committed
recipe is [`configs/revision_factorial_v1.json`](../configs/revision_factorial_v1.json).
Generated data, manifests, checkpoints, predictions, and figures stay outside
Git. The five models share seed 42, the frozen training-row order and
validation IDs, 30,000 optimizer steps, and the 384-token context. The base
run and four small runs use the tokenizer/representation pairs in the recipe.

## 1. Stage the frozen inputs

Place the corpus Parquet and row-ID arrays under
`data/pretrain/chembl36_selfies/`, four tokenizer bundles under
`tokenizer/revision_factorial_v1/`, the 25 frozen task JSON files under
`data/prepared/`, and the imported baseline CSV at the configured path.
From a clean checkout of the intended code commit:

```bash
uv run --locked python scripts/stage_revision_factorial_v1.py
```

This verifies the frozen files and writes
`outputs/revision_factorial_v1/campaign_manifest.json`. Preserve this file
with the run outputs; it identifies the code, inputs, and environment. If an
input changes, investigate it before creating a new campaign manifest.

## 2. Train the five encoders on Helios

Check the GPU compute-process list immediately before each CUDA command;
training and GPU embedding must wait if another process is using the shared
GPU. The launcher also refuses an occupied GPU and a nonempty fresh output
directory. Inspect the five commands before running them:

```bash
for run in small_ape_selfies small_ape_smiles small_bpe_selfies small_bpe_smiles base_ape_selfies; do
  uv run --locked python scripts/run_revision_factorial_v1.py "$run" --dry_run
done
```

Run one model at a time, in the intended order, with
`uv run --locked python scripts/run_revision_factorial_v1.py <run_id>`. To
resume an interrupted run, supply its intact Trainer checkpoint through
`--resume_from_checkpoint <checkpoint-path>`. Each run writes a
`run_identity.json` next to its checkpoints; after completion it records the
selected step and final weights. Do not use a partial run as a final model.

## 3. Embed the frozen benchmark tasks

Use the completed `final_model` of each run. For each `run_id`, use a unique
source embedder name `REVISION_<run_id>`:

```bash
uv run --locked python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/revision_factorial_v1/<run_id>/seed42/final_model \
  --tokenizer-path runs/revision_factorial_v1/<run_id>/seed42/final_model \
  --embedder REVISION_<run_id> \
  --batch-size 32 --device cuda --pooling mean
```

After the five-model queue reports complete, the Helios wrapper
`bash scripts/run_revision_final_embeddings.sh` checks the campaign and final
model hashes, requires an idle shared GPU at each model boundary, logs each
embedding pass, and then materializes the common cohort. It stops on a failed
check or embedding. Use `UV_BIN` to override `/opt/lab/bin/uv` on another host.

Preserve the source embedding joblibs in `data/embedded/`. The current registry
contains exactly the 25 frozen paper tasks. The embedding pipeline records the
final weight hash, row IDs and rejection counts; it rejects lossy, unknown,
over-context, and invalid molecules under the declared policy. Avoid
overwriting an embedding without rebuilding the evaluation manifest.

## 4. Build and score the common cohort

After all five final embedding sets exist:

```bash
uv run --locked python scripts/materialize_revision_common_embeddings.py
```

This verifies the five final models and their source embeddings, then writes
the common embedding joblibs and
`outputs/revision_factorial_v1/evaluation_manifest.json`. Every task uses the
same retained row order, labels, splits, and five-fold training-side CV rule.
Review coverage and sparse-endpoint viability in that manifest before
interpreting scores.

Score each task/model pair with
`uv run --locked python scripts/run_revision_common_scoring.py <run_id> <task>`.
The wrapper checks the evaluation manifest and selected embedding. It runs
the three eligible heads with `--missing-labels as-negative`; kNN is disabled
for HIV and MUV. Results go under
`outputs/eval/revision_factorial_v1/common_rows/`, with predictions under
`data/predictions/`. Repeating a command skips a head only when its CSV row
and prediction archive carry the current scoring identity. A conflicting
cohort or policy raises an error.

On Helios, run `bash scripts/run_revision_final_scoring.sh` in a persistent
`tmux` session after reviewing the final evaluation manifest. The wrapper
checks that all 25 tasks have all five models, scores smaller tasks first on
CPU with four workers by default, and writes
`outputs/revision_factorial_v1/final_scoring_queue.status` plus one log per
task/model under `outputs/revision_factorial_v1/logs/final_scoring/`. It stops
at a failed pair. Re-running it resumes only heads whose saved result row and
prediction archive match the same scoring identity. Set `N_JOBS` to change
the per-pair CPU worker count.

## 5. Tables, figures, and upload

Use [`revision_run.md`](revision_run.md) for the CV head-selection and paper
generation commands. Feed the final head-level CSVs for all 25 tasks and five
models, together with their prediction archives,
into `scripts/paper/build_common_row_benchmark.py`; use its common-row task
matrix for internal model comparisons. The imported Praski baseline CSV is
external context: its test molecules cannot be paired with the internal
common cohort. Keep the selected-head table, final score matrices, source
data, and figures together with the evaluation manifest.

Only after final-model reload and result review, stage each model upload with:

```bash
uv run python -m modernmolbert.upload_model \
  --run_dir runs/revision_factorial_v1/<run_id>/seed42 \
  --repo_id <owner/repo> --checkpoint final --dry_run
```

Review the staged model card and files, then
run the upload command without `--dry_run`. The uploaded model card is derived
from the final run record; historical pilot and audit files are diagnostic
evidence, not additional launch requirements.

The initial seed-42 campaign has one pretraining seed; the completed expansion has five. Its tokenizer vocabularies have
different realized sizes, so the factorial compares complete
tokenizer/representation configurations. Report both facts with the results.

## 6. Four additional seeds per configuration

The five completed seed-42 runs are the first replicate. The separate
`configs/revision_factorial_multiseed_v1.json` recipe requests seeds 43–46
for each of the same five configurations, with the same frozen corpus,
tokenizers, training-row order, validation IDs, exposure, and optimizer recipe.
Stage a new manifest from the clean code commit used for these runs:

```bash
uv run --locked python scripts/stage_revision_factorial_v1.py \
  --spec configs/revision_factorial_multiseed_v1.json
```

On Helios, inspect `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory
--format=csv,noheader` before starting the persistent queue. Run
`bash scripts/run_revision_multiseed_queue.sh` in `tmux`. The queue checks
GPU occupancy before each run and waits while another compute process uses
it. It writes progress to
`outputs/revision_factorial_multiseed_v1/queue_status.txt` and one training
log per run/seed, verifies step 30,000 and the final weight hash, then checks
the separate `/data/modernmolbert_revision_factorial_v1/` copy before
advancing. An interrupted run resumes only from its newest complete Trainer
checkpoint; a partial run without such a checkpoint stops for inspection.
The original seed-42 campaign manifest and final evaluation files remain
their own evidence; merge results across seeds only after each seed has
passed the same embedding and scoring policy.

After the training queue completes, embed each new seed on an idle GPU with
`bash scripts/run_revision_multiseed_embeddings.sh <seed>`. This verifies the
five final weights against the multiseed manifest and all frozen prepared
tasks, writes per-model logs, then creates
`outputs/revision_factorial_multiseed_v1/evaluation_seed<seed>.json`.
It checks that the retained supervised rows, labels and split hashes match
the accepted seed-42 cohort. An occupied GPU stops an embedding pass.

Score each accepted seed on CPU with
`bash scripts/run_revision_final_scoring.sh <seed>` in a persistent `tmux`
session. The queue writes `scoring_seed<seed>.status` and per-task/model
logs under `outputs/revision_factorial_multiseed_v1/`. Its scorer verifies
the seed-specific common embedding and manifest hashes before every pair.

When every seed has 125 accepted task/model scores, build each seed's
`common_task_matrix.csv` with `build_common_row_benchmark.py`, passing that
seed's evaluation manifest via `--evaluation-manifest` and all five
`REVISION_COMMON_s<seed>_<run_id>` embedders. Give them the same five paper
labels. Seed 42 uses the unsuffixed `REVISION_COMMON_<run_id>` prefix. Keep
the resulting selection `manifest.json` next to each matrix.

Pass those five matrices and five evaluation manifests as `SEED=PATH`
arguments to `scripts/paper/aggregate_revision_seeds.py`. It requires all
25 task rows and the same supervised row, label, split and endpoint hashes
across seeds. Its `mean_common_task_matrix.csv` supplies internal paper
scores; `seed_task_contrasts.csv` and `overall_contrast_summary.csv` retain
five independent seed-level effects. Run
`scripts/paper/compute_revision_contrast_intervals.py` on the mean matrix
and overall contrast summary for separate task and fixed-family bootstrap
intervals. These intervals condition on the five trained seeds; report the
between-seed SD beside them, not as 125 independent training replicates.

## 7. Accepted five-seed analysis and loss source data

The accepted seed selections and aggregate were regenerated from the clean
code commit `95b97ac` after all CPU scores completed. On the analysis
checkout, run these stages in order after verifying the five evaluation
manifests and 125 score files per seed:

```bash
for seed in 42 43 44 45 46; do
  bash scripts/run_revision_multiseed_analysis.sh select-seed "$seed"
done
bash scripts/run_revision_multiseed_analysis.sh aggregate
bash scripts/run_revision_multiseed_analysis.sh paper
```

Each selection manifest records input and output hashes. The aggregate
manifest pins the five common-row matrices, five overlap-exclusion matrices,
seeds, code commit and output hashes. The paper matrix combines five-seed
internal means with four table-only imported baselines; external molecule
rows and head-selection provenance are not matched. The six internal
contrasts use the unchanged 18-family mapping, with task and family
resampling conditional on the five seeds.

For within-run MLM diagnostics, collect the top-level `trainer_state.json`
from each of the five model directories and five `seed42`–`seed46`
subdirectories under `runs/revision_factorial_v1/`, preserving their
relative paths. Then run:

```bash
uv run python scripts/paper/make_revision_loss_curves.py \
  --run-root runs/revision_factorial_v1 \
  --figure outputs/eval/revision_factorial_multiseed_v1/paper/revision_loss_curves.pdf \
  --source-data outputs/eval/revision_factorial_multiseed_v1/paper/loss_curves_25runs.csv
```

The script requires terminal step 30,000 and the expected 300 training and
seven validation loss records for each run. The CSV records every plotted
value and each trainer-history SHA-256. Compare curves within a run only;
MLM targets differ across tokenizers.

To inventory the exact 625 CV-selected native prediction archives for release,
run the manifest builder against the five selection directories and the
25-row training identity table. Passing `--repo-root` recomputes every archive
SHA-256 and byte count; omit it only when preparing an unverified draft.

```bash
uv run python scripts/paper/build_revision_release_manifest.py \
  --selection-root outputs/eval/revision_factorial_multiseed_v1 \
  --training-models path/to/revision_training_models.csv \
  --repo-root . \
  --output outputs/eval/revision_factorial_multiseed_v1/prediction_release_manifest.csv
```

This manifest describes only the selected head for each model, seed and task;
other candidate archives are diagnostic. It also records the final weight and
tokenizer hashes needed to link published checkpoints. The existing
`modernmolbert.upload_model` card and validation path is specific to
APE–SELFIES and must be generalized or replaced before publishing SMILES or
BPE variants; do not use its hard-coded description for those runs.

For model publication, `modernmolbert.upload_model` remains limited to the
historical APE–SELFIES format. Use the representation-aware staging command
for each revision run instead; it verifies the terminal weight/tokenizer
hashes, copies the deployable final model, writes an accurate variant card,
reloads the tokenizer and masked-LM model on CPU, and hashes the staged files:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/stage_revision_model_release.py \
  --run-dir runs/revision_factorial_v1/small_bpe_smiles/seed42 \
  --output-dir outputs/revision_release_staging/small_bpe_smiles_seed42 \
  --repo-id HauserGroup/ModernMolBERT-revision-small-bpe-smiles-seed42
```

All 25 combinations of the five run IDs and seeds 42–46 passed this CPU
staging validation on Helios. The repository names shown in the staging
cards are proposals; confirm the final Hub layout before upload. The stage
command performs no network publication.
