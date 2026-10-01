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

Preserve the source embedding joblibs in `data/embedded/`. The embedding
pipeline records row IDs and rejection counts; it rejects lossy, unknown,
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

The campaign has one pretraining seed. Its tokenizer vocabularies have
different realized sizes, so the factorial compares complete
tokenizer/representation configurations. Report both facts with the results.
