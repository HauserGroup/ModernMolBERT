# SMIRK five-seed pilot

This is an isolated experiment with one **small ModernMolBERT / SMIRK / SMILES**
configuration trained at seeds 42–46. It uses the ChEMBL 36 corpus after
excluding 68/2,390,314 training rows that SMIRK cannot represent losslessly
or that exceed 384 tokens. The retained molecules preserve their original
relative frozen row order. It uses the same validation IDs, 384-token context,
30,000-step optimizer recipe and 25 prepared benchmark tasks as the revision campaign. It is not
part of the manuscript or its five-configuration aggregate.

SMIRK 0.3.0 is the [upstream OpenSMILES tokenizer](https://eeg.engin.umich.edu/smirk/).
Its fixed grammar vocabulary is not learned from the corpus. The frozen local
artifact has 165 IDs under this version, including its seven special tokens.
The model uses `[BOS]`/`[EOS]` boundaries. SMIRK's package metadata requires
Transformers 4.x while this repository requires 5.x; the tokenization,
checkpoint reload and training adapter are tested with the repository's 5.x
runtime, so the pilot installs SMIRK without its conflicting declared
dependencies. Do not use this install method for the main revision campaign.

## Stage and train on Helios

Use a clean checkout of the experiment code. The committed tokenizer and
`configs/experimental_smirk_v1.json` pin all inputs. On Helios:

```bash
/opt/lab/bin/uv sync --locked
/opt/lab/bin/uv pip install --python .venv/bin/python --no-deps smirk==0.3.0
/opt/lab/bin/uv run --locked --no-sync python scripts/experiments/prepare_smirk_corpus.py
/opt/lab/bin/uv run --locked --no-sync python scripts/experiments/audit_smirk_corpus.py
/opt/lab/bin/uv run --locked --no-sync python scripts/experiments/stage_smirk_campaign.py
```

The filtered corpus audit must show zero unknown, lossy and over-context rows
for training and the same 4,096 selected validation rows before a full run.
The original full validation file has one unsupported row, which is not in
the frozen selected validation set. Validate a short checkpoint and
reload it before starting the production queue. Check GPU processes immediately
before each GPU run:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

The queue checks that query again at each seed boundary, waits while the GPU
is occupied, and backs up verified final runs to
`/data/modernmolbert_experimental_smirk_v1/`:

```bash
tmux new -s smirk-pilot
bash scripts/experiments/run_smirk_queue.sh
```

Progress is in `outputs/experimental_smirk_v1/queue_status.txt` and
`logs/train_seed<seed>.log`. A final model is accepted only with 30,000 steps,
7.68 million presentations, a matching campaign/code identity and a verified
weight hash. Interruptions resume only from a complete Trainer checkpoint.

## Evaluation policy

For each completed seed, embed the same 25 prepared tasks using the final
SMIRK model, frozen mean pooling and the 384-token context. The tokenizer
preflight found 36 failures in the 25 prepared tasks; ten of these are in the
accepted revision cohort (two HIV and eight SIDER rows). Preserve the
source-row map and compare it with that seed's accepted revision cohort before
any paired comparison. Use the same training-side five-fold CV, RF/L2-logistic/
kNN candidates, HIV/MUV kNN exclusion, `as-negative` missing-label policy and
prediction-provenance checks. If SMIRK rejects a row in the accepted cohort,
report the reduced intersection explicitly and re-evaluate comparison models
on that intersection before treating scores as paired. Keep all pilot outputs
under `outputs/experimental_smirk_v1/`; do not feed them to manuscript result
generators until the pilot is reviewed.

The separate Helios analysis checkout can evaluate all five models after the
training queue completes. It checks the GPU before every embedding stage and
scores on CPU. Its queue status and logs stay under
`outputs/experimental_smirk_v1/` in that checkout:

```bash
# From /home/jakob/projects/ModernMolBERT-analysis after installing smirk==0.3.0:
tmux new -s smirk-evaluation
bash scripts/experiments/run_smirk_evaluation_queue.sh
```

For each task the evaluation manifest records the exact source-row intersection
with the accepted five-model cohort. Existing comparator results can be used
when the cohort is unchanged. For any task with lost rows, the queue writes
separate matched comparator embeddings and scores all five on that reduced
cohort. No accepted embedding, score, or manuscript file is overwritten.
