# Clean small-model revision run

This is a proposed, **single full run** for the submission revision: train one
APE tokenizer using only the ChEMBL 36 training split, pretrain one small
ModernMolBERT encoder with standard masking, then embed and score every one of
the 25 datasets in the benchmark configuration. Keep the released preprint
tokenizer, checkpoints, embeddings, and result CSVs as historical artifacts.
The commands below have not been run as a full experiment.

Run from the repository root. Before starting, freeze the code commit, the
prepared ChEMBL data (`data/pretrain/chembl36_selfies/metadata.json` and Parquet
hashes), the benchmark configuration and prepared splits, package lockfile, and
the hyperparameters below in a run record. The tokenizer's metadata
records its own vocabulary and scanned training-Parquet hashes; the model writes
`run_args.json` and run metadata. Do not label the new run as a reproduction of
the released weights: its tokenizer and training recipe are new.

## 1. Train the corpus-only tokenizer

The new name is deliberate; the committed 631-token preprint vocabulary must
remain intact. APE learns merges from two million **training-split** SELFIES.
The additional scan adds any primitives seen in the *full* training Parquet,
including component separators. It does not read validation or benchmark
molecules and cannot be combined with either `--extra_vocab_*` option.

```bash
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column selfies --representation SELFIES \
  --tokenizer_train_size 2000000 --max_vocab_size 2000 \
  --min_freq_for_merge 3000 --max_merge_pieces 2 --seed 42 \
  --corpus_primitive_parquet data/pretrain/chembl36_selfies/train.parquet
```

Check the adjacent `.metadata.json`: `extra_vocab_symbols_requested` and
`extra_vocab_symbols_added` must both be zero, and `corpus_primitive_scan`
must identify the complete training Parquet with its row count and SHA256.
Check lossless round-trip for disconnected SELFIES (`[C].[O]`). If any of these
checks fail, fix the input or tokenizer before model training.

## 2. Validate and preflight

The validator samples rows; a zero unknown rate in a sample is not proof of
full-corpus coverage. The full training-split primitive scan above supplies
that coverage check. Also inspect validation and prepared benchmark coverage
separately; unsupported benchmark symbols should be counted, never added to
the tokenizer from those benchmark inputs.

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --dataset_name data/pretrain/chembl36_selfies --molecule_column selfies \
  --split train --n 10000 --max_seq_length 128 --unk_rate_threshold 0

uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --dataset_name data/pretrain/chembl36_selfies --molecule_column selfies \
  --split valid --n 10000 --max_seq_length 128 --unk_rate_threshold 0

uv run python scripts/audit_benchmark_inputs.py \
  --vocab tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --max-length 128 \
  --output outputs/audit/revision_clean_small_v1_benchmark_inputs.csv
```

Record unknown, conversion-failure, disconnected-component, and truncation
counts by split. If the validation check finds unknowns, report and resolve
that before choosing how validation loss will be used. Do not add its symbols
to the training vocabulary. `scripts/audit_benchmark_inputs.py` reads the
prepared benchmark JSON files; confirm that all 25 configured datasets are
represented in its output.

Use a **separate** debug directory for the short smoke test. `--debug` caps
training at 200 steps, so its loss is not a scientific result.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug --output_dir runs/revision_clean_small_v1_debug \
  --dataset_name data/pretrain/chembl36_selfies --selfies_column selfies \
  --train_split train --use_validation_split --validation_split valid \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --require_corpus_only_vocab --model_size small --max_seq_length 128 \
  --masking_strategy standard --mlm_probability 0.15 --no-bf16
```

Reload `runs/revision_clean_small_v1_debug/final_model` with
`AutoModelForMaskedLM` and load its tokenizer from the `ape_tokenizer/`
subdirectory, as in [tests.md](tests.md). Require finite logits and a
successful validation pass before the full run.

## 3. One full encoder pretraining run

The archived selected small standard run's `run_args.json` records 30,000
optimizer steps, learning rate `4e-4`, masking probability `0.15`, 128-token
context, seed 42, 1,500 warmup steps, and validation loss for checkpoint
selection. The command keeps those settings and its effective single-device
batch of 256, using 128 × 2 gradient accumulation as a proposed hardware
arrangement. Freeze that arrangement and precision before launch; record
any difference from the archived run. Keep the evaluation schedule aligned
with the save schedule. `--no-bf16` is the portable full-precision setting;
supported CUDA hardware may use `--bf16`.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/revision_clean_small_v1 \
  --dataset_name data/pretrain/chembl36_selfies --selfies_column selfies \
  --train_split train --use_validation_split --validation_split valid \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.metadata.json \
  --require_corpus_only_vocab --model_size small --max_seq_length 128 \
  --masking_strategy standard --mlm_probability 0.15 \
  --max_steps 30000 --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 2 --learning_rate 4e-4 \
  --warmup_steps 1500 --eval_size 4096 \
  --eval_steps 5000 --save_steps 5000 \
  --seed 42 --no-bf16
```

On completion, use `runs/revision_clean_small_v1/final_model` as the selected
checkpoint; the trainer loads the best evaluated checkpoint by `eval_loss`
before saving `final_model/`. Keep its tokenizer bundle and metadata with the
weights. Check `trainer_state.json`, final evaluation metrics, run arguments,
and checkpoint reload. Report the observed 128-token truncation rate on the
population actually measured.

## 4. Re-embed and re-score the full configured benchmark

Use a unique embedder name and output CSV. The embedding script currently has
an **independent historical default** for `--tokenizer-path`, so specify the
new model path in *both* arguments. It rejects lossy or unknown tokenization
and removes failed rows before scoring; preserve its per-dataset metadata and
report valid/failed test counts. Mean pooling excludes special tokens.

```bash
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py \
  --datasets all

uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all \
  --model-dir runs/revision_clean_small_v1/final_model \
  --tokenizer-path runs/revision_clean_small_v1/final_model \
  --embedder modernmolbert_revision_clean_small_v1 \
  --batch-size 32 --device auto --max-seq-length 128 --pooling mean

uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --datasets all --heads rf ridge knn \
  --embedder modernmolbert_revision_clean_small_v1 \
  --output-csv outputs/eval/revision_clean_small_v1/results.csv \
  --checkpoint-dir outputs/eval/revision_clean_small_v1/checkpoints
```

`download.py` uses cached prepared datasets by default. Freeze and verify
those splits rather than silently replacing them. The 25 entries in
`src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml`
define this full run; score all eligible heads without
subsampling. New embeddings live at
`data/embedded/<dataset>/modernmolbert_revision_clean_small_v1.joblib`, so
the historical embedding files remain available. Inspect every failed head
and confirm one score row for each eligible dataset/head pair. Select the
paper-facing head for each dataset using **training-side `cv_metric` only**;
evaluate the selected head's held-out `test_metric` once. Keep the complete
head-level CSV and the selected-head table, and derive the manuscript table,
paired statistics, and quantitative figures from that same verified table.
The existing paper aggregation scripts use historical embedder names and
mixed-provenance source files; update their input mapping and verify baseline
comparability before using them for the revision. The run above alone does not
establish that older baseline scores used identical prepared rows or grids.

Before reporting a cross-model comparison, record dataset/split IDs, the
number of retained rows for each model and baseline, scoring-grid/version
hashes, and the handling of conversion failures. Report any test-set overlap
with pretraining separately. Cite the historical preprint results only as
historical results until these checks pass.
