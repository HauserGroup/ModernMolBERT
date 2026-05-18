# ModernMolBERT

[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<img src="imgs/mmbert_text.png" width="600"/>

ModernMolBERT is a SELFIES-native molecular representation project. It trains a ModernBERT masked-language model on SELFIES strings and evaluates the resulting encoder as a frozen molecular featuriser through one shared benchmark pipeline.

The core path is:

```text
SELFIES tokenizer
-> ModernBERT masked-language pretraining
-> saved checkpoint
-> ModernMolBERTSelfiesFeaturizer
-> benchmark suite runner
-> shared downstream models and metrics
-> reports and plots
```

## Project scope

ModernMolBERT focuses on:

- **Input representation:** SELFIES.
- **Tokenizer:** Hugging Face-compatible `APEPreTrainedTokenizer`, trained separately from model training and loadable with `AutoTokenizer`.
- **Model:** ModernBERT architecture trained from scratch with the SELFIES tokenizer vocabulary.
- **Objective:** masked language modeling.
- **Primary use:** frozen molecular featurisation.
- **Primary benchmark path:** config-driven benchmark suites using shared featurizers, downstream models, metrics, caching, and reporting.

The Answer.AI ModernBERT checkpoints are used as architecture/config references. The main SELFIES pretraining run is not initialized from English/code pretrained weights.

Supported model-size choices are:

```bash
--model_size base
--model_size large
```

Use `base` for development, debugging, and pilot runs. Do not use `large` until the tokenizer, training loop, checkpoint reload, featuriser, and frozen benchmark path are stable.

## Documentation

Detailed documentation is split by topic:

- [Datasets](docs/datasets.md): MoleculeNet preparation, prepared dataset layout, and dataset config examples.
- [Baselines and external models](docs/baselines.md): ECFP4, Hugging Face SMILES encoders, ChemBERTa, and MoLFormer notes.
- [Tests and readiness checks](docs/tests.md): unit tests, smoke tests, readiness gates, and optional environment-specific tests.
- [Evaluation suite configs](configs/eval_suites/README.md): benchmark suite YAML schema and examples.

## Repository layout

```text
src/modernmolbert/
  tokenization_ape.py                  # Hugging Face-compatible APE tokenizer
  ape_tokenizer.py                     # deprecated legacy tokenizer compatibility API
  train_ape_tokenizer.py               # tokenizer training CLI
  validate_tokenizer.py                # tokenizer validation CLI
  train_selfies_ape_modernbert.py      # MLM pretraining CLI
  select_pretraining_run.py            # rank sweep runs and select best checkpoint
  collator.py                          # MolecularMLMCollator (standard / span / hetero_span)
  paths.py                             # repository-relative path helpers
  utils.py                             # shared dataset/tokenizer helpers

src/modernmolbert/eval/
  featurizers/                         # frozen representation featurizers
  cache.py                             # feature cache
  datasets.py                          # evaluation dataset loading
  downstream.py                        # downstream models
  metrics.py                           # shared metrics
  pooling.py                           # shared pooling utilities
  registry.py                          # config-driven featurizer registry
  task_eval.py                         # task-level label/feature alignment and evaluation
  runner.py                            # single benchmark runner
  suite.py                             # benchmark suite runner
  reporting.py                         # summary tables and plots
  moleculenet.py                       # MoleculeNet preparation utilities
  cli/                                 # evaluation CLIs

configs/
  eval_suites/                         # benchmark suite YAML files
  featurizers/                         # individual featurizer JSON configs

docs/
  baselines.md
  datasets.md
  tests.md
```

Older direct embedding/sklearn scripts should not be used for benchmark results. ModernMolBERT should be evaluated through `ModernMolBERTSelfiesFeaturizer` and the shared benchmark suite machinery.

## Installation

Create a Python 3.13 environment and install dependencies with `uv`:

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv sync
```

For CUDA training, install the CUDA optional dependencies appropriate for your machine:

```bash
uv sync --extra cuda
```

FlashAttention is optional and CUDA-oriented. Do not install it for Mac MPS. Install it only if your PyTorch, CUDA, and GPU setup support it:

```bash
uv pip install flash-attn --no-build-isolation
```

## Canonical tokenizer artifacts

Tokenizer training is intentionally separate from model training. Model training should only use a validated tokenizer plus matching metadata.

Canonical artifacts:

```text
tokenizer/selfies_ape_tokenizer.json
tokenizer/selfies_ape_tokenizer.metadata.json
```

The metadata records the tokenizer representation, vocabulary hash, special token IDs, and training provenance. The model training script validates the metadata and SHA256 hash before model construction.

### Hugging Face tokenizer vocab files

`APEPreTrainedTokenizer` is a slow Hugging Face tokenizer and can be loaded with
`AutoTokenizer.from_pretrained(..., trust_remote_code=True)`. A tokenizer repo
may use either the legacy single-vocab layout or representation-specific vocab
files:

```text
vocab.json          # fallback / legacy active vocabulary
selfies_vocab.json  # optional SELFIES vocabulary
smiles_vocab.json   # optional SMILES vocabulary
```

At load time, `representation` selects the active vocabulary. If
`representation="SELFIES"` and `selfies_vocab.json` exists, that file is used.
If `representation="SMILES"` and `smiles_vocab.json` exists, that file is used.
Otherwise the tokenizer falls back to `vocab.json`.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "path-or-hf-repo",
    trust_remote_code=True,
    representation="SMILES",
)
```

The tokenizer still keeps one active vocabulary in memory. The multiple file
parameters are for choosing the vocabulary at construction/load time, not for
switching representations on an already-instantiated tokenizer.

### Uploading trained checkpoints to Hugging Face

`train_selfies_ape_modernbert.py` writes a Hub-ready `final_model/` directory
containing model weights, model config, tokenizer files, custom tokenizer code,
and a model card. Upload that folder as a model repository after validating it
locally:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("HauserGroup/<repo-name>", repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path="runs/.../final_model",
    repo_id="HauserGroup/<repo-name>",
    repo_type="model",
    commit_message="Upload ModernMolBERT checkpoint",
)
```

Uploaded checkpoints load the model from the repository root. With current
Transformers versions, load the custom tokenizer from the `ape_tokenizer/`
subfolder because root ModernBERT configs disable remote tokenizer code:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("HauserGroup/<repo-name>")
tokenizer = AutoTokenizer.from_pretrained(
    "HauserGroup/<repo-name>",
    subfolder="ape_tokenizer",
    trust_remote_code=True,
)
```

## Training workflow

### 0. Prepare the pretrain dataset

Prepare the ChEMBL36 SELFIES dataset for pretraining:

```bash
uv run python -m modernmolbert.data.prepare_chembl36_selfies \
  --output_dir data/pretrain/chembl36_selfies
```

This prepares the dataset with no test split by default. The prepared dataset will be used for model pretraining in the training steps below.

### 1. Train the SELFIES tokenizer

```bash
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --dataset_name mikemayuare/PubChem10M_SMILES_SELFIES \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000
```

Expected outputs:

```text
tokenizer/selfies_ape_tokenizer.json
tokenizer/selfies_ape_tokenizer_freq.json
tokenizer/selfies_ape_tokenizer.metadata.json
```

### 2. Validate the tokenizer

Tokenizer validation is a mandatory gate before model training.

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --n 1000
```

Expected output includes:

```text
representation: SELFIES
vocab_size: ...
unk_rate: ...
mean_len: ...
p95_len: ...
truncation_rate@256: ...
special_ids: ...
```

### 3. Debug model training

The debug run is a tiny end-to-end check. Use this before any pilot or long run.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --model_size base \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

### 4. MPS smoke run

This is a Mac smoke test, not a useful model.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/mps_base_smoke_512 \
  --device_backend mps \
  --model_size base \
  --max_seq_length 512 \
  --max_steps 100 \
  --eval_size 32 \
  --max_eval_batches 4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --mlm_probability 0.30 \
  --learning_rate 1e-4 \
  --logging_steps 10 \
  --eval_steps 25 \
  --save_steps 50 \
  --save_total_limit 2 \
  --num_workers 0 \
  --report_to tensorboard \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

View logs:

```bash
uv run tensorboard --logdir runs/mps_base_smoke_512
```

### 5. CUDA pilot run

This is the first meaningful training pilot. It is smaller than a final pretraining run, but large enough to check learning dynamics, throughput, checkpointing, and evaluation behavior.

```bash
uv run accelerate launch -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/cuda_base_pilot_512 \
  --device_backend cuda \
  --model_size base \
  --max_seq_length 512 \
  --max_steps 5000 \
  --eval_size 2048 \
  --max_eval_batches 32 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --mlm_probability 0.30 \
  --learning_rate 1e-4 \
  --logging_steps 25 \
  --eval_steps 500 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --num_workers 4 \
  --report_to tensorboard \
  --bf16 \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

If the CUDA device does not support bf16, disable it:

```bash
--no-bf16
```

### 6. Larger training run

Only launch a long run after tokenizer validation, debug training, checkpoint reload, and the CUDA pilot run pass.

```bash
uv run accelerate launch -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/selfies_main \
  --device_backend cuda \
  --model_size base \
  --max_seq_length 512 \
  --max_steps 150000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --eval_size 100000 \
  --max_eval_batches 128 \
  --mlm_probability 0.30 \
  --learning_rate 1e-4 \
  --logging_steps 50 \
  --eval_steps 5000 \
  --save_steps 5000 \
  --save_total_limit 3 \
  --num_workers 4 \
  --report_to tensorboard \
  --bf16 \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

### 7. Select the best run from a sweep

After running multiple training jobs under a shared directory, rank them by evaluation metric:

```bash
uv run python -m modernmolbert.select_pretraining_run \
  --run_root runs/my_lr_sweep \
  --metric eval_loss \
  --lower_is_better \
  --output_report runs/my_lr_sweep/report.md \
  --copy_best_to runs/best_model
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--run_root` | required | Directory containing one subdirectory per run |
| `--metric` | `eval_loss` | Metric to rank by |
| `--lower_is_better` / `--no-lower_is_better` | `True` | Direction of the metric |
| `--require_complete` | `False` | Skip runs that did not reach `max_steps` |
| `--output_report` | — | Write a Markdown summary to this path |
| `--output_csv` | — | Write the full ranked table as CSV |
| `--output_json` | — | Write the full ranked table as JSON |
| `--copy_best_to` | — | Copy `final_model/` of the best run to this path |

A run is discovered if its subdirectory contains `run_args.json` or `trainer_state.json`. Hidden directories are skipped. If `--require_complete` is set, only runs where `global_step >= max_steps` and a `final_model/` directory exists are ranked.

The Markdown report (`--output_report`) contains a ranked overview table, best-run hyperparameters, and all evaluation metrics.

## Training output layout

Each training run writes:

```text
output_dir/
  run_args.json
  run_metadata.json
  tokenizer_metadata.json
  vocab.json
  ape_tokenizer/
    vocab.json
    tokenizer_config.json
    special_tokens_map.json
    tokenization_ape.py
  README.checkpoint.md
  checkpoint-*/
  final_model/
    config.json
    model.safetensors
    tokenizer_metadata.json
    vocab.json
    ape_tokenizer/
      vocab.json
      tokenizer_config.json
      special_tokens_map.json
      tokenization_ape.py
    README.checkpoint.md
```

`final_model/` should be self-contained for inference and frozen featurisation, aside from installed package dependencies.

## Reload a trained checkpoint

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("runs/cuda_base_pilot_512/final_model")
tokenizer = AutoTokenizer.from_pretrained(
    "runs/cuda_base_pilot_512/final_model/ape_tokenizer",
    trust_remote_code=True,
)

batch = tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print(out.logits.shape)
```

## Use or train the tokenizer

Load a saved APE tokenizer through the standard Transformers entry point:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "path/to/final_model/ape_tokenizer",
    trust_remote_code=True,
)

tokens = tokenizer.tokenize("[C][C][O]")
ids = tokenizer.convert_tokens_to_ids(tokens)
batch = tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")
print(tokens)
print(ids)
print(batch["input_ids"])
```

Train a new SELFIES APE tokenizer on your own in-memory data:

```python
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

corpus = [
    "[C][C][O]",
    "[C][O][C]",
    "[C][C][=Branch1][C][=O][O]",
]

tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
tokenizer.train(
    corpus,
    representation="SELFIES",
    max_vocab_size=5000,
    min_freq_for_merge=2,
)
tokenizer.save_pretrained("my_selfies_ape_tokenizer")
```

For SMILES data, set `representation="SMILES"`:

```python
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

tokenizer = APEPreTrainedTokenizer(representation="SMILES")
tokenizer.train(
    ["CCO", "CCN", "c1ccccc1"],
    representation="SMILES",
    max_vocab_size=5000,
    min_freq_for_merge=2,
)
tokenizer.save_pretrained("my_smiles_ape_tokenizer")
```

The CLI also supports local parquet files:

```bash
uv run python -m modernmolbert.train_ape_tokenizer \
  --data_files "data/my_selfies/*.parquet" \
  --selfies_column SELFIES \
  --output_vocab_path tokenizer/my_selfies_ape_tokenizer.json \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2
```

Then load the saved directory with `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.


## Collator

`MolecularMLMCollator` is a `DataCollatorMixin` subclass used as the MLM data collator
during pretraining. Pass it directly to `Trainer` or any PyTorch `DataLoader`.

```python
from modernmolbert.collator import MolecularMLMCollator

collator = MolecularMLMCollator(
    pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=len(tokenizer),
    mlm_probability=0.15,
    special_token_ids=[tokenizer.bos_token_id, tokenizer.eos_token_id,
                       tokenizer.pad_token_id, tokenizer.unk_token_id,
                       tokenizer.mask_token_id],
    masking_strategy="hetero_span",          # "standard" | "span" | "hetero_span"
    ids_to_tokens=tokenizer.ids_to_tokens,   # required for hetero_span weight table
)

# Direct use
batch = collator([{"input_ids": [0, 5, 6, 7, 8, 2]},
                  {"input_ids": [0, 9, 10, 2]}])
# -> {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}

# Or pass to Trainer
trainer = Trainer(..., data_collator=collator)
```

Three masking strategies are available:

| Strategy | Description |
|---|---|
| `standard` | Independent Bernoulli per eligible token (original BERT) |
| `span` | Contiguous spans sampled from Geometric(`span_p_geom`), clamped to `span_max_length` |
| `hetero_span` | Span masking with start positions weighted toward heteroatom-containing tokens |

All strategies apply the BERT corruption rule after position selection: 80% `<mask>`,
10% random token, 10% unchanged. Special tokens and padding are never masked.

See [docs/masking_strategies.md](docs/masking_strategies.md) for design rationale.

## Evaluation workflow

ModernMolBERT is evaluated as a frozen molecular featuriser. All featurisers share the same benchmark machinery:

```text
SMILES dataset
-> RepresentationFeaturizer
-> FeatureBatch(X, valid_mask, metadata)
-> cached features
-> downstream model
-> shared metrics
-> results.csv / manifest.json
-> summary tables and plots
```

Prepare datasets as described in [docs/datasets.md](docs/datasets.md), then run a benchmark suite:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core \
  --overwrite
```

Generate reports:

```bash
uv run python -m modernmolbert.eval.cli.report_benchmark_results \
  --results_csv outputs/eval/pilot_core/results.csv \
  --output_dir outputs/eval/pilot_core/report
```

Expected output layout:

```text
outputs/eval/pilot_core/
  results.csv
  skipped_tasks.csv        # only if tasks were skipped
  manifest.json
  cache/
  report/
    tables/
      summary.csv
      <metric>_matrix.csv
      <metric>_average_rank.csv
    plots/
      <metric>_by_dataset.png
```

## Feature caching

By default, benchmark suites write cached features to:

```text
<output_dir>/cache
```

To share cached features across runs, pass a shared cache directory:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite \
  --suite configs/eval_suites/pilot_core.yaml \
  --output_dir outputs/eval/pilot_core_rerun \
  --cache_dir outputs/eval/shared_feature_cache \
  --overwrite
```

The cache key depends on dataset split, ordered molecule values, molecule column name, and featurizer identity/config. It does not depend on downstream model or seed, so cached embeddings can be reused across downstream models and repeated suite runs.

## ModernMolBERT featurizer config

```json
{
  "type": "modernmolbert_selfies",
  "name": "modernmolbert_pilot",
  "model_dir": "runs/pubchem10m_mps_base_pilot_256/final_model",
  "tokenizer_path": "runs/pubchem10m_mps_base_pilot_256/final_model",
  "max_seq_length": 256,
  "pooling": "mean",
  "device": "auto",
  "batch_size": 32
}
```

The ModernMolBERT featurizer accepts SMILES as input, converts valid SMILES to SELFIES internally, tokenizes with the trained APE tokenizer, and returns pooled encoder representations.

Other featurizers and external baselines are documented in [docs/baselines.md](docs/baselines.md).


## TensorBoard

Enable TensorBoard logging with:

```bash
--report_to tensorboard
```

Then run:

```bash
uv run tensorboard --logdir runs
```

or for a specific run:

```bash
uv run tensorboard --logdir runs/cuda_base_pilot_512
```

## Path handling in examples and notebooks

Examples should resolve files relative to the repository root rather than the current working directory. This avoids broken paths when notebooks are launched from `examples/`, `notebooks/`, VS Code, or Jupyter.

```python
from modernmolbert.paths import data_path, outputs_path

dataset_dir = data_path("eval", "moleculenet_sanitized", "bbbp")
output_dir = outputs_path("examples", "ecfp4_moleculenet")
```

## Troubleshooting

### MPS DataLoader workers crash

Use:

```bash
--num_workers 0
```

### High truncation rate

Increase sequence length:

```bash
--max_seq_length 512
```

Also inspect tokenized length percentiles in the tokenizer validation output.

### CUDA bf16 failure

Disable bf16:

```bash
--no-bf16
```

or use a CUDA device that supports bf16.

### TensorBoard is not logging

Ensure the run uses:

```bash
--report_to tensorboard
```

and that TensorBoard is installed through `uv sync`.

### Checkpoint reload fails

Test the final model directory directly:

```bash
uv run python - <<'PY'
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("runs/debug_selfies/final_model")
tok = AutoTokenizer.from_pretrained(
    "runs/debug_selfies/final_model/ape_tokenizer",
    trust_remote_code=True,
)
batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print("reload ok", out.logits.shape)
PY
```


## Development principles

- Keep tokenizer training separate from model training.
- Treat tokenizer validation as a mandatory gate.
- Copy tokenizer artifacts into every final model directory.
- Evaluate ModernMolBERT as a normal featuriser.
- Use one benchmark pipeline for all frozen representations.
- Prefer benchmark suites over one-off scripts for reproducible results.
- Avoid adding new benchmark scope until the core tokenizer, model, featuriser, and runner path are stable.
- Prefer small smoke tests before expensive training.
- Record enough metadata to reproduce every run.
- Treat `results.csv`, `manifest.json`, and report tables as the canonical outputs of a benchmark run.

## Citation

If you use this work, please cite the accompanying paper. See [`CITATION.cff`](CITATION.cff) or use the “Cite this repository” button on GitHub.

## References

**Span and hetero-span masking** take inspiration from:

> Peng, T., Li, Y., Li, X., Bian, J., Xie, Z., Sui, N., Mumtaz, S., Xu, Y., Kong, L., & Xiong, H. (2025).
> Pre-trained molecular language models with random functional group masking.
> *npj Artificial Intelligence*, 1, 28.
> https://doi.org/10.1038/s44387-025-00029-3

**APE tokenizer** is based on:

> Leon, M., Perezhohin, Y., Peres, F. et al. (2024).
> Comparing SMILES and SELFIES tokenization for enhanced chemical language modeling.
> *Scientific Reports*, 14, 25016.
> https://doi.org/10.1038/s41598-024-76440-8
