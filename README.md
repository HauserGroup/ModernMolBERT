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
- **Tokenizer:** custom `APETokenizer`, trained separately from model training.
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
  ape_tokenizer.py                     # SELFIES APE tokenizer
  train_ape_tokenizer.py               # tokenizer training CLI
  validate_tokenizer.py                # tokenizer validation CLI
  train_selfies_ape_modernbert.py      # MLM pretraining CLI
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

## Training workflow

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

## Training output layout

Each training run writes:

```text
output_dir/
  run_args.json
  run_metadata.json
  tokenizer.json
  tokenizer_metadata.json
  README.checkpoint.md
  checkpoint-*/
  final_model/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_metadata.json
    README.checkpoint.md
```

`final_model/` should be self-contained for inference and frozen featurisation, aside from installed package dependencies.

## Reload a trained checkpoint

```python
from transformers import AutoModelForMaskedLM
from modernmolbert.ape_tokenizer import APETokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("runs/cuda_base_pilot_512/final_model")
tokenizer = APETokenizer.from_pretrained("runs/cuda_base_pilot_512/final_model")

batch = tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print(out.logits.shape)
```

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

The ModernMolBERT featurizer accepts SMILES as input, converts valid SMILES to SELFIES internally, tokenizes with the trained `APETokenizer`, and returns pooled encoder representations.

Other featurizers and external baselines are documented in [docs/baselines.md](docs/baselines.md).

## Optional one-off frozen benchmark

For ad hoc debugging, a one-off frozen benchmark CLI may be used:

```bash
uv run python -m modernmolbert.eval.cli.run_frozen_benchmark \
  --name modernmolbert_smoke \
  --task_type classification \
  --task_names label \
  --train_csv path/to/train.csv \
  --test_csv path/to/test.csv \
  --featurizer_config configs/featurizers/modernmolbert_selfies.json \
  --output_dir outputs/eval/modernmolbert_smoke \
  --cache_dir outputs/eval/cache \
  --batch_size 32
```

For reproducible benchmark results, prefer the suite runner:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite
```

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
from transformers import AutoModelForMaskedLM
from modernmolbert.ape_tokenizer import APETokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("runs/debug_selfies/final_model")
tok = APETokenizer.from_pretrained("runs/debug_selfies/final_model")
batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print("reload ok", out.logits.shape)
PY
```

### Benchmark results differ between scripts

Use the shared benchmark suite runner for benchmark results:

```bash
uv run python -m modernmolbert.eval.cli.run_benchmark_suite
```

Avoid comparing results from older direct embedding/sklearn paths.

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
