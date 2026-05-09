# ModernMolBERT
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<!-- ![ModernMolBERT](imgs/mmbert_text.png) -->

<img src="imgs/mmbert_text.png" width="600"/>.


ModernMolBERT (Modern Molecular BERT) is a SELFIES-native molecular representation project. It trains a ModernBERT masked-language model on SELFIES strings and evaluates the resulting encoder as a frozen molecular featuriser against existing molecular representations through one shared benchmark pipeline.

The core objective is not to maintain a collection of separate scripts. The core objective is to support a reliable path from tokenizer training to model pretraining to frozen-feature benchmarking:

```text
SELFIES tokenizer
-> ModernBERT masked-language pretraining
-> saved checkpoint
-> ModernMolBERTSelfiesFeaturizer
-> FrozenBenchmarkRunner
-> shared downstream models and metrics
```

## Project scope

ModernMolBERT focuses on the following:

- **Input representation:** SELFIES.
- **Tokenizer:** custom `APETokenizer` trained separately from model training.
- **Model:** ModernBERT architecture trained from scratch with the SELFIES tokenizer vocabulary.
- **Objective:** masked language modeling.
- **Primary use:** frozen molecular featurisation.
- **Primary benchmark path:** shared frozen benchmark runner used for ModernMolBERT and baselines.

The Answer.AI ModernBERT checkpoints are used as architecture/config references. The model is not initialized from English/code pretrained weights for the main SELFIES pretraining run.

Supported public model-size choices are:

```bash
--model_size base
--model_size large
```

Use `base` for development, debugging, and pilot runs. Do not use `large` until the tokenizer, training loop, checkpoint reload, featuriser, and frozen benchmark path are stable.

---

## Repository layout

```text
src/modernmolbert/
  ape_tokenizer.py                     # SELFIES APE tokenizer
  train_ape_tokenizer.py               # tokenizer training CLI
  validate_tokenizer.py                # tokenizer validation CLI
  train_selfies_ape_modernbert.py      # MLM pretraining CLI
  utils.py                             # shared dataset/tokenizer helpers

src/modernmolbert/eval/
  featurizers/                         # frozen representation featurisers
    base.py                            # FeatureBatch and protocol
    rdkit_ecfp.py                      # ECFP4 baseline
    hf_smiles.py                       # Hugging Face SMILES encoder baseline
    modernmolbert_selfies.py           # trained ModernMolBERT SELFIES featuriser
  cache.py                             # feature cache
  datasets.py                          # evaluation dataset loading
  downstream.py                        # shared downstream learners
  metrics.py                           # shared metrics
  pooling.py                           # shared pooling utilities
  registry.py                          # config-driven featuriser registry
  runner.py                            # canonical frozen benchmark runner
  moleculenet.py                       # MoleculeNet preparation utilities
  cli/
    run_frozen_benchmark.py            # canonical benchmark CLI
    run_modernmolbert_eval.py          # optional wrapper around shared runner

configs/featurizers/
  ecfp4_2048.json
  chemberta_77m_mlm.json
  molformer_xl_both_10pct_cpu.json
  modernmolbert_selfies.json

tests/
  ...
```

The retired early evaluation helpers should not be used for benchmark results. ModernMolBERT should be evaluated through `ModernMolBERTSelfiesFeaturizer` and `FrozenBenchmarkRunner`, not through a separate embedding/sklearn path.

---

## Installation

Create a Python 3.13 environment and install the project dependencies with `uv`:

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

---

## Canonical tokenizer artifacts

Tokenizer training is intentionally separate from model training. Model training should only use a validated tokenizer plus matching metadata.

Canonical artifacts:

```text
tokenizer/selfies_ape_tokenizer.json
tokenizer/selfies_ape_tokenizer.metadata.json
```

The metadata records the tokenizer representation, vocabulary hash, special token IDs, and training provenance. The training script validates the metadata and SHA256 hash before model construction.

---

## 1. Train the SELFIES tokenizer

Example tokenizer training command:

```bash
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --dataset_name mikemayuare/PubChem10M_SMILES_SELFIES \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000
```

The tokenizer stage should produce:

```text
tokenizer/selfies_ape_tokenizer.json
tokenizer/selfies_ape_tokenizer_freq.json
tokenizer/selfies_ape_tokenizer.metadata.json
```

---

## 2. Validate the tokenizer

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

For an alternate dataset such as `zpn/zinc20`, first inspect one row:

```bash
uv run python - <<'PY'
from datasets import load_dataset

row = next(iter(load_dataset("zpn/zinc20", split="train", streaming=True)))
print("keys:", list(row.keys()))
print("selfies:", row["selfies"][:200])
PY
```

Then validate:

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --dataset_name zpn/zinc20 \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --n 1000
```

---

## 3. Debug model training

The debug run is a tiny end-to-end check. It validates:

- tokenizer metadata validation
- dataset loading
- MLM collator behavior
- model construction
- training loop
- checkpoint save
- tokenizer artifact copying
- final model reloadability

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --model_size base \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

Use this before any pilot or long run.

---

## 4. MPS smoke run

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

---

## 5. CUDA pilot run

This is the first meaningful training pilot. It is still smaller than a final pretraining run, but it is large enough to check learning dynamics, throughput, checkpointing, and evaluation behavior.

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

View logs:

```bash
uv run tensorboard --logdir runs/cuda_base_pilot_512
```

---

## 6. Larger training run

Only launch a long run after the readiness gate, debug run, checkpoint reload test, and CUDA pilot run pass.

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

---

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

`final_model/` should be self-contained for inference and frozen featurisation, aside from installed Python package dependencies.

---

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

---

## SMILES to SELFIES conversion

Checkpoints produced by this project expect SELFIES input. Benchmark datasets may store SMILES, so the ModernMolBERT featuriser converts SMILES to SELFIES before tokenization.

```python
import selfies as sf


def smiles_to_selfies(smiles: str) -> str:
    return sf.encoder(smiles)
```

Guidance:

- Canonicalize SMILES upstream if deterministic behavior matters.
- Preserve stereochemistry upstream; conversion should operate on stereochemically complete strings.
- Handle invalid SMILES explicitly.
- Benchmark featurisers should mark invalid molecules through `FeatureBatch.valid_mask`.

---

# Frozen benchmark evaluation

All frozen-representation evaluations should use the shared benchmark runner.

This is the canonical architecture:

```text
SMILES dataset
-> RepresentationFeaturizer
-> FeatureBatch(X, valid_mask, metadata)
-> FrozenBenchmarkRunner
-> shared downstream learner
-> shared metrics
-> results.json / results.csv
```

ModernMolBERT is evaluated as a featuriser, not through a separate embedding script.

This ensures ModernMolBERT and baselines share:

- the same invalid-molecule handling
- the same label masking
- the same downstream models
- the same standardization behavior
- the same metrics
- the same cache semantics
- the same output format

## ModernMolBERT featuriser config

Canonical config:

```json
{
  "batch_size": 32,
  "device": "auto",
  "max_seq_length": 256,
  "model_dir": "outputs/modernmolbert/final_model",
  "pooling": "mean",
  "tokenizer_path": "outputs/modernmolbert/final_model",
  "type": "modernmolbert_selfies"
}
```

Save this as:

```text
configs/featurizers/modernmolbert_selfies.json
```

`model_dir` and `tokenizer_path` may point to the same `final_model/` directory if tokenizer artifacts were copied there during training.

## Run the canonical frozen benchmark

Example classification benchmark:

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

Example regression benchmark:

```bash
uv run python -m modernmolbert.eval.cli.run_frozen_benchmark \
  --name esol_modernmolbert \
  --task_type regression \
  --task_names solubility \
  --train_csv data/eval/esol/train.csv \
  --test_csv data/eval/esol/test.csv \
  --featurizer_config configs/featurizers/modernmolbert_selfies.json \
  --output_dir outputs/eval/esol_modernmolbert \
  --cache_dir outputs/eval/cache \
  --batch_size 32
```

Use the same runner for ECFP4, Hugging Face SMILES encoders, MoLFormer, and ModernMolBERT.

## Optional ModernMolBERT wrapper CLI

If retained, the ModernMolBERT-specific CLI should be only a wrapper around `FrozenBenchmarkRunner`.

Example:

```bash
uv run python -m modernmolbert.eval.cli.run_modernmolbert_eval \
  --dataset_dir data/eval/moleculenet_sanitized/bbbp \
  --model_dir outputs/modernmolbert/final_model \
  --tokenizer_path outputs/modernmolbert/final_model \
  --output_dir outputs/eval/bbbp_modernmolbert \
  --cache_dir outputs/eval/cache \
  --eval_split test \
  --pooling mean \
  --batch_size 32
```

This wrapper must delegate to the same shared benchmark machinery. It should not implement its own embedding, sklearn, masking, or metric path.

---

## MoleculeNet preparation

Prepared MoleculeNet datasets should contain train/valid/test splits and metadata. The benchmark runner can then consume CSV or prepared parquet-style datasets depending on the CLI used.

Typical prepared layout:

```text
data/eval/moleculenet_sanitized/<dataset>/
  metadata.json
  train.parquet
  valid.parquet
  test.parquet
```

The important requirement is that every featuriser sees the same molecules, labels, and splits.

---

## Supported featurisers

### ModernMolBERT SELFIES

```json
{
  "type": "modernmolbert_selfies",
  "model_dir": "outputs/modernmolbert/final_model",
  "tokenizer_path": "outputs/modernmolbert/final_model",
  "max_seq_length": 256,
  "pooling": "mean",
  "device": "auto",
  "batch_size": 32
}
```

### ECFP4

```json
{
  "type": "ecfp4",
  "name": "ecfp4_2048",
  "n_bits": 2048,
  "radius": 2
}
```

### Hugging Face SMILES encoder

```json
{
  "type": "hf_smiles",
  "name": "chemberta_77m_mlm",
  "model_name_or_path": "DeepChem/ChemBERTa-77M-MLM",
  "max_seq_length": 256,
  "pooling": "mean",
  "device": "auto",
  "trust_remote_code": false
}
```

---

## MoLFormer baseline

MoLFormer is included as a frozen SMILES-encoder baseline. It should be evaluated with the same downstream learners as every other representation:

- classification: logistic regression
- regression: ridge or RidgeCV

Do not fine-tune MoLFormer for the primary frozen benchmark.

Pinned checkpoint:

```text
ibm-research/MoLFormer-XL-both-10pct
revision: 7b12d946c181a37f6012b9dc3b002275de070314
```

MoLFormer requires `trust_remote_code=True`, so the revision is pinned for reproducibility.

### Separate MoLFormer-only environment

MoLFormer currently needs a Transformers 4.x environment because its remote code depends on older Transformers APIs. Keep this environment separate from the main ModernMolBERT training environment.

Create `environment-molformer-only.yml`:

```yaml
name: molformer-only
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scikit-learn
  - pytorch
  - pytest
  - pip:
      - "transformers>=4.38,<5"
      - "huggingface-hub<1.0"
      - "safetensors"
      - "tokenizers"
      - "tqdm"
```

Create and activate:

```bash
conda env create -f environment-molformer-only.yml
conda activate molformer-only
```

Do not install the package with `pip install -e .` in this environment. Instead, run from the repo root with:

```bash
export PYTHONPATH="$PWD/src"
```

### MoLFormer featuriser config

```json
{
  "type": "hf_smiles",
  "name": "molformer_xl_both_10pct",
  "model_name_or_path": "ibm-research/MoLFormer-XL-both-10pct",
  "revision": "7b12d946c181a37f6012b9dc3b002275de070314",
  "max_seq_length": 128,
  "pooling": "mean",
  "device": "cpu",
  "trust_remote_code": true
}
```

### MoLFormer smoke test

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
from modernmolbert.eval.featurizers.hf_smiles import HuggingFaceSmilesFeaturizer

f = HuggingFaceSmilesFeaturizer(
    name="molformer_xl_both_10pct",
    model_name_or_path="ibm-research/MoLFormer-XL-both-10pct",
    revision="7b12d946c181a37f6012b9dc3b002275de070314",
    max_seq_length=128,
    pooling="mean",
    device="cpu",
    trust_remote_code=True,
)

out = f.featurize_smiles(["CCO", "c1ccccc1", "CC(=O)O"], batch_size=2)

print("valid_mask", out.valid_mask)
print("X", out.X.shape, out.X.dtype)
print("metadata", out.metadata)
PY
```

Expected shape:

```text
valid_mask [ True  True  True]
X (3, 768) float32
```

Run optional tests inside the MoLFormer environment:

```bash
PYTHONPATH="$PWD/src" MODERNMOLBERT_RUN_MOLFORMER_TESTS=1 \
  python -m pytest tests/test_eval_molformer.py -q -s
```

---

## Readiness gate

Do not launch long training until all of the following pass.

```bash
uv run ruff check .
uv run pytest
```

Validate tokenizer:

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --n 1000
```

Run debug training:

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --model_size base \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

Reload the debug checkpoint:

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

Run focused core tests:

```bash
uv run pytest \
  tests/test_tokenizer_training.py \
  tests/test_collator.py \
  tests/test_training_cli.py \
  tests/test_tokenizer_validation.py \
  tests/test_smoke_training.py \
  tests/test_checkpoint_reload.py \
  tests/test_eval_modernmolbert_selfies.py \
  tests/test_frozen_benchmark_modernmolbert_smoke.py \
  -q
```

---

## Optional Mac smoke tests

The optional pytest smoke test is skipped unless explicitly enabled.

```bash
MODERNMOLBERT_RUN_SMOKE=1 MODERNMOLBERT_RUN_MPS=1 \
  uv run pytest -m "smoke and mps" -s
```

Use `--num_workers 0` for MPS runs.

---

## Important training options

| Option | Typical value | Purpose |
|---|---:|---|
| `--model_size` | `base` | Official ModernBERT architecture size: `base` or `large` |
| `--tokenizer_vocab_path` | `tokenizer/selfies_ape_tokenizer.json` | SELFIES tokenizer vocabulary |
| `--tokenizer_metadata_path` | `<vocab>.metadata.json` | Metadata with representation/hash checks |
| `--unk_rate_threshold` | `0.001` | Fail if unknown-token rate is too high |
| `--max_seq_length` | `256` or `512` | Maximum tokenized SELFIES length |
| `--max_eval_batches` | `20` or higher | Cap evaluation batches for memory safety |
| `--report_to` | `tensorboard` | Logging backend |
| `--val_split_mod` | `100` | Deterministic split modulus |
| `--val_split_bucket` | `0` | Validation bucket for deterministic split |
| `--device_backend` | `auto`, `cuda`, `mps`, `cpu` | Runtime backend |
| `--num_workers` | `0` on Mac/MPS | DataLoader workers |
| `--bf16` / `--no-bf16` | backend-dependent | Enable or disable bf16 |

---

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

---

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

Use only the shared benchmark runner for benchmark results:

```bash
python -m modernmolbert.eval.cli.run_frozen_benchmark
```

If `run_modernmolbert_eval.py` is retained, it must delegate to the shared runner. Do not compare results from a separate direct embedding/sklearn path.

---

## Preparing MoleculeNet evaluation data

MoleculeNet datasets are prepared locally from DeepChem into sanitized Parquet files containing raw SMILES, canonical SMILES, SELFIES, validity flags, and task labels.

The preparation pipeline intentionally loads each DeepChem dataset **unsplit**, sanitizes molecules first, converts valid molecules to canonical SMILES and SELFIES, and then applies a local train/validation/test split. This avoids DeepChem/RDKit scaffold splitting before invalid molecules have been removed.

### Prepare only the core suit.

```bash
uv run python -m modernmolbert.eval.cli.prepare_moleculenet \
  --split scaffold \
  --seed 13 \
  --frac_train 0.8 \
  --frac_valid 0.1 \
  --frac_test 0.1 \
  --output_root data/eval/moleculenet_sanitized \
  --deepchem_data_dir data/deepchem/raw \
  --deepchem_save_dir data/deepchem/processed

uv run python -m modernmolbert.eval.cli.prepare_moleculenet --list_datasets # list available
```

## Development principles

- Keep tokenizer training separate from model training.
- Treat tokenizer validation as a mandatory gate.
- Copy tokenizer artifacts into every final model directory.
- Evaluate ModernMolBERT as a normal featuriser.
- Use one benchmark runner for all frozen representations.
- Avoid adding new benchmark scope until the core tokenizer, model, featuriser, and runner path are stable.
- Prefer small smoke tests before expensive training.
- Record enough metadata to reproduce every run.

### Path handling in examples and notebooks

Examples should resolve files relative to the repository root rather than the current working directory. This avoids broken paths when notebooks are launched from `examples/`, `notebooks/`, VS Code, or Jupyter.

Use:

```python
from modernmolbert.paths import data_path, outputs_path

dataset_dir = data_path("eval", "moleculenet_sanitized", "bbbp")
output_dir = outputs_path("examples", "ecfp4_moleculenet")
```

## Citation
If you use this work, please cite the accompanying paper. See [`CITATION.cff`](CITATION.cff) or use the "Cite this repository" button on GitHub.
