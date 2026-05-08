# ModernMolBERT SELFIES Training

This repository trains a ModernBERT masked-language model on SELFIES strings only.

Core policy:

- Tokenizer training is a separate step.
- Model training only loads a vetted tokenizer + metadata pair.
- Every run copies tokenizer artifacts into both `output_dir/` and `final_model/`.
- The public model-size interface is restricted to official ModernBERT sizes: `base` and `large`.

## Scope

- Dataset: `mikemayuare/PubChem10M_SMILES_SELFIES`
- Representation: `SELFIES` only
- Tokenizer: `APETokenizer`
- Model objective: masked language modeling (MLM)
- Model sizes:
  - `base`: official `answerdotai/ModernBERT-base` architecture
  - `large`: official `answerdotai/ModernBERT-large` architecture

The model is trained from scratch with the SELFIES tokenizer vocabulary. The Answer.AI checkpoints are used only as architecture/config references, not as English/code pretrained weights.

## Canonical tokenizer artifacts

- `tokenizer/selfies_ape_tokenizer.json`
- `tokenizer/selfies_ape_tokenizer.metadata.json`

The metadata includes the tokenizer representation and SHA256 hash. Training validates both before model construction.

## Installation

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv sync
```

## Commands

### 1) Train/update tokenizer (separate stage)

Tokenizer training is intentionally separate from model training.

```bash
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --dataset_name mikemayuare/PubChem10M_SMILES_SELFIES \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000
```

### 2) Validate tokenizer (mandatory gate)

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --n 1000
```

Example expected output:

```text
representation: SELFIES
vocab_size: ...
unk_rate: ...
mean_len: ...
p95_len: ...
truncation_rate@256: ...
special_ids: ...
```

### 2b) Inspect one row from zpn/zinc20

```bash
uv run python - <<'PY'
from datasets import load_dataset

row = next(iter(load_dataset("zpn/zinc20", split="train", streaming=True)))
print("keys:", list(row.keys()))
print("selfies:", row["selfies"][:200])
PY
```

### 2c) Validate tokenizer on zpn/zinc20

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --dataset_name zpn/zinc20 \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --n 1000
```

### 3) Debug training run

This is a tiny end-to-end check. It validates the training loop, model save, tokenizer artifact copying, and final reload path.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --model_size base \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

### 4) Slightly larger MPS smoke run

This is larger than `--debug`, but still intended as a Mac smoke test rather than a useful model.

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

### 4b) Short zpn/zinc20 base pilot

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/zinc20_base_pilot_256 \
  --dataset_name zpn/zinc20 \
  --use_validation_split \
  --model_size base \
  --max_seq_length 256 \
  --max_steps 200 \
  --eval_size 64 \
  --max_eval_batches 4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_workers 0 \
  --report_to tensorboard \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

Follow with TensorBoard:

```bash
uv run tensorboard --logdir runs/mps_base_smoke_512
```

### 5) CUDA pilot run

This is the first meaningful pilot run. It is still much smaller than a final pretraining run, but large enough to check learning dynamics, throughput, checkpointing, and evaluation behavior.

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

Follow with TensorBoard:

```bash
uv run tensorboard --logdir runs/cuda_base_pilot_512
```

### 6) Larger training run

Use this only after the readiness gate and pilot run pass.

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
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

## Accelerate and FlashAttention

For CUDA pilot and long runs, prefer Accelerate:

```bash
uv run accelerate launch -m modernmolbert.train_selfies_ape_modernbert ...
```

For multi-GPU CUDA training, configure Accelerate first:

```bash
uv run accelerate config

```

FlashAttention is optional and CUDA-oriented. Do not install it for Mac MPS.
For CUDA environments, install it only if supported by your PyTorch/CUDA/GPU setup
ModernBERT can run without FlashAttention, but CUDA training/inference may be slower and use more memory.

```bash
uv pip install flash-attn --no-build-isolation
```




## Model-size selection

Only two official model-size choices are supported:

```bash
--model_size base
--model_size large
```

`base` uses the official ModernBERT-base architecture. `large` uses the official ModernBERT-large architecture.

The training script should not expose arbitrary architecture knobs such as hidden size, number of layers, attention heads, or intermediate size.

For development and pilot runs, start with:

```bash
--model_size base
```

Do not start with `large` until the pipeline, tokenizer, checkpoint reload, and CUDA pilot run are stable.

## SMILES to SELFIES inference conversion

Checkpoints produced here expect SELFIES input. Convert SMILES before tokenization.

```python
import selfies as sf


def smiles_to_selfies(smiles: str) -> str:
    # Keep conversion behavior explicit so invalid inputs can be handled upstream.
    return sf.encoder(smiles)
```

Guidance:

- Canonicalize SMILES in your serving pipeline if deterministic behavior matters.
- Preserve stereochemistry upstream; conversion should operate on stereochemically complete strings.
- Handle invalid SMILES explicitly (exception/log/drop) before tokenizer use.

## Important options

| Option | Default | Purpose |
|---|---:|---|
| `--model_size` | `base` | Official ModernBERT architecture size: `base` or `large` |
| `--tokenizer_vocab_path` | `tokenizer/selfies_ape_tokenizer.json` | Canonical SELFIES tokenizer vocabulary |
| `--tokenizer_metadata_path` | `<vocab>.metadata.json` | Metadata with representation/hash checks |
| `--unk_rate_threshold` | `0.001` | Fail if unknown-token rate is too high |
| `--max_seq_length` | script default | Maximum tokenized SELFIES length used by the model |
| `--max_eval_batches` | `20` | Cap evaluation batches for memory safety |
| `--report_to` | `none` | Logging backend: `none` or `tensorboard` |
| `--val_split_mod` | `100` | Deterministic non-overlapping split modulus |
| `--val_split_bucket` | `0` | Validation bucket for deterministic split |
| `--device_backend` | `auto` | Runtime backend: `auto`, `cuda`, `mps`, or `cpu` |
| `--num_workers` | script default | DataLoader workers; use `0` on Mac/MPS |

## Following training

### Console output

The script prints:

- backend and precision mode
- tokenizer vocabulary size
- special token IDs
- tokenizer validation statistics
- dataset/tokenization preview
- model size and parameter count
- training plan
- checkpoint and final model locations

### TensorBoard

Enable TensorBoard with:

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

### Checkpointing

The script checkpoints automatically when these options are set:

```bash
--save_steps 1000
--save_total_limit 3
```

This creates intermediate checkpoints:

```text
output_dir/
  checkpoint-1000/
  checkpoint-2000/
  checkpoint-3000/
  ...
```

Only the most recent `save_total_limit` checkpoints are kept.

At the end of training, the script writes:

```text
output_dir/final_model/
```

## Output layout

Each run writes:

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

`final_model/` should be self-contained for inference, aside from installed Python package dependencies.

## Reloading a checkpoint

```python
from transformers import AutoModelForMaskedLM
from modernmolbert.ape_tokenizer import APETokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("runs/cuda_base_pilot_512/final_model")

tokenizer = APETokenizer()
tokenizer.load_vocabulary("runs/cuda_base_pilot_512/final_model/tokenizer.json")

batch = tokenizer("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print(out.logits.shape)
```

## Readiness gate

Do not launch long training until all pass:

```bash
uv run pytest
uv run ruff check .

uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --n 1000

uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --model_size base \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

## Optional smoke test on Mac

The optional pytest smoke test is skipped unless explicitly enabled.

```bash
MODERNMOLBERT_RUN_SMOKE=1 MODERNMOLBERT_RUN_MPS=1 \
  uv run pytest -m "smoke and mps" -s
```

The smoke test is intentionally tiny. It validates the full pipeline but does not train a useful model.

## Troubleshooting

### MPS DataLoader workers crash

Use:

```bash
--num_workers 0
```

### High truncation rate

Increase:

```bash
--max_seq_length 512
```

or inspect tokenized length percentiles in the tokenizer validation output.

### TensorBoard is not logging

Ensure the run uses:

```bash
--report_to tensorboard
```

and that `tensorboard` is installed through `uv sync`.

### Reload fails because of config serialization

The training script should save a reloadable `final_model/`. Always test:

```bash
uv run python - <<'PY'
from transformers import AutoModelForMaskedLM
from modernmolbert.ape_tokenizer import APETokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("runs/debug_selfies/final_model")
tok = APETokenizer()
tok.load_vocabulary("runs/debug_selfies/final_model/tokenizer.json")
batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
print("reload ok", out.logits.shape)
PY
```

## MoLFormer evaluation baseline

MoLFormer is included as a frozen SMILES-encoder baseline. In the main benchmark, its embeddings should be evaluated with the same downstream models as the other representations:

- classification: logistic regression
- regression: ridge / RidgeCV

We do not fine-tune MoLFormer for the primary benchmark.

### Checkpoint

Use the pinned Hugging Face checkpoint:

```text
ibm-research/MoLFormer-XL-both-10pct
revision: 7b12d946c181a37f6012b9dc3b002275de070314
```

The checkpoint requires `trust_remote_code=True`, so the revision is pinned for reproducibility.

### Separate MoLFormer-only environment

MoLFormer currently needs a Transformers 4.x environment because its Hugging Face remote code depends on older Transformers APIs. Keep this separate from the main ModernMolBERT training environment.

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

Do not install the package with `pip install -e .` in this environment. Instead, run MoLFormer commands from the repo root with:

```bash
export PYTHONPATH="$PWD/src"
```

### Featurizer config

Create `configs/featurizers/molformer_xl_both_10pct_cpu.json`:

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

### Smoke test

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

### Optional pytest

MoLFormer tests are skipped by default. Run them explicitly inside the `molformer-only` environment:

```bash
PYTHONPATH="$PWD/src" MODERNMOLBERT_RUN_MOLFORMER_TESTS=1 \
  python -m pytest tests/test_eval_molformer.py -q -s
```

A registry-only test can be run without loading the model:

```bash
PYTHONPATH="$PWD/src" python -m pytest tests/test_eval_molformer.py -q -k registry
```

Register these markers in `pyproject.toml` if they are not already present:

```toml
[tool.pytest.ini_options]
markers = [
    "smoke: optional smoke tests",
    "mps: tests requiring Apple MPS",
    "model: tests that load trained model checkpoints or external pretrained models",
    "molformer: tests requiring the separate molformer-only environment",
]
```

### CLI example

```bash
PYTHONPATH="$PWD/src" python -m modernmolbert.eval.cli.run_frozen_benchmark \
  --name tiny_molformer_demo \
  --task_type classification \
  --task_names label \
  --train_csv tmp_eval/train.csv \
  --test_csv tmp_eval/test.csv \
  --featurizer_config configs/featurizers/molformer_xl_both_10pct_cpu.json \
  --output_dir tmp_eval/results_molformer \
  --cache_dir tmp_eval/cache \
  --batch_size 4
```

Notes:

- Use this environment only for MoLFormer evaluation.
- Use the main `uv`/ModernMolBERT environment for training and normal tests.
- MoLFormer consumes SMILES, not SELFIES.
- The embedding is mean-pooled from `last_hidden_state` using the attention mask.
