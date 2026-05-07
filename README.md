# ModernMolBERT SELFIES Training

This repository trains a ModernBERT masked-language model on SELFIES strings only.

Core policy:
- Tokenizer training is a separate step.
- Model training only loads a vetted tokenizer + metadata pair.
- Every run copies tokenizer artifacts into both `output_dir/` and `final_model/`.

## Scope

- Dataset: `mikemayuare/PubChem10M_SMILES_SELFIES`
- Representation: `SELFIES` only
- Tokenizer: `APETokenizer`
- Model objective: MLM

## Canonical tokenizer artifacts

- `tokenizer/selfies_ape_tokenizer.json`
- `tokenizer/selfies_ape_tokenizer.metadata.json`

The metadata includes representation and SHA256; training validates both.

## Installation

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv sync
```

## Commands

### 1) Train/update tokenizer (separate stage)

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

### 3) Debug training run

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --debug \
  --output_dir runs/debug_selfies \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

### 4) Larger training run

```bash
uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/selfies_main \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json \
  --max_steps 150000 \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --report_to tensorboard
```

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
| `--tokenizer_vocab_path` | `tokenizer/selfies_ape_tokenizer.json` | Canonical SELFIES tokenizer vocabulary |
| `--tokenizer_metadata_path` | `<vocab>.metadata.json` | Metadata with representation/hash checks |
| `--unk_rate_threshold` | `0.001` | Fail if unknown-token rate is too high |
| `--max_eval_batches` | `20` | Cap evaluation size for memory safety |
| `--report_to` | `none` | Logging backend (`none` or `tensorboard`) |
| `--val_split_mod` | `100` | Deterministic non-overlapping split modulus |
| `--val_split_bucket` | `0` | Validation bucket for deterministic split |

## Output layout

Each run writes:

```text
output_dir/
  run_args.json
  ape_tokenizer_metadata.json
  tokenizer.json
  tokenizer_metadata.json
  README.checkpoint.md
  final_model/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_metadata.json
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
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```
