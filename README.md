# ModernBERT + APE Tokenizer Molecular MLM Training

This directory contains a training script for masked-language-model pretraining on molecular strings using:

- `mikemayuare/PubChem10M_SMILES_SELFIES`
- `APETokenizer`
- `ModernBERT`
- SELFIES or SMILES as the molecular representation

The intended workflow is:

1. Run a small Mac MPS smoke test.
2. Confirm the tokenizer, dataset, model, loss, and checkpointing all work.
3. Move to CUDA for serious training.

## Files

- `train_selfies_ape_modernbert.py` — main training script.
- `README.md` — this document.

## Installation

### Mac MPS

Do not install `flash-attn` on Mac.

```bash
conda create -n molbert-mps python=3.11 -y
conda activate molbert-mps

pip install torch torchvision torchaudio
pip install "transformers>=4.48" datasets accelerate evaluate tqdm numpy tensorboard
pip install git+https://github.com/mikemayuare/apetokenizer.git
```

Check that MPS is available:

```bash
python - <<'PY'
import torch
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
PY
```

### CUDA

```bash
conda create -n molbert-cuda python=3.11 -y
conda activate molbert-cuda

pip install "torch>=2.2" "transformers>=4.48" datasets accelerate evaluate tqdm numpy tensorboard
pip install git+https://github.com/mikemayuare/apetokenizer.git

# Optional on supported NVIDIA GPUs
pip install flash-attn --no-build-isolation
```

## Quick Start

### Mac MPS debug run

This is only a smoke test.

```bash
python train_selfies_ape_modernbert.py \
  --output_dir ./runs/mps_debug_selfies_ape_modernbert \
  --device_backend mps \
  --debug \
  --representation SELFIES \
  --tokenizer_train_size 10000 \
  --eval_size 1000 \
  --hidden_size 128 \
  --num_hidden_layers 4 \
  --num_attention_heads 4 \
  --intermediate_size 512 \
  --max_seq_length 256 \
  --mlm_probability 0.30 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4
```

### Longer Mac MPS smoke test

```bash
python train_selfies_ape_modernbert.py \
  --output_dir ./runs/mps_selfies_ape_modernbert_tiny \
  --device_backend mps \
  --representation SELFIES \
  --tokenizer_train_size 50000 \
  --eval_size 2000 \
  --max_steps 5000 \
  --hidden_size 128 \
  --num_hidden_layers 4 \
  --num_attention_heads 4 \
  --intermediate_size 512 \
  --max_seq_length 256 \
  --mlm_probability 0.30 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --logging_steps 50 \
  --eval_steps 500 \
  --save_steps 1000 \
  --learning_rate 1e-4
```

### CUDA 15M-ish SELFIES model

```bash
accelerate launch train_selfies_ape_modernbert.py \
  --output_dir ./runs/cuda_selfies_ape_modernbert_15m \
  --device_backend cuda \
  --representation SELFIES \
  --tokenizer_train_size 2000000 \
  --eval_size 100000 \
  --max_steps 150000 \
  --hidden_size 256 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --intermediate_size 1024 \
  --max_seq_length 512 \
  --mlm_probability 0.30 \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --bf16
```

### Matched CUDA SMILES control

```bash
accelerate launch train_selfies_ape_modernbert.py \
  --output_dir ./runs/cuda_smiles_ape_modernbert_15m \
  --device_backend cuda \
  --representation SMILES \
  --tokenizer_train_size 2000000 \
  --eval_size 100000 \
  --max_steps 150000 \
  --hidden_size 256 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --intermediate_size 1024 \
  --max_seq_length 512 \
  --mlm_probability 0.30 \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --bf16
```

## Input Representation

The script supports two representations:

- `SELFIES`
- `SMILES`

If you train with:

```bash
--representation SELFIES
```

then the model expects SELFIES strings at inference time. It will not directly accept SMILES unless you convert them first.

Example:

```python
import selfies as sf

smiles = "CC(=O)Oc1ccccc1C(=O)O"
selfies_string = sf.encoder(smiles)
```

If you train with:

```bash
--representation SMILES
```

then the model expects SMILES strings.

## Option Reference

### Paths

| Option | Default | Meaning |
|---|---:|---|
| `--output_dir` | required | Directory for checkpoints, final model, tokenizer vocabulary, and metadata. |
| `--tokenizer_vocab_path` | `None` | Existing APE vocabulary JSON. If omitted, uses `output_dir/ape_<representation>_vocab.json`. |

### Dataset

| Option | Default | Meaning |
|---|---:|---|
| `--dataset_name` | `mikemayuare/PubChem10M_SMILES_SELFIES` | Hugging Face dataset name. |
| `--representation` | `SELFIES` | Which dataset column to train on: `SELFIES` or `SMILES`. |
| `--tokenizer_train_size` | `2000000` | Number of sequences used to train the APE tokenizer. |
| `--eval_size` | `100000` | Number of sequences used for finite validation. |
| `--shuffle_buffer_size` | `100000` | Streaming shuffle buffer. Larger improves shuffle quality but uses more memory. |
| `--seed` | `13` | Random seed. |

### Tokenizer

| Option | Default | Meaning |
|---|---:|---|
| `--max_vocab_size` | `5000` | Maximum APE vocabulary size. |
| `--min_freq_for_merge` | `2000` | Minimum frequency threshold for APE pair merges. |
| `--train_tokenizer` | on | Train a new APE tokenizer. |
| `--no_train_tokenizer` | off | Load an existing APE tokenizer vocabulary instead. |

### Model Architecture

| Option | Default | Meaning |
|---|---:|---|
| `--hidden_size` | `256` | Transformer hidden dimension. |
| `--num_hidden_layers` | `8` | Number of ModernBERT layers. |
| `--num_attention_heads` | `8` | Number of attention heads. Must divide hidden size. |
| `--intermediate_size` | `1024` | Feed-forward dimension. |
| `--max_seq_length` | `512` | Maximum tokenized sequence length. |
| `--global_attn_every_n_layers` | `3` | Use global attention every N layers. |
| `--local_attention` | `64` | Local attention window size. |

### MLM

| Option | Default | Meaning |
|---|---:|---|
| `--mlm_probability` | `0.30` | Fraction of eligible tokens selected for MLM corruption. |

The collator uses BERT-style corruption:

- 80% of selected tokens become `<mask>`
- 10% become random tokens
- 10% remain unchanged

Special tokens and padding are never masked.

### Training

| Option | Default | Meaning |
|---|---:|---|
| `--max_steps` | `150000` | Total optimizer steps. Streaming datasets use step-based training. |
| `--per_device_train_batch_size` | `128` | Per-device batch size. Use much smaller values on MPS. |
| `--per_device_eval_batch_size` | `128` | Per-device validation batch size. |
| `--gradient_accumulation_steps` | `2` | Accumulate gradients across this many steps. |
| `--learning_rate` | `1e-4` | Initial learning rate. |
| `--weight_decay` | `0.01` | AdamW weight decay. |
| `--warmup_ratio` | `0.06` | Fraction of training used for warmup. |
| `--max_grad_norm` | `1.0` | Gradient clipping norm. |

Effective batch size is:

```text
per_device_train_batch_size × gradient_accumulation_steps × number_of_devices
```

### Runtime and Checkpointing

| Option | Default | Meaning |
|---|---:|---|
| `--logging_steps` | `100` | Log every N steps. |
| `--eval_steps` | `5000` | Evaluate every N steps. |
| `--save_steps` | `5000` | Save checkpoint every N steps. |
| `--save_total_limit` | `3` | Keep only the most recent N checkpoints. |
| `--device_backend` | `auto` | One of `auto`, `cuda`, `mps`, or `cpu`. |
| `--bf16` | on | Enable bfloat16. Automatically disabled on MPS/CPU. |
| `--fp16` | off | Enable float16. Not recommended for MPS. |
| `--num_workers` | `4` | DataLoader workers. Reduced automatically on MPS/CPU. |
| `--compute_masked_accuracy` | off | Compute masked-token accuracy during eval. Off by default to avoid large MLM logits in memory. |
| `--debug` | off | Tiny run for smoke testing. |

## Output Files

Each run writes:

```text
output_dir/
  ape_<representation>_vocab.json
  ape_tokenizer_metadata.json
  README.checkpoint.md
  run_args.json
  trainer_state.json
  train_results.json
  eval_results.json
  checkpoint-*/
  final_model/
    config.json
    model.safetensors
```

The tokenizer is not a standard Hugging Face tokenizer. Always keep the APE vocabulary JSON with the model checkpoint.

## Reloading a Trained Checkpoint

```python
from transformers import AutoModelForMaskedLM
from ape_tokenizer import APETokenizer

model_dir = "./runs/cuda_selfies_ape_modernbert_15m/final_model"
vocab_path = "./runs/cuda_selfies_ape_modernbert_15m/ape_selfies_vocab.json"

tokenizer = APETokenizer()
tokenizer.load_vocabulary(vocab_path)

model = AutoModelForMaskedLM.from_pretrained(model_dir)
model.eval()

seq = "[C][C][O]"
batch = tokenizer(seq, add_special_tokens=True, return_tensors="pt")
out = model(**batch)
print(out.logits.shape)
```

For a SELFIES-trained model, convert SMILES first:

```python
import selfies as sf

smiles = "CCO"
seq = sf.encoder(smiles)
batch = tokenizer(seq, add_special_tokens=True, return_tensors="pt")
```

## Recommended Experimental Matrix

Start with:

| Model | Representation | Tokenizer | Params | Mask |
|---|---|---|---:|---:|
| ModernBERT-SELFIES-APE | SELFIES | APE | small | 0.30 |
| ModernBERT-SMILES-APE | SMILES | APE | small | 0.30 |

Then run masking-ratio ablations for the best setup:

```text
0.15, 0.30, 0.40, 0.50
```

Only scale the best variants.

## Notes on Mac MPS

MPS is suitable for debugging and short smoke tests. It is not recommended for final pretraining.

Use:

- full precision
- small batch sizes
- small models
- short `max_steps`
- no `flash-attn`
- no `bf16`
- no `fp16`

The script disables bf16/fp16 automatically when `--device_backend mps` is used.
