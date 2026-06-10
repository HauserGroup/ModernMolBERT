# ModernMolBERT

[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="imgs/mmbert_text.png" width="600"/>

Compact [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) encoders for **SELFIES** molecular embeddings, pre-trained with masked language modeling on ~2.4M small molecules from ChEMBL 36.

ModernMolBERT is a small family of two models released in standard Hugging Face Transformers format. It takes a SELFIES string and returns a general-purpose molecular embedding for property prediction, similarity search, clustering, and retrieval — used as a **frozen featuriser**, no fine-tuning required. A chemically aware Atom Pair Encoding (APE) tokenizer keeps the vocabulary small (631 tokens) and inference has no custom runtime dependencies.

For benchmarks, baselines (ECFP4, ChemBERTa-2, SELFormer, MoLFormer), and full results, see the preprint — citation in [`CITATION.cff`](CITATION.cff).

## Models

| Model | Params | Layers | Hidden | Hugging Face repo |
|-------|--------|--------|--------|-------------------|
| ModernMolBERT-small | 34M | 8 | 512 | [`HauserGroup/ModernMolBERT-small`](https://huggingface.co/HauserGroup/ModernMolBERT-small) |
| ModernMolBERT-base  | 114M | 12 | 768 | [`HauserGroup/ModernMolBERT-base`](https://huggingface.co/HauserGroup/ModernMolBERT-base) |

Both use vocab size 631 and a max sequence length of 128 tokens.

## Install

Python 3.13, with [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv sync
```

For CUDA training, add the CUDA extra: `uv sync --extra cuda`. FlashAttention is optional and CUDA-only — do not install it for Mac MPS.

The package is not published to PyPI. To install it directly into another environment, point pip at this GitHub repository:

```bash
pip install git+https://github.com/HauserGroup/ModernMolBERT
```

To only *use* a released model, you do not need this package at all — just `transformers`, `torch`, and `selfies`, and load the weights from the Hugging Face Hub:

```bash
pip install transformers torch selfies
```

## Quickstart — get embeddings

The model consumes **SELFIES** strings tokenized with the APE tokenizer. The standard molecular representation is the first-token embedding:

```python
import torch
from transformers import AutoModel, AutoTokenizer

repo = "HauserGroup/ModernMolBERT-base"
model = AutoModel.from_pretrained(repo).eval()
tokenizer = AutoTokenizer.from_pretrained(
    repo,
    subfolder="ape_tokenizer",
    trust_remote_code=True,
    use_fast=False,
)

# A SELFIES string (one bracketed token per primitive); here aspirin.
selfies = "[C][C][=Branch1][C][=O][O][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][O]"

inputs = tokenizer(selfies, return_tensors="pt")
with torch.no_grad():
    embedding = model(**inputs).last_hidden_state[:, 0]

print(embedding.shape)  # (1, hidden_size)
```

Starting from **SMILES**? Convert it first with the [`selfies`](https://github.com/aspuru-guzik-group/selfies) package:

```python
import selfies
selfies_str = selfies.encoder("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
```

For masked-token prediction, load the same checkpoint with `AutoModelForMaskedLM`:

```python
from transformers import AutoModelForMaskedLM

mlm = AutoModelForMaskedLM.from_pretrained(repo)
logits = mlm(**inputs).logits
```

> The tokenizer loads from the `ape_tokenizer/` subfolder: current Transformers releases disable custom root tokenizers for `model_type="modernbert"`, so the custom APE code ships there.

## Embedding SMILES in a pipeline

For SMILES-in workflows (e.g. benchmarking), use `ModernMolBERTSelfiesFeaturizer` ([src/modernmolbert/eval/featurizers/modernmolbert_selfies.py](src/modernmolbert/eval/featurizers/modernmolbert_selfies.py)). It accepts SMILES, converts valid ones to SELFIES internally, tokenizes with the APE tokenizer, and returns pooled features. Configure it with JSON:

```json
{
  "type": "modernmolbert_selfies",
  "name": "modernmolbert_base",
  "model_dir": "HauserGroup/ModernMolBERT-base",
  "tokenizer_path": "HauserGroup/ModernMolBERT-base",
  "max_seq_length": 128,
  "pooling": "mean",
  "device": "auto",
  "batch_size": 32
}
```

## Reproduce the benchmarks

Evaluation is a three-stage frozen-featuriser pipeline — download datasets, embed with a checkpoint, score with downstream heads:

```bash
# 1. download + prepare datasets (TDC + OGB)
uv run python src/modernmolbert/eval/benchmarking_molecular_models/download.py --datasets all

# 2. embed with a checkpoint
uv run python src/modernmolbert/eval/benchmarking_molecular_models/embed_modernmolbert.py \
  --datasets all --model-dir runs/<run>/final_model --embedder my_model \
  --batch-size 32 --device auto --pooling mean

# 3. score (rf / ridge / knn heads) -> results CSV
uv run python src/modernmolbert/eval/benchmarking_molecular_models/score.py \
  --embedder my_model --output-csv outputs/eval/my_run/results.csv \
  --checkpoint-dir outputs/eval/my_run/checkpoints
```

Full flags, dataset list, output schema, and custom-dataset setup: [docs/evaluation.md](docs/evaluation.md). Baseline featurisers (ECFP4, ChemBERTa, MoLFormer): [docs/baselines.md](docs/baselines.md). Paper-ready figures are generated from the notebooks in [analysis/](analysis/).

## Train your own

Train from scratch in four steps. Each command below is the minimal form; see the linked docs for full options.

```bash
# 1. prepare the ChEMBL 36 SELFIES pretraining dataset      (docs/datasets.md)
uv run python -m modernmolbert.data.prepare_chembl36_selfies \
  --output_dir data/pretrain/chembl36_selfies

# 2. train + validate the APE tokenizer                     (docs/tokenizer.md)
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --dataset_name mikemayuare/PubChem10M_SMILES_SELFIES \
  --max_vocab_size 5000 --min_freq_for_merge 2000
uv run python -m modernmolbert.validate_tokenizer --representation SELFIES \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json

# 3. tiny end-to-end debug run before anything long
uv run python -m modernmolbert.train_selfies_ape_modernbert --debug \
  --output_dir runs/debug_selfies --model_size base \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json

# 4. full training run (CUDA)
uv run accelerate launch -m modernmolbert.train_selfies_ape_modernbert \
  --output_dir runs/selfies_main --device_backend cuda --model_size base \
  --max_seq_length 128 --bf16 --report_to tensorboard \
  --tokenizer_vocab_path tokenizer/selfies_ape_tokenizer.json \
  --tokenizer_metadata_path tokenizer/selfies_ape_tokenizer.metadata.json
```

Tokenizer validation is a mandatory gate before model training. Masking strategies (`standard` / `span` / `hetero_span`): [docs/masking_strategies.md](docs/masking_strategies.md). Readiness checks and tests: [docs/tests.md](docs/tests.md). Uploading checkpoints and tokenizers to the Hub: [docs/upload.md](docs/upload.md).

Each run writes a self-contained `final_model/` directory (weights, config, tokenizer, custom tokenizer code) ready for inference and Hub upload. Reload it like any Transformers checkpoint, loading the tokenizer from its `ape_tokenizer/` subfolder.

## Repository layout

```text
src/modernmolbert/
  train_ape_tokenizer.py               # tokenizer training CLI
  validate_tokenizer.py                # tokenizer validation CLI
  train_selfies_ape_modernbert.py      # MLM pretraining CLI
  select_pretraining_run.py            # rank sweep runs, select best checkpoint
  collator.py                          # MLM collator (standard / span / hetero_span)
  data/                                # pretraining dataset prep
  eval/
    featurizers/                       # frozen featurisers (ModernMolBERT, base)
    benchmarking_molecular_models/     # download / embed / score benchmark pipeline
docs/                                  # datasets, tokenizer, evaluation, baselines, masking, upload, tests
configs/featurizers/                   # featuriser JSON configs
analysis/                              # benchmark audit + paper figures
```

## Documentation

- [docs/datasets.md](docs/datasets.md) — pretraining data prep and dataset layout.
- [docs/tokenizer.md](docs/tokenizer.md) — APE tokenizer training and validation.
- [docs/masking_strategies.md](docs/masking_strategies.md) — MLM masking strategies.
- [docs/evaluation.md](docs/evaluation.md) — benchmark scoring pipeline.
- [docs/baselines.md](docs/baselines.md) — ECFP4 and external SMILES encoders.
- [docs/upload.md](docs/upload.md) — uploading models and tokenizers to the Hub.
- [docs/tests.md](docs/tests.md) — tests and readiness gates.

## Citation

If you use this work, cite the accompanying paper — see [`CITATION.cff`](CITATION.cff) or the "Cite this repository" button on GitHub.

## References

The APE tokenizer is based on Leon, M., Perezhohin, Y., Peres, F. et al. (2024), *Comparing SMILES and SELFIES tokenization for enhanced chemical language modeling*, Scientific Reports 14, 25016. https://doi.org/10.1038/s41598-024-76440-8

Span and hetero-span masking take inspiration from Peng, T. et al. (2025), *Pre-trained molecular language models with random functional group masking*, npj Artificial Intelligence 1, 28. https://doi.org/10.1038/s44387-025-00029-3
