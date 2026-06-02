#!/usr/bin/env python3
"""
write_model_cards.py

Generate HuggingFace model cards (README.md) for the four released
ModernMolBERT checkpoints, in a structure approximately matching MODEL_CARD.md
and including the simplest possible SELFIES encode + tokenize example.

This is the single source of truth for the static cards; the same card body is
mirrored by `build_readme()` in `src/modernmolbert/upload_model.py` for live
uploads. Run from the repo root: `python scripts/write_model_cards.py`.
"""

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/chembl36_small_mask_mlm_lr_sweep"

# Per-variant facts (verified from each checkpoint's config / training logs).
VARIANTS: list[dict[str, Any]] = [
    dict(
        path=RUNS / "modernmolbert_best_base/README.md",
        title="ModernMolBERT-base",
        repo="HauserGroup/ModernMolBERT-base",
        role="base release model",
        size="base",
        params="114.34M",
        hidden=768,
        layers=12,
        heads=12,
        inter=3072,
        maxpos=128,
        masking="standard",
        mlm=0.15,
        lr="2e-4",
    ),
    dict(
        path=RUNS / "modernmolbert_best_standard/final_model/README.md",
        title="ModernMolBERT-small",
        repo="HauserGroup/ModernMolBERT-small",
        role="small release model (standard masking)",
        size="small",
        params="34.15M",
        hidden=512,
        layers=8,
        heads=8,
        inter=2048,
        maxpos=128,
        masking="standard",
        mlm=0.15,
        lr="4e-4",
    ),
    dict(
        path=RUNS / "modernmolbert_best_span/final_model/README.md",
        title="ModernMolBERT-small-span",
        repo="HauserGroup/ModernMolBERT-small-span",
        role="small ablation variant (span masking)",
        size="small",
        params="34.15M",
        hidden=512,
        layers=8,
        heads=8,
        inter=2048,
        maxpos=128,
        masking="span",
        mlm=0.20,
        lr="2e-4",
    ),
    dict(
        path=RUNS / "modernmolbert_best_hetero_span/README.md",
        title="ModernMolBERT-small-hetero-span",
        repo="HauserGroup/ModernMolBERT-small-hetero-span",
        role="small ablation variant (heteroatom-biased span masking)",
        size="small",
        params="34.15M",
        hidden=512,
        layers=8,
        heads=8,
        inter=2048,
        maxpos=128,
        masking="hetero_span",
        mlm=0.15,
        lr="4e-4",
    ),
]

# Aspirin SELFIES, used as the worked example (one bracketed token per primitive).
EXAMPLE_SELFIES = (
    "[C][C][=Branch1][C][=O][O][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][O]"
)


def card(v: dict) -> str:
    repo = v["repo"]
    example_selfies = EXAMPLE_SELFIES
    return f"""---
license: mit
library_name: transformers
pipeline_tag: fill-mask
tags:
- chemistry
- molecules
- selfies
- ape-tokenizer
- modernbert
- masked-language-modeling
---

# {v["title"]}

ModernMolBERT is a family of compact encoder-only transformer models for
small-molecule representation learning. It pairs the
[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) architecture
with a chemically aware **Atom Pair Encoding (APE)** tokenizer and is pre-trained
from scratch with masked language modeling (MLM) on ~2.4M unique **SELFIES**
strings from ChEMBL 36. This checkpoint is the **{v["role"]}**.

The model expects **SELFIES** input (not SMILES) and is intended primarily as a
*frozen* molecular embedder.

## Model Details

- **Developed by:** Hauser Group, Department of Drug Design and Pharmacology, University of Copenhagen
- **Model type:** ModernBERT encoder &mdash; masked language model / frozen molecular embedder
- **Input representation:** SELFIES (convert SMILES first; see below)
- **Tokenizer:** Atom Pair Encoding (APE), 631-token SELFIES vocabulary
- **Pre-training data:** ChEMBL 36 (~2.4M unique small molecules)
- **License:** MIT
- **Repository:** https://github.com/HauserGroup/ModernMolBERT
- **Weights & tokenizer:** https://huggingface.co/HauserGroup

| field | value |
|-------|-------|
| size preset | {v["size"]} |
| parameters | {v["params"]} |
| hidden size | {v["hidden"]} |
| layers | {v["layers"]} |
| attention heads | {v["heads"]} |
| FFN intermediate size | {v["inter"]} |
| max sequence length | {v["maxpos"]} |
| vocabulary size | 631 |
| masking strategy | `{v["masking"]}` |
| MLM probability | {v["mlm"]} |
| peak learning rate | {v["lr"]} |

## How to Get Started with the Model

The model consumes **SELFIES** strings tokenized with the APE tokenizer.
Minimal end-to-end example (prints the tokenizer output and the model output):

```python
# pip install transformers torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

repo = "{repo}"
model = AutoModelForMaskedLM.from_pretrained(repo)
tokenizer = AutoTokenizer.from_pretrained(
    repo,
    subfolder="ape_tokenizer",   # load the custom APE tokenizer
    trust_remote_code=True,
    use_fast=False,
)

# A SELFIES string (one bracketed token per primitive); here aspirin.
selfies = "{example_selfies}"

inputs = tokenizer(selfies, return_tensors="pt")
print(inputs["input_ids"])
# tensor([[  0, 334, 335, 370, 333, 333, 333, 338, 377, 511,   6,   2]])
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# ['<s>', '[C][C]', '[=Branch1][C]', '[=O][O]', '[C][=C]', '[C][=C]', '[C][=C]', '[Ring1][=Branch1]', '[C][=Branch1]', '[C][=O]', '[O]', '</s>']

outputs = model(**inputs)
print(outputs.logits.shape)
# torch.Size([1, 12, 631])   # (batch, sequence_length, vocab_size)
```

If you start from SMILES, convert it to SELFIES first (e.g. the
[`selfies`](https://github.com/aspuru-guzik-group/selfies) package:
`selfies.encoder("CC(=O)Oc1ccccc1C(=O)O")`).

### Frozen molecular embedding (intended use)

Use the model as a molecular embedder by mean-pooling the final hidden states
over non-special SELFIES tokens:

```python
import torch
from transformers import AutoModel, AutoTokenizer

repo = "{repo}"
encoder = AutoModel.from_pretrained(repo).eval()
tokenizer = AutoTokenizer.from_pretrained(
    repo, subfolder="ape_tokenizer", trust_remote_code=True, use_fast=False,
)

selfies = "{example_selfies}"
inputs = tokenizer(selfies, return_tensors="pt")
with torch.no_grad():
    hidden = encoder(**inputs).last_hidden_state
    content_mask = inputs["attention_mask"].bool()
    for token_id in [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.mask_token_id,
    ]:
        if token_id is not None:
            content_mask = content_mask & inputs["input_ids"].ne(token_id)
    empty_rows = content_mask.sum(dim=1).eq(0)
    if empty_rows.any():
        content_mask[empty_rows] = inputs["attention_mask"].bool()[empty_rows]
    mask = content_mask.unsqueeze(-1).to(hidden.dtype)
    embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

print(embedding.shape)
# torch.Size([1, {v["hidden"]}])   # (batch, hidden_size)
```

> The APE tokenizer is a custom slow tokenizer shipped in the `ape_tokenizer/`
> subfolder. Loading from the repo root can route `AutoTokenizer` to the built-in
> fast ModernBERT tokenizer instead, so always pass `subfolder="ape_tokenizer"`,
> `trust_remote_code=True`, and `use_fast=False`.

## Uses

- **Direct use:** frozen molecular embeddings for property prediction,
  similarity search, clustering, and retrieval; masked-token fill-in.
- **Downstream use:** fine-tuning for molecular classification or regression on
  SELFIES inputs (e.g. with `AutoModelForSequenceClassification`).
- **Out of scope:** natural-language text; tasks that require generating valid
  SMILES; 3D/conformer-dependent tasks.

## Bias, Risks, and Limitations

The model is pre-trained only on drug-like ChEMBL 36 chemistry and may not
generalize to natural products, agrochemicals, fragments, or other
under-represented chemical space. It was evaluated as a *frozen* embedder (no
fine-tuning) on 25 binary classification benchmarks; performance under full
fine-tuning, on regression tasks, or on out-of-distribution scaffolds is not
characterised. The model has no access to 3D/conformer information.

## Citation

```bibtex
@article{{madsen_modernmolbert,
  title  = {{ModernMolBERT: A ModernBERT Encoder Family for SELFIES Molecular Language Modeling}},
  author = {{Madsen, Jakob S. and Angelucci, Sara and Hauser, Alexander S.}},
  year   = {{2026}}
}}
```

The APE tokenizer follows Leon et al., *Comparing SMILES and SELFIES
tokenization for enhanced chemical language modeling*, Sci. Rep. 14, 25016 (2024).
"""


TOKENIZER_REPO = "HauserGroup/ApeTokenizer-SELFIES"
TOKENIZER_VOCAB_SIZE = 631
TOKENIZER_MAX_LENGTH = 128
TOKENIZER_TRAIN_SIZE = "2M"
TOKENIZER_STAGING = ROOT / "tmp-hf-tokenizer"


def tokenizer_card() -> str:
    example_selfies = EXAMPLE_SELFIES
    repo = TOKENIZER_REPO
    return f"""---
license: mit
library_name: transformers
tags:
- chemistry
- molecules
- selfies
- ape-tokenizer
- tokenizer
---

# ApeTokenizer-SELFIES

ApeTokenizer-SELFIES is the **Atom Pair Encoding (APE)** tokenizer used by
[ModernMolBERT](https://github.com/HauserGroup/ModernMolBERT) — a family of
compact encoder-only transformer models for small-molecule representation
learning pre-trained on SELFIES strings from ChEMBL 36.

APE is a byte-pair-style merging scheme applied directly to SELFIES bracket
tokens, so every token boundary aligns with a chemically valid SELFIES
primitive. The vocabulary is derived from ~{TOKENIZER_TRAIN_SIZE} unique
SELFIES strings from ChEMBL 36.

## Tokenizer Details

- **Developed by:** Hauser Group, Department of Drug Design and Pharmacology, University of Copenhagen
- **Input representation:** SELFIES (convert SMILES first; see below)
- **Algorithm:** Atom Pair Encoding (APE) — pair merging over SELFIES bracket tokens
- **Vocabulary size:** {TOKENIZER_VOCAB_SIZE}
- **Max merge pieces:** 2
- **Min merge frequency:** 3000
- **Training corpus size:** {TOKENIZER_TRAIN_SIZE} unique SELFIES (ChEMBL 36)
- **License:** MIT
- **Repository:** https://github.com/HauserGroup/ModernMolBERT

| special token | id |
|---------------|----|
| `<s>` (BOS) | 0 |
| `<pad>` | 1 |
| `</s>` (EOS) | 2 |
| `<unk>` | 3 |
| `<mask>` | 4 |

## How to Get Started

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "{repo}",
    trust_remote_code=True,
    use_fast=False,
)

# A SELFIES string — here aspirin.
selfies = "{example_selfies}"

tokens = tokenizer.tokenize(selfies)
print(tokens)
# ['[C][C]', '[=Branch1][C]', '[=O][O]', '[C][=C]', '[C][=C]', '[C][=C]', '[Ring1][=Branch1]', '[C][=Branch1]', '[C][=O]', '[O]']

inputs = tokenizer(selfies, return_tensors="pt")
print(inputs["input_ids"])
# tensor([[  0, 334, 335, 370, 333, 333, 333, 338, 377, 511,   6,   2]])
```

If you start from SMILES, convert first:

```python
import selfies
smi = "CC(=O)Oc1ccccc1C(=O)O"
sf = selfies.encoder(smi)   # '[C][C][=Branch1][C][=O][O][C]...'
inputs = tokenizer(sf, return_tensors="pt")
```

### Using with ModernMolBERT models

This tokenizer is shared by all four ModernMolBERT checkpoints. Load it from
the model repo using `subfolder="ape_tokenizer"` to avoid routing
`AutoTokenizer` to the built-in fast ModernBERT tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "HauserGroup/ModernMolBERT-small",
    subfolder="ape_tokenizer",
    trust_remote_code=True,
    use_fast=False,
)
```

Or load this standalone repo directly as shown above — both produce identical
tokenizations.

## Citation

```bibtex
@article{{madsen_modernmolbert,
  title  = {{ModernMolBERT: A ModernBERT Encoder Family for SELFIES Molecular Language Modeling}},
  author = {{Madsen, Jakob S. and Angelucci, Sara and Hauser, Alexander S.}},
  year   = {{2026}}
}}
```

The APE algorithm follows Leon et al., *Comparing SMILES and SELFIES
tokenization for enhanced chemical language modeling*, Sci. Rep. 14, 25016 (2024).
"""


def main() -> None:
    for v in VARIANTS:
        p: Path = v["path"]
        if not p.parent.exists():
            print(f"SKIP (missing dir): {p}")
            continue
        p.write_text(card(v))
        print(f"wrote {p.relative_to(ROOT)}")

    tok_readme = TOKENIZER_STAGING / "README.md"
    if not TOKENIZER_STAGING.exists():
        print(f"SKIP (missing dir): {tok_readme}")
    else:
        tok_readme.write_text(tokenizer_card())
        print(f"wrote {tok_readme.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
