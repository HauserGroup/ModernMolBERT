# Baselines and external models

This document covers non-ModernMolBERT featurizers and external frozen encoder baselines.

All baselines should use the same frozen benchmark machinery as ModernMolBERT:

```text
SMILES dataset
-> RepresentationFeaturizer
-> FeatureBatch(X, valid_mask, metadata)
-> cached features
-> downstream model
-> shared metrics
-> results.csv / manifest.json
```

Do not compare benchmark results from separate embedding/sklearn scripts.

## ECFP4

ECFP4 is the primary classical fingerprint baseline.

```json
{
  "type": "ecfp4",
  "name": "ecfp4_2048",
  "n_bits": 2048,
  "radius": 2
}
```

Use this baseline in early pilot and core MoleculeNet suites.

## Hugging Face SMILES encoders

Generic Hugging Face SMILES encoders use the `hf_smiles` featurizer.

Example ChemBERTa config:

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

## MoLFormer baseline

MoLFormer is included as a frozen SMILES-encoder baseline and should be evaluated with the same downstream learners as every other representation. Do not fine-tune MoLFormer for the primary frozen benchmark.

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

### MoLFormer featurizer config

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

## Baseline policy

- Use one shared benchmark pipeline for all frozen representations.
- Keep baselines frozen unless explicitly running a fine-tuning experiment.
- Keep external-model environment quirks out of the main training path.
- Pin revisions when `trust_remote_code=True` is required.
