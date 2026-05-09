## Evaluation and baselines

| Baseline | Input | Featurizer / tokenizer | Model | Training mode | Purpose |
|---|---|---|---|---|---|
| ECFP4 + Logistic/Ridge | SMILES | RDKit Morgan radius=2, 2048 bits | sklearn | supervised on train split | classical fingerprint baseline |
| ECFP4 + Random Forest | SMILES | RDKit Morgan radius=2, 2048 bits | sklearn RF | supervised on train split | nonlinear fingerprint baseline |
| ModernMolBERT frozen | SMILES→SELFIES | SELFIES symbol tokenizer | local ModernMolBERT checkpoint | frozen embeddings + sklearn | representation-quality baseline |
| ModernMolBERT fine-tuned | SMILES→SELFIES | SELFIES symbol tokenizer | local ModernMolBERT checkpoint | end-to-end fine-tuning | main model evaluation |
| ChemBERTa external | SMILES | HF tokenizer | HF checkpoint name | frozen/fine-tuned, specify | literature/model baseline |
| MoLFormer external | SMILES | HF tokenizer | HF checkpoint name | frozen/fine-tuned, specify | literature/model baseline |

### External Hugging Face checkpoints

The following external checkpoints may be used for comparison. They are not trained by this repository unless explicitly stated.

| Name in tables | Hugging Face ID | Input representation | Notes |
|---|---|---|---|
| ChemBERTa-2 | `DeepChem/ChemBERTa-77M-MLM` or chosen checkpoint | SMILES | specify exact one |
| MoLFormer | `ibm/MoLFormer-XL-both-10pct` | SMILES | commonly used open MoLFormer checkpoint |
| ModernMolBERT-base | local path / future HF ID | SELFIES | trained by this repo |
| ModernMolBERT-large | local path / future HF ID | SELFIES | trained by this repo |

### Benchmark contract

All baselines must use:
- the same prepared MoleculeNet parquet splits,
- the same canonical SMILES column,
- the same task labels,
- the same train/valid/test split,
- the same metric implementation,
- no test-set use during model selection.

Frozen-feature baselines train sklearn heads on train or train+valid only.
Fine-tuned neural baselines select hyperparameters on validation and report test once.

### Benchmark contract

All baselines must use:
- the same prepared MoleculeNet parquet splits,
- the same canonical SMILES column,
- the same task labels,
- the same train/valid/test split,
- the same metric implementation,
- no test-set use during model selection.

Frozen-feature baselines train sklearn heads on train or train+valid only.
Fine-tuned neural baselines select hyperparameters on validation and report test once.

### Example evaluations

```bash
uv run python examples/ecfp4_moleculenet_example.py
uv run python examples/modernmolbert_moleculenet_example.py
```
