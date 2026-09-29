# Analysis

One home for all standalone analysis, evaluation, and visualisation scripts.
Training infrastructure and data preparation remain in `scripts/`. R scripts remain in `R/`.

---

## Directory map

```
analysis/
├── tokenization/   sequence length and vocabulary coverage checks
├── sweep/          MLM hyperparameter sweep collection and comparison
├── benchmark/      downstream benchmark evaluation and visualisation
├── validation/     tokenizer and model sanity checks
├── examples/       self-contained worked examples
└── Fig_PacMap/     PaCMAP embedding figure notebooks
```

---

## tokenization/

### `check_tokenized_lengths.py`
Analyses SELFIES tokenized sequence length distributions across all prepared
datasets. Computes truncation rates at different `max_seq_length` settings to
inform the choice of sequence length cap during pretraining.

**Run:**
```bash
uv run python analysis/tokenization/check_tokenized_lengths.py
```

---

## sweep/

### `fixed_eval_best_models.py`
Apples-to-apples fixed-mask evaluation. Loads the best checkpoint from each
masking group, freezes a common validation dataset (standard masking at 15%),
and evaluates all models on it. Eliminates confounding from per-run masking
differences in the training-time eval.

**Run:**
```bash
uv run python analysis/sweep/fixed_eval_best_models.py \
  --sweep runs/chembl36_small_mask_mlm_lr_sweep
```

### `01A_ideal_masking_probability.py`
Identifies the optimal learning rate per masking probability using sweep
results. Loads `sweep_results.csv`, selects the best LR for each
masking strategy × probability combination, benchmarks on MoleculeNet
datasets, and plots ROC curves by masking probability.

**Depends on:** `runs/.../sweep_results.csv` existing (run `R/collect_sweep_results.R` first).

---

## benchmark/

### `visualizations.ipynb`
Comprehensive benchmark visualisation notebook. Loads and ranks embedded
models, applies best-variant selection logic, generates AUROC tables,
per-dataset performance tables, and cross-model win-rate plots. Exploratory
only; the paper's benchmark tables and figures come from `scripts/paper/`.

---

## validation/

### `check_tokenizer_model_compatibility.py`
Validates that a HuggingFace tokenizer is compatible with a given model
config: checks vocab size, pad/mask token IDs, and runs an end-to-end
tokenization + forward pass to catch shape mismatches early.

**Run:**
```bash
uv run python analysis/validation/check_tokenizer_model_compatibility.py \
  --tokenizer tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --model runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span/final_model
```

### `check_hf_tokenizer_matches_local.py`
Compares the local `APEPreTrainedTokenizer` against the HuggingFace-hosted
version. Verifies vocabulary, metadata, and that encoding produces byte-for-byte
identical token sequences. Run before uploading a new tokenizer to the Hub.

**Run:**
```bash
uv run python analysis/validation/check_hf_tokenizer_matches_local.py \
  --local tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --hf HauserGroup/ModernMolBERT
```

---

## Related files outside this directory

| Location | Purpose |
|---|---|
| `R/collect_sweep_results.R` | Sweep collector; outputs `sweep_results.csv` and `fixed_eval_collected.csv` |
| `R/FigX.R` | ggplot2 figure comparing masking strategies across MLM probabilities and learning rates |
| `scripts/sweeps/run_sweep.py --model-size small` | Small-model sweep (standard + span, three MLM probs, three LRs; add `--masking standard span hetero_span` for the opt-in hetero_span ablation) |
| `scripts/sweeps/run_sweep.py --model-size small --masking standard` | Standard-masking-only small-model sweep |
| `scripts/sweeps/run_sweep.py --model-size base` | Base-size model sweep |
| `docs/tokenizer.md` | Documents `--extra_vocab_symbols_path` / `--extra_vocab_selfies_path` arguments for tokenizer training |
