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
├── debugging/      diagnostic notebooks for data and encoding issues
├── examples/       self-contained worked examples
└── pxr/            PXR Challenge 2025 submission code
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

### `collect_sweep_results.py`
Python version of sweep result aggregation. Scans a sweep directory for
`mask_*__mlm_*__lr_*` subdirectories and collects metrics from
`all_results.json`, `run_args.json`, and `trainer_state.json` into a CSV.

Superseded by `R/collect_sweep_results.R`, which is more complete (adds
throughput metrics, span hyperparams, best-run flags, and fixed-eval join).
Kept for environments without R.

**Run:**
```bash
uv run python analysis/sweep/collect_sweep_results.py \
  --sweep runs/chembl36_small_mask_mlm_lr_sweep \
  --out results/sweep_results.csv
```

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

### `wrangle_for_dabest.py`
Wrangles benchmark results into the format expected by DABEST (Data Analysis
using Bootstrap-coupled ESTimation). Loads ModernMolBERT results alongside
Praski et al. reference results and normalises them to a common schema for
estimation-statistics comparison.

**Run:**
```bash
uv run python analysis/benchmark/wrangle_for_dabest.py \
  --results outputs/eval/praski_best_span/results.csv \
  --out outputs/dabest/combined.csv
```

### `fixup.ipynb`
Data-cleaning notebook. Converts legacy embedded-model joblib files from
bare numpy arrays to the current `EmbeddedDataset` format, and removes broken
molecules from embedding datasets. Run once after downloading pre-computed
embeddings that were produced with an older version of the pipeline.

### `visualizations.ipynb`
Comprehensive benchmark visualisation notebook. Loads and ranks embedded
models, applies best-variant selection logic, generates AUROC tables,
per-dataset performance tables, and cross-model win-rate plots. Primary
notebook for producing paper-ready benchmark figures.

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

## debugging/

### `selfies_debugger.ipynb`
Diagnostic notebook. Validates SELFIES encoding across all prepared datasets,
detects dative bonds and SELFIES parsing failures, and writes per-dataset JSON
reports. Use when a dataset shows unexpectedly high `<unk>` rates or embedding
failures.

---

## pxr/

PXR Challenge 2025 (activity track) submission code. Both scripts predict
hPXR pEC50 from binary ECFP4 fingerprints using Tanimoto kernel methods.

### `pxr_tanimoto_svr_knn.py`
Ensemble of Support Vector Regression (Tanimoto kernel) and KNN. Lighter
and faster; used as the primary submission.

### `pxr_tanimoto_krr.py`
Selectivity-weighted heteroskedastic Tanimoto Kernel Ridge Regression.
Weights training samples by reliability estimates derived from measurement
variance. More complex; used to explore uncertainty-aware regression.

---

## Related files outside this directory

| Location | Purpose |
|---|---|
| `R/collect_sweep_results.R` | Preferred sweep collector — richer than the Python version; outputs `sweep_results.csv` and `fixed_eval_collected.csv` |
| `R/FigX.R` | ggplot2 figure comparing masking strategies across MLM probabilities and learning rates |
| `scripts/train_chembl36_small_sweep.sh` | Launches the full hyperparameter sweep (span + hetero_span + standard, three MLM probs, three LRs) |
| `scripts/train_chembl36_small_sweep_standard.sh` | Standard-masking-only sweep |
| `scripts/train_chembl36_base_sweep_standard.sh` | Base-size model sweep |
| `exploratory/README.md` | Documents `--extra_vocab_symbols_path` / `--extra_vocab_selfies_path` arguments for tokenizer training |
