# Molecular tokenizers

`modernmolbert.train_tokenizer` supports APE and character-level BPE on both
SELFIES and SMILES. Select a combination with `--algorithm APE|BPE` and
`--representation SELFIES|SMILES`. The algorithm and representation are saved in
the adjacent `.metadata.json`; validation, model training and embedding load
the appropriate tokenizer from this metadata or a checkpoint directory.

APE starts from molecular primitives and preserves those boundaries. BPE starts
from characters and may merge across primitive boundaries. APE's
`--max_vocab_size` excludes five special tokens; BPE's includes them. The
`--max_merge_pieces` and `--extra_vocab_*` flags apply only to APE. Both can use
`--corpus_primitive_parquet` to include every primitive or character in the
full training split. For BPE, the initial alphabet and five special tokens must
fit inside `--max_vocab_size`.

For SMILES BPE, use the training command below with
`--algorithm BPE --representation SMILES --molecule_column smiles`, omit
`--max_merge_pieces`, and give the output a `tokenizer.json` filename. APE is
the default algorithm.

## What APE is

APE (Atomic Pair Encoding) is a BPE-inspired tokenizer designed for molecular string representations. Starting from a primitive alphabet (individual SELFIES bracket tokens or SMILES atoms/bonds), APE iteratively merges the most-frequent adjacent pair of tokens into a single merged token until a vocabulary size or frequency threshold is reached.

### Training rules in this repository

The [APE paper](https://doi.org/10.1038/s41598-024-76440-8) describes primitive
molecular tokens followed by repeated adjacent-pair merges. This implementation
makes the following choices explicit:

- Molecules stay separate: pairs cannot cross a molecule boundary. Malformed
  SELFIES rows are skipped and counted.
- Pair selection counts every adjacent occurrence, including overlapping
  candidates such as the three `[C][C]` pairs in `[C][C][C][C]`. The
  `--min_freq_for_merge` threshold applies to that candidate count. Ties go to
  the pair whose first occurrence is earliest in corpus order.
- A selected pair is replaced left to right without overlapping replacements.
  The `_freq.json` diagnostic records token occurrences **after** replacement:
  `[C][C][C][C]` produces two `[C][C]` tokens, not three. Tokens whose final
  frequency is zero remain in the vocabulary for primitive coverage. Correcting
  this diagnostic does not change merge selection, vocabulary order, or token IDs.
- `--max_merge_pieces` limits the primitive span of a merged token. This is a
  ModernMolBERT extension, not a constraint specified by the paper.

`modernmolbert.train_tokenizer` uses the incremental-count trainer in
`tokenization/ape.py`. It was checked against an independent small-corpus
specification and the former array-based implementation. On the first 2 million
rows of `data/pretrain/chembl36_selfies/train.parquet`, with vocabulary limit
2000, minimum pair frequency 3000 and span cap 2, both produced identical
ordered vocabularies and corrected frequencies (581 tokens); the incremental
trainer took 28.38 seconds versus 92.94 seconds for the array implementation.
This measurement includes training only, not Parquet reading.

The trainer uses dense pair-count tables whose size grows with the square of
`--max_vocab_size`. It rejects settings needing more than 512 MiB for the count
and eligibility tables before allocating them; this is a resource guard, not an
APE rule from the paper.
The self-contained `tokenization_ape.py` handles inference and model export.

Key properties:

- Merges never cross molecule boundaries.
- A `max_merge_pieces` cap limits how many primitive tokens one merged token may span, preventing over-compression of long sequences.
- The vocabulary includes all primitive symbols seen in the corpus, so `unk_rate` is always 0 for any molecule whose primitives appear at least once.

The implementation lives in `src/modernmolbert/tokenization_ape.py` as `APEPreTrainedTokenizer`, which extends `PreTrainedTokenizer` so it works directly with HuggingFace `Trainer` and `AutoTokenizer`.

## Representations

Pass `--representation SELFIES` or `--representation SMILES`. SELFIES is the default and the representation used for all published ModernMolBERT checkpoints. SMILES support is available in tokenizer training, model training and embedding.

SELFIES primitive tokens are bracket tokens: `[C]`, `[=O]`, `[Branch1_2]`, etc.
SMILES primitive tokens are atoms and bond/ring characters: `C`, `O`, `Br`, `(`, `=`, `%12`, etc.

## Training a tokenizer

### Production command (ChEMBL36 SELFIES)

```bash
uv run python -m modernmolbert.train_tokenizer \
  --output_vocab_path tokenizer/custom_selfies_ape.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 2000 \
  --min_freq_for_merge 3000 \
  --max_merge_pieces 2 \
  --seed 42
```

This produces:
- `tokenizer/custom_selfies_ape.json` — vocabulary
- `tokenizer/custom_selfies_ape.metadata.json` — training provenance and SHA256

> **Shipped tokenizer predates the current default.** The committed
> `tokenizer/chembl36_selfies_2m_ape_max2_min3000.json` was built with
> `--extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt`
> (42 benchmark-derived symbols added, 631 tokens total; recorded in its
> `.metadata.json`). The command above writes a separate tokenizer and does
> not reproduce the shipped file.

The corpus-only replacement for the revision run is
`tokenizer/chembl36_selfies_2m_ape_max2_min3000_corpus_v1.json`. It has 588
tokens, no injected symbols, and every primitive in the full training split,
including the `.` separator. Its checks and provenance are in
[revision_run_record.md](revision_run_record.md).

### Extra vocabulary symbols (off by default)

By default no tokens are force-added: the vocabulary is exactly what APE merge
training learns from the corpus. Two opt-in flags append primitive tokens after
merge learning:

- `--extra_vocab_symbols_path`: text file with one primitive token per line
  (SELFIES `[C@@H1]`, SMILES `[Fe+3]`).
- `--extra_vocab_selfies_path`: text file with one full SELFIES string per line;
  all bracket symbols are extracted (SELFIES only).

Only use these deliberately. Symbol lists derived from benchmark molecules
(e.g. `tokenizer/extra_symbols/benchmark_missing_selfies_symbols_min10.txt`) put
evaluation-set information into the vocabulary.

### Key hyperparameters

| Flag | Default | Effect |
|---|---|---|
| `--tokenizer_train_size` | 2 000 000 | Molecules sampled from the corpus for merge training |
| `--max_vocab_size` | 5000 | APE: maximum non-special tokens; BPE: total vocabulary size including five special tokens |
| `--min_freq_for_merge` | 3000 | Stop merging when best pair frequency falls below this |
| `--max_merge_pieces` | 8 | Max primitive tokens a merged token may span. 0/negative = no cap |
| `--extra_vocab_symbols_path` | off (`None`) | Opt-in. Text file with one primitive token per line; force-added after training |
| `--extra_vocab_selfies_path` | off (`None`) | Opt-in, SELFIES only. Full SELFIES strings; bracket symbols extracted and force-added |

### Conservative vs. moderate vs. production settings

```bash
# Conservative: more fragmented, longer sequences, lower compression
uv run python -m modernmolbert.train_tokenizer \
  --output_vocab_path tokenizer/custom_selfies_ape_max4.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 500000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000 \
  --max_merge_pieces 4 \
  --seed 42

# Moderate
uv run python -m modernmolbert.train_tokenizer \
  --output_vocab_path tokenizer/custom_selfies_ape_max8.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 500000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000 \
  --max_merge_pieces 8 \
  --seed 42
```

### Output structure after training

```text
tokenizer/
  chembl36_selfies_2m_ape_max2_min3000.json          # vocabulary: {"[C]": 5, "[O]": 6, ...}
  chembl36_selfies_2m_ape_max2_min3000.metadata.json  # provenance
  chembl36_selfies_2m_ape_max2_min3000_freq.json      # token frequencies (diagnostic)
  extra_symbols/
    benchmark_missing_selfies_symbols_min10.txt        # force-added primitive symbols
```

## Validating a tokenizer

Run before every training job.

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SELFIES \
  --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.metadata.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column selfies \
  --split train \
  --n 10000 \
  --max_seq_length 256
```

### What the validator checks

1. Vocabulary file exists and its SHA256 matches the metadata record.
2. Representation in metadata matches `--representation`.
3. Vocabulary has ≥ 100 tokens.
4. Ethanol (`[C][C][O]` for SELFIES, `CCO` for SMILES) tokenizes without unknowns.
5. Over `n` sampled molecules: unk rate, truncation rate, empty-sequence rate, sequence length percentiles.

### Target metrics

```text
unk_rate:              0
mostly_unknown_rate:   0
truncation_rate@256:   ~0
mean_len:              25–60   (for max2 settings)
p95_len:               < 150
```

Signs of misconfigured tokenizer:

| Symptom | Likely cause |
|---|---|
| `unk_rate > 0` | Missing SELFIES primitives; broaden the tokenizer corpus, or opt in to `--extra_vocab_symbols_path` deliberately |
| `mean_len < 10` | Over-merged; reduce `--max_merge_pieces` or increase `--min_freq_for_merge` |
| `mean_len > 100` | Under-merged; fewer training molecules or lower `--min_freq_for_merge` |
| `truncation_rate > 0.05` | Sequences too long for `max_seq_length`; increase or reduce `max_merge_pieces` |
| Large gap between ChEMBL validation and benchmark molecules | Broaden the tokenizer corpus; forcing benchmark-derived symbols via `--extra_vocab_symbols_path` is opt-in and leaks eval-set vocabulary |

## Saving and loading

```python
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

# Save HuggingFace-compatible tokenizer directory
tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
tokenizer.load_vocabulary_file("tokenizer/chembl36_selfies_2m_ape_max2_min3000.json")
tokenizer.save_pretrained("runs/my_run/ape_tokenizer")

# Reload
tok = APEPreTrainedTokenizer.from_pretrained(
    "runs/my_run/ape_tokenizer",
    trust_remote_code=True,
)
ids = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")
```

## Special tokens

| Token | ID |
|---|---|
| `<s>` (BOS) | 0 |
| `<pad>` | 1 |
| `</s>` (EOS) | 2 |
| `<unk>` | 3 |
| `<mask>` | 4 |
| First learned token | 5 |

## SMILES tokenizer (experimental)

Train on SMILES instead of SELFIES by switching the representation and pointing at a SMILES column:

```bash
uv run python -m modernmolbert.train_tokenizer \
  --output_vocab_path tokenizer/custom_smiles_ape.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --molecule_column smiles \
  --representation SMILES \
  --tokenizer_train_size 500000 \
  --max_vocab_size 500 \
  --min_freq_for_merge 1000 \
  --max_merge_pieces 4 \
  --seed 42
```

Validate:

```bash
uv run python -m modernmolbert.validate_tokenizer \
  --representation SMILES \
  --tokenizer_vocab_path tokenizer/custom_smiles_ape.json \
  --tokenizer_metadata_path tokenizer/custom_smiles_ape.metadata.json \
  --molecule_column smiles \
  --n 1000
```
