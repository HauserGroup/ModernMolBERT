## Extra SELFIES vocabulary symbols
`train_ape_tokenizer` supports forcing additional SELFIES primitive symbols into
the tokenizer vocabulary after APE merge training.
This is useful when a tokenizer trained on the pretraining corpus has low
sequence truncation but nonzero `<unk>` rates on downstream datasets because rare
valid SELFIES primitives are absent from the tokenizer vocabulary.
The extra symbols are added after APE training and before writing the final
vocabulary. They do not affect learned APE merge rules.

Then document the two accepted formats.

Expected format: `--extra_vocab_symbols_path`

A plain UTF-8 text file with one SELFIES primitive token per line.

Example:

```text
[C@@H1]
[C@H1]
[/C]
[\C]
[C@@]
[C@]
[/N]
[\N]

```

Rules:

 - one token per line
 - blank lines are ignored
 - lines starting with # are ignored
 - each non-comment line must be one bracketed SELFIES symbol

So this is valid:

```text
# stereochemistry
[C@@H1]
[C@H1]
[C@@]
[C@]
# directional bonds
[/C]
[\C]
[/N]
[\N]

```

This is not valid:

`[C][C@@H1][O]`

because that is a full SELFIES string, not one primitive token.

Expected format: `--extra_vocab_selfies_path`

A plain UTF-8 text file with one full SELFIES string per line.

Example:

```text
[C][C@@H1][Branch1][C][O][C]
[C][=C][/C][=C][\C][Ring1][=Branch1]
[N][C@H1][C][=O][O]
```

Rules:

 - one SELFIES string per line
 - blank lines are ignored
 - lines starting with # are ignored
 - all bracketed SELFIES symbols are extracted with `\[[^\]]+\]`

This is useful if your benchmark-analysis script exports full converted SELFIES strings rather than a precomputed symbol list.

Example documentation block

### Forcing rare SELFIES primitives into the APE tokenizer
Some downstream benchmark datasets may contain rare but valid SELFIES primitive
symbols that are absent from a tokenizer trained only on the pretraining corpus.
This shows up as nonzero `<unk>` rates despite negligible truncation.
To force specific primitive symbols into the tokenizer, create a text file with
one SELFIES token per line:
```text
[C@@H1]
[C@H1]
[/C]
[\C]
[C@@]
[C@]
[/N]
[\N]
```

Then pass it during tokenizer training:

```bash
uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_benchmark_covered_ape_tokenizer.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000 \
  --extra_vocab_symbols_path tokenizer/extra_symbols/benchmark_missing_selfies_symbols.txt

Alternatively, pass a file with one full SELFIES string per line:

uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_benchmark_covered_ape_tokenizer.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 2000 \
  --extra_vocab_selfies_path data/prepared/benchmark_selfies_for_vocab_coverage.txt
```


## Where to store the symbol file


`tokenizer/extra_symbols/`

because the file is an input to tokenizer training, not a dataset artifact.
