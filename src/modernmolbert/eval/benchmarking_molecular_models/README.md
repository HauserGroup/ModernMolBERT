# ModernMolBERT Molecular Benchmark

This package is the focused benchmark entrypoint for trained ModernMolBERT
checkpoints. The old upstream model-zoo implementation has been moved to
`src/modernmolbert/eval/junk/benchmarking_molecular_models_upstream/` for later
deletion.

This benchmark evaluates frozen molecular representations by training
lightweight supervised heads on top of cached embeddings. It does not fine-tune
the encoder and should not be used as a substitute for downstream fine-tuning
experiments.

## Run

Prepare MoleculeNet datasets first:

```bash
uv run python -m modernmolbert.eval.cli.prepare_moleculenet \
  --split scaffold \
  --seed 13 \
  --frac_train 0.8 \
  --frac_valid 0.1 \
  --frac_test 0.1 \
  --output_root data/eval/moleculenet_sanitized \
  --deepchem_data_dir data/deepchem/raw \
  --deepchem_save_dir data/deepchem/processed
```

Run the focused benchmark:

```bash
uv run python -m modernmolbert.eval.benchmarking_molecular_models.run \
  --model-path runs/pubchem10m_mps_base_pilot_256/final_model \
  --output-dir outputs/molecular_eval/modernmolbert_pilot \
  --datasets bbbp bace esol \
  --pooling mean \
  --heads auto \
  --batch-size 64 \
  --embed-batch-size 32 \
  --max-length 256 \
  --device auto
```

Outputs:

```text
outputs/molecular_eval/modernmolbert_pilot/
  embeddings/
  manifest.json
  results.csv
  results.jsonl
  run_config.json
  summary.csv
```

`datasets.yaml` lists the prepared datasets available to the wrapper. The
implementation delegates feature extraction, caching, task alignment, metrics,
and sklearn heads to the native `modernmolbert.eval` modules so there is only
one benchmark path to maintain.
