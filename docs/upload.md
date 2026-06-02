# Uploading to HuggingFace Hub

## upload_model

Stages and uploads a trained ModernMolBERT checkpoint to a HuggingFace model repo.

### Command

```bash
uv run python -m modernmolbert.upload_model \
  --run_dir runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span \
  --repo_id HauserGroup/ModernMolBERT-small-chembl36 \
  --checkpoint final \
  --private
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--run_dir` | required | Training run directory |
| `--repo_id` | required | HuggingFace repo, e.g. `HauserGroup/ModernMolBERT-small-chembl36` |
| `--checkpoint` | `final` | `final`, `best`, or numeric step e.g. `25000` |
| `--private` | false | Create/update repo as private |
| `--commit_message` | `"Upload trained ModernMolBERT checkpoint"` | HF commit message |
| `--hf_login` | false | Call `huggingface_hub.login()` using `HF_TOKEN_ORG` or `HF_TOKEN` from env |
| `--dry_run` | false | Stage and validate without uploading |
| `--keep_staging_dir` | — | Keep staged files at this path for inspection |

### Checkpoint resolution

- `final` → `<run_dir>/final_model/`
- `best` → reads `trainer_state.json` for `best_model_checkpoint`; falls back to `final_model/` if missing
- `<step>` → `<run_dir>/checkpoint-<step>/`

Fails if `model.safetensors` or `config.json` is absent in the resolved directory.

### What gets staged

```text
<staging_dir>/
  model.safetensors
  config.json                  # patched: model_type, special token IDs, vocab_size
  vocab.json
  selfies_vocab.json           # copy of vocab.json
  tokenizer_config.json        # patched: auto_map, model_max_length, use_fast=false
  special_tokens_map.json
  tokenization_ape.py
  README.md                    # auto-generated model card
  ape_tokenizer/               # compatibility copy of the same tokenizer files
    vocab.json
    selfies_vocab.json
    tokenizer_config.json
    special_tokens_map.json
    tokenization_ape.py
  run_args.json                # if present in run_dir
  trainer_state.json           # if present
  eval_results.json            # if present
  train_results.json           # if present
  all_results.json             # if present
  best_span_run.json           # if present
```

### Validation before upload

1. All required files present in staging dir.
2. `tokenizer_config.json` has no `tokenizer_class`, correct `auto_map`, `model_max_length=256`, `use_fast=false`.
3. Config and tokenizer load cleanly via `AutoConfig`, `AutoTokenizer` from `ape_tokenizer/`, and `APEPreTrainedTokenizer`.
4. Tokenizer and model vocab sizes match.
5. Forward pass on an example SELFIES string produces finite logits.

### Authentication

Set `HF_TOKEN_ORG` or `HF_TOKEN` in the environment or a `.env` file. Pass `--hf_login` to call `huggingface_hub.login()` interactively instead.

### Programmatic API

```python
from modernmolbert.upload_model import upload_model_to_hub
from pathlib import Path

result = upload_model_to_hub(
    run_dir=Path("runs/my_run"),
    repo_id="HauserGroup/ModernMolBERT-small-chembl36",
    checkpoint="final",
    private=True,
    dry_run=True,
)
print(result["staged_files"])
```

---

## upload_tokenizer

Uploads the APE SELFIES tokenizer to `HauserGroup/ApeTokenizer-SELFIES`. Paths are hardcoded; run from repo root.

### Command

```bash
uv run python -m modernmolbert.upload_tokenizer
```

No CLI flags. Edit the constants at the top of [upload_tokenizer.py](../src/modernmolbert/upload_tokenizer.py) to change targets.

### Hardcoded constants

| Constant | Value |
|----------|-------|
| `REPO_ID` | `HauserGroup/ApeTokenizer-SELFIES` |
| `VOCAB_PATH` | `tokenizer/chembl36_selfies_2m_ape_max2_min3000.json` |
| `METADATA_PATH` | `tokenizer/chembl36_selfies_2m_ape_max2_min3000.metadata.json` |
| `TOKENIZER_CODE` | `src/modernmolbert/tokenization_ape.py` |
| `MODEL_MAX_LENGTH` | `256` |

### What it does

1. Verifies metadata against expected values: `vocab_size=631`, `representation=SELFIES`, `max_merge_pieces=2`, `min_freq_for_merge=3000`, `tokenizer_train_size=2_000_000`, and special token IDs `bos=0 pad=1 eos=2 unk=3 mask=4`.
2. Instantiates `APEPreTrainedTokenizer` and saves it to `./tmp-hf-tokenizer/`.
3. Copies `tokenization_ape.py` and `metadata.json` into the staging directory.
4. Reloads the tokenizer via `AutoTokenizer` and verifies `model_max_length=256`, `vocab_size=631`, and that an example SELFIES fits within max length.
5. Creates the HF repo (private, `exist_ok=True`) and uploads.
6. Deletes `./tmp-hf-tokenizer/`.

### Authentication

Reads credentials from the HuggingFace cache (`huggingface-cli login`) or `HF_TOKEN` env var. No explicit login call in the script.
