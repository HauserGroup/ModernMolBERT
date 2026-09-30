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
| `--keep_staging_dir` | — | Keep staged files at this path for inspection (must be empty or absent) |

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
2. `tokenizer_config.json` has no `tokenizer_class`, correct `auto_map`, `model_max_length=128`, `use_fast=false`.
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

Uploads an APE tokenizer to the Hub. Defaults target `HauserGroup/ApeTokenizer-SELFIES` and the
committed 631-token vocabulary; run from the repository root. The card it writes is the fixed
historical one (audit finding R41), so do not use it for the factorial tokenizers before that is
generalised.

### Command

```bash
uv run python -m modernmolbert.upload_tokenizer \
  --repo_id HauserGroup/ApeTokenizer-SELFIES \
  --vocab_path tokenizer/chembl36_selfies_2m_ape_max2_min3000.json \
  --dry_run
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--repo_id` | `HauserGroup/ApeTokenizer-SELFIES` | Target Hub repository |
| `--vocab_path` | `tokenizer/chembl36_selfies_2m_ape_max2_min3000.json` | Vocabulary JSON |
| `--metadata_path` | `<vocab stem>.metadata.json` | Tokenizer metadata |
| `--staging_dir` | `./tmp-hf-tokenizer` | Staging directory; must be empty or absent |
| `--model_max_length` | `128` | Context stored in the tokenizer config |
| `--commit_message` | generated | Hub commit message |
| `--private` | false | Create or update the repository as private |
| `--hf_login` | false | Log in with `HF_TOKEN_ORG` or `HF_TOKEN` from the environment or `.env` |
| `--dry_run` | false | Stage and validate without contacting the Hub |
| `--keep_staging_dir` | false | Keep the staging directory after a successful upload |

### What it does

1. Verifies the metadata: representation is SELFIES or SMILES, `max_merge_pieces`,
   `min_freq_for_merge`, `tokenizer_train_size` and `vocab_size` are present, the vocabulary
   file's SHA-256 matches `tokenizer_sha256`, the vocabulary size matches `vocab_size`, and
   the special IDs are `bos=0 pad=1 eos=2 unk=3 mask=4`.
2. Saves an `APEPreTrainedTokenizer` to the staging directory and copies `tokenization_ape.py`
   and the metadata (as `metadata.json`, `tokenizer_metadata.json` and
   `ape_tokenizer_metadata.json`).
3. Writes the model card, reloads the staged tokenizer through `AutoTokenizer` and checks
   `model_max_length`, the vocabulary size and that an example molecule fits.
4. Creates the repository (`exist_ok=True`) and uploads; a dry run stops before this step.
5. Removes the staging directory unless `--keep_staging_dir` is set. A dry run leaves it, so
   delete it before the next run.

### Authentication

Credentials come from the Hugging Face cache (`huggingface-cli login`) or `HF_TOKEN_ORG` /
`HF_TOKEN`; `--hf_login` performs an explicit login.
