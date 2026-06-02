#!/usr/bin/env python3

"""
uv run python -m modernmolbert.upload_model \
  --run_dir runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span \
  --repo_id HauserGroup/ModernMolBERT-small-chembl36 \
  --checkpoint final \
  --private \
  --dry_run \
  --keep_staging_dir tmp-hf-model
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi

from modernmolbert.utils import repo_root


MODEL_MAX_LENGTH = 256

# Default collator parameters per masking strategy.
# mlm_probability and span params match training runs documented in write_model_cards.py.
MASKING_DEFAULTS: dict[str, dict[str, Any]] = {
    "standard": {
        "masking_strategy": "standard",
        "mlm_probability": 0.15,
    },
    "span": {
        "masking_strategy": "span",
        "mlm_probability": 0.20,
        "span_p_geom": 0.4,
        "span_max_length": 6,
    },
    "hetero_span": {
        "masking_strategy": "hetero_span",
        "mlm_probability": 0.20,
        "span_p_geom": 0.4,
        "span_max_length": 6,
        "heteroatom_start_weight": 2.0,
    },
}

EXPECTED_SPECIAL_IDS = {
    "bos_token_id": 0,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
    "mask_token_id": 4,
}
EXAMPLE_SELFIES = (
    "[C][N][Branch1][C][C][C][C][C][=C][NH1][C][=C][C][=C][C]"
    "[Branch1][#Branch2][O][P][=Branch1][C][=O][Branch1][C][O][O]"
    "[=C][Ring1][=C][Ring1][O]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a trained ModernMolBERT checkpoint to HuggingFace Hub.",
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Training run directory, for example runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID, for example HauserGroup/ModernMolBERT-small-chembl36.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final",
        help="Checkpoint to upload: final, best, or a numeric step such as 25000.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the HuggingFace repo as private.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload trained ModernMolBERT checkpoint",
        help="Commit message for the HuggingFace upload.",
    )
    parser.add_argument(
        "--hf_login",
        action="store_true",
        help="Call huggingface_hub.login() before uploading. Reads HF_TOKEN_ORG or HF_TOKEN from env or .env.",
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default="standard",
        choices=["standard", "span", "hetero_span"],
        help=(
            "Masking strategy used during pre-training (default: standard). "
            "Written to collator_config.json in the staged upload so users "
            "know what was used and can switch strategies when fine-tuning."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Stage and validate upload contents without creating or updating a Hub repo.",
    )
    parser.add_argument(
        "--keep_staging_dir",
        type=Path,
        default=None,
        help="Keep staged upload files in this directory for debugging.",
    )
    return parser.parse_args()


def resolve_source_dir(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "final":
        source = run_dir / "final_model"

    elif checkpoint == "best":
        state_path = run_dir / "trainer_state.json"

        if not state_path.exists():
            fallback = run_dir / "final_model"
            if fallback.exists():
                print(
                    f"trainer_state.json not found in {run_dir}; falling back to {fallback}",
                    flush=True,
                )
                source = fallback
            else:
                raise FileNotFoundError(f"trainer_state.json not found in {run_dir}")

        else:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            best = state.get("best_model_checkpoint")

            if not best:
                fallback = run_dir / "final_model"
                if fallback.exists():
                    print(
                        f"trainer_state.json has no best_model_checkpoint; falling back to {fallback}",
                        flush=True,
                    )
                    source = fallback
                else:
                    raise ValueError("trainer_state.json has no best_model_checkpoint entry")

            else:
                source = Path(best)
                if not source.is_absolute():
                    source = repo_root() / source

                if not source.exists():
                    fallback = run_dir / "final_model"
                    if fallback.exists():
                        print(
                            f"best_model_checkpoint does not exist: {source}; falling back to {fallback}",
                            flush=True,
                        )
                        source = fallback
                    else:
                        raise FileNotFoundError(f"best_model_checkpoint does not exist: {source}")

    else:
        source = run_dir / f"checkpoint-{checkpoint}"

    if not (source / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors not found in {source}")

    if not (source / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {source}")

    return source


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_patch_config(source_dir: Path, run_dir: Path, vocab_size: int) -> dict[str, Any]:
    config_path = source_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    config.setdefault("model_type", "modernbert")
    config.setdefault("architectures", ["ModernBertForMaskedLM"])

    config["vocab_size"] = vocab_size
    config["bos_token_id"] = EXPECTED_SPECIAL_IDS["bos_token_id"]
    config["pad_token_id"] = EXPECTED_SPECIAL_IDS["pad_token_id"]
    config["eos_token_id"] = EXPECTED_SPECIAL_IDS["eos_token_id"]
    config["unk_token_id"] = EXPECTED_SPECIAL_IDS["unk_token_id"]
    config["mask_token_id"] = EXPECTED_SPECIAL_IDS["mask_token_id"]

    config["cls_token_id"] = EXPECTED_SPECIAL_IDS["bos_token_id"]
    config["sep_token_id"] = EXPECTED_SPECIAL_IDS["eos_token_id"]

    config.pop("auto_map", None)

    run_args = load_json_if_exists(run_dir / "run_args.json")
    if "max_seq_length" in run_args:
        config.setdefault("max_position_embeddings", int(run_args["max_seq_length"]))

    return config


def load_direct_ape_tokenizer(tmp: Path):
    import importlib.util

    tokenizer_py = tmp / "tokenization_ape.py"
    spec = importlib.util.spec_from_file_location("tokenization_ape", tokenizer_py)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import tokenizer code from {tokenizer_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.APEPreTrainedTokenizer.from_pretrained(str(tmp))


def validate_direct_ape_tokenizer(tmp: Path, expected_vocab_size: int) -> None:
    import importlib.util

    tokenizer_py = tmp / "tokenization_ape.py"
    spec = importlib.util.spec_from_file_location("tokenization_ape", tokenizer_py)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import tokenizer code from {tokenizer_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tokenizer = module.APEPreTrainedTokenizer.from_pretrained(str(tmp))

    if tokenizer.vocab_size != expected_vocab_size:
        raise ValueError(
            f"Direct tokenizer vocab mismatch: {tokenizer.vocab_size} != {expected_vocab_size}"
        )

    if tokenizer.model_max_length != MODEL_MAX_LENGTH:
        raise ValueError(
            f"Direct tokenizer max length mismatch: {tokenizer.model_max_length} != {MODEL_MAX_LENGTH}"
        )

    print(
        f"[validate] direct APE tokenizer OK: vocab_size={tokenizer.vocab_size}, "
        f"max_length={tokenizer.model_max_length}",
        flush=True,
    )


def build_quickstart_output(source_dir: Path, example_selfies: str, hidden_size: Any) -> str:
    def fallback_output() -> str:
        return (
            "Output:\n\n"
            "```text\n"
            "Token IDs:\n[computed when generating the model card]\n\n"
            "Tokens:\n[computed when generating the model card]\n\n"
            f"Embedding shape: (1, {hidden_size})\n"
            "Embedding first 5 values:\n[computed when generating the model card]\n"
            "```\n\n"
        )

    if (
        not (source_dir / "model.safetensors").exists()
        or not (source_dir / "tokenization_ape.py").exists()
    ):
        return fallback_output()

    try:
        import contextlib
        import io

        import torch
        from transformers import AutoModelForMaskedLM

        tokenizer = load_direct_ape_tokenizer(source_dir)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = AutoModelForMaskedLM.from_pretrained(
                source_dir,
                local_files_only=True,
            ).eval()

        inputs = tokenizer(example_selfies, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
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

        token_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        embedding_shape = tuple(embedding.shape)
        embedding_head = [round(value, 4) for value in embedding[0, :5].tolist()]

    except Exception:
        return fallback_output()

    return (
        "Output:\n\n"
        "```text\n"
        f"Token IDs:\n{token_ids}\n\n"
        f"Tokens:\n{tokens}\n\n"
        f"Embedding shape: {embedding_shape}\n"
        f"Embedding first 5 values:\n{embedding_head}\n"
        "```\n\n"
    )


def get_example_sequence_length(source_dir: Path, example_selfies: str) -> int | str:
    try:
        if not (source_dir / "tokenization_ape.py").exists():
            return "sequence_length"
        tokenizer = load_direct_ape_tokenizer(source_dir)
        inputs = tokenizer(example_selfies, return_tensors="pt")
        return int(inputs["input_ids"].shape[1])
    except Exception:
        return "sequence_length"


def find_tokenizer_vocab(source_dir: Path, run_dir: Path) -> Path:
    candidates = [
        source_dir / "ape_tokenizer" / "vocab.json",
        source_dir / "vocab.json",
        run_dir / "final_model" / "ape_tokenizer" / "vocab.json",
        run_dir / "vocab.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find tokenizer vocab.json. Checked:\n{checked}")


def find_tokenization_code(source_dir: Path, run_dir: Path) -> Path:
    candidates = [
        repo_root() / "src" / "modernmolbert" / "tokenization_ape.py",
        source_dir / "ape_tokenizer" / "tokenization_ape.py",
        source_dir / "tokenization_ape.py",
        run_dir / "final_model" / "ape_tokenizer" / "tokenization_ape.py",
        run_dir / "final_model" / "tokenization_ape.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find tokenization_ape.py. Checked:\n{checked}")


def read_vocab_size(vocab_path: Path) -> int:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict):
        raise ValueError(f"Expected vocab JSON object at {vocab_path}")
    return len(vocab)


def write_tokenizer_config(tmp: Path) -> None:
    tokenizer_config_path = tmp / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json: {tokenizer_config_path}")

    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

    tokenizer_config.pop("tokenizer_class", None)
    tokenizer_config.update(
        {
            "representation": "SELFIES",
            "model_max_length": MODEL_MAX_LENGTH,
            "model_input_names": ["input_ids", "attention_mask"],
            "use_fast": False,
            "auto_map": {
                "AutoTokenizer": [
                    "tokenization_ape.APEPreTrainedTokenizer",
                    None,
                ],
            },
        }
    )

    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stage_tokenizer_files(
    source_dir: Path,
    run_dir: Path,
    tmp: Path,
    vocab_path: Path,
) -> None:
    from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

    tokenizer = APEPreTrainedTokenizer(
        representation="SELFIES",
        model_max_length=MODEL_MAX_LENGTH,
    )
    tokenizer.load_vocabulary_file(vocab_path)
    tokenizer.save_pretrained(str(tmp))

    write_tokenizer_config(tmp)
    shutil.copy(tmp / "vocab.json", tmp / "selfies_vocab.json")

    tokenization_code = find_tokenization_code(source_dir, run_dir)
    shutil.copy(tokenization_code, tmp / "tokenization_ape.py")

    for metadata_name in (
        "tokenizer_metadata.json",
        "ape_tokenizer_metadata.json",
        "metadata.json",
    ):
        candidates = [
            source_dir / "ape_tokenizer" / metadata_name,
            source_dir / metadata_name,
            run_dir / "final_model" / "ape_tokenizer" / metadata_name,
            run_dir / "final_model" / metadata_name,
            run_dir / metadata_name,
        ]

        for candidate in candidates:
            if candidate.exists():
                shutil.copy(candidate, tmp / metadata_name)
                break


def build_readme(source_dir: Path, run_dir: Path, repo_id: str, vocab_size: int) -> str:

    config = load_and_patch_config(
        source_dir,
        run_dir,
        vocab_size=vocab_size,
    )
    frontmatter = (
        "---\n"
        "license: mit\n"
        "library_name: transformers\n"
        "pipeline_tag: fill-mask\n"
        "tags:\n"
        "- chemistry\n"
        "- molecules\n"
        "- selfies\n"
        "- ape-tokenizer\n"
        "- modernbert\n"
        "- masked-language-modeling\n"
        "---\n"
    )

    hidden = config.get("hidden_size", "-")
    sequence_length = get_example_sequence_length(source_dir, EXAMPLE_SELFIES)

    model_details = (
        "## Model Details\n\n"
        "- **Developed by:** Hauser Group, Department of Drug Design and "
        "Pharmacology, University of Copenhagen\n"
        "- **Model type:** ModernBERT encoder &mdash; molecular embedding model "
        "trained with masked language modeling\n"
        "- **Input representation:** SELFIES (convert SMILES first; see below)\n"
        "- **Tokenizer:** Atom Pair Encoding (APE) over SELFIES primitives\n"
        "- **Pre-training data:** ChEMBL 36 (~2.4M unique small molecules)\n"
        "- **License:** MIT\n"
        "- **Repository:** https://github.com/HauserGroup/ModernMolBERT\n\n"
        "| field | value |\n"
        "|-------|-------|\n"
        f"| model_type | {config.get('model_type', 'modernbert')} |\n"
        f"| vocab_size | {config.get('vocab_size', '-')} |\n"
        f"| hidden_size | {hidden} |\n"
        f"| num_hidden_layers | {config.get('num_hidden_layers', '-')} |\n"
        f"| num_attention_heads | {config.get('num_attention_heads', '-')} |\n"
        f"| intermediate_size | {config.get('intermediate_size', '-')} |\n"
        f"| max_position_embeddings | {config.get('max_position_embeddings', '-')} |\n"
    )

    quickstart_output = build_quickstart_output(source_dir, EXAMPLE_SELFIES, hidden)

    # Minimal SELFIES tokenize + embedding example. No SMILES->SELFIES conversion step.
    quickstart = (
        "## How to Get Started with the Model\n\n"
        "The model consumes **SELFIES** strings tokenized with the APE "
        "tokenizer. For molecular representation learning, mean-pool the final "
        "hidden states over non-special SELFIES tokens:\n\n"
        "```python\n"
        "# pip install transformers torch\n"
        "import torch\n"
        "from transformers import AutoModel, AutoTokenizer\n\n"
        f"repo = '{repo_id}'\n"
        "model = AutoModel.from_pretrained(repo).eval()\n"
        "tokenizer = AutoTokenizer.from_pretrained(\n"
        "    repo,\n"
        "    subfolder='ape_tokenizer',\n"
        "    trust_remote_code=True,\n"
        "    use_fast=False,\n"
        ")\n\n"
        "# A SELFIES string (one bracketed token per primitive); here psilocybin.\n"
        f"selfies = '{EXAMPLE_SELFIES}'\n\n"
        "inputs = tokenizer(selfies, return_tensors='pt')\n"
        "with torch.no_grad():\n"
        "    outputs = model(**inputs)\n"
        "    hidden = outputs.last_hidden_state\n"
        "    content_mask = inputs['attention_mask'].bool()\n"
        "    for token_id in [\n"
        "        tokenizer.bos_token_id,\n"
        "        tokenizer.eos_token_id,\n"
        "        tokenizer.pad_token_id,\n"
        "        tokenizer.unk_token_id,\n"
        "        tokenizer.mask_token_id,\n"
        "    ]:\n"
        "        if token_id is not None:\n"
        "            content_mask = content_mask & inputs['input_ids'].ne(token_id)\n"
        "    empty_rows = content_mask.sum(dim=1).eq(0)\n"
        "    if empty_rows.any():\n"
        "        content_mask[empty_rows] = inputs['attention_mask'].bool()[empty_rows]\n"
        "    mask = content_mask.unsqueeze(-1).to(hidden.dtype)\n"
        "    embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)\n\n"
        "tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])\n"
        "embedding_preview = [round(x, 4) for x in embedding[0, :5].tolist()]\n"
        "print(f\"Token IDs:\\n{inputs['input_ids'][0].tolist()}\\n\")\n"
        'print(f"Tokens:\\n{tokens}\\n")\n'
        'print(f"Embedding shape: {tuple(embedding.shape)}")\n'
        'print(f"Embedding first 5 values:\\n{embedding_preview}")\n'
        "```\n\n"
        + quickstart_output
        + "If you start from SMILES, convert it to SELFIES first (e.g. the "
        "[`selfies`](https://github.com/aspuru-guzik-group/selfies) package: "
        '`selfies.encoder("CC(=O)Oc1ccccc1C(=O)O")`).\n\n'
        "For masked-token predictions, load the same checkpoint with "
        "`AutoModelForMaskedLM`:\n\n"
        "```python\n"
        "from transformers import AutoModelForMaskedLM\n\n"
        "mlm = AutoModelForMaskedLM.from_pretrained(repo)\n"
        "logits = mlm(**inputs).logits\n"
        'print(f"Logits shape: {tuple(logits.shape)}")\n'
        "```\n\n"
        "Output:\n\n"
        "```text\n"
        f"Logits shape: (1, {sequence_length}, {config.get('vocab_size', '-')})\n"
        "```\n\n"
        "> Current Transformers releases disable custom root tokenizers for "
        "`model_type='modernbert'` before loading `auto_map`, so the tokenizer "
        "must be loaded from `ape_tokenizer/`. The root tokenizer files are also "
        "shipped for forward compatibility.\n"
    )

    uses = (
        "## Uses\n\n"
        "- **Direct use:** molecular embeddings for property prediction, "
        "similarity search, clustering, and retrieval; masked-token fill-in.\n"
        "- **Downstream use:** fine-tuning for molecular classification or "
        "regression on SELFIES inputs.\n"
        "- **Out of scope:** natural-language text; generating valid SMILES; "
        "3D/conformer-dependent tasks.\n\n"
        "## Bias, Risks, and Limitations\n\n"
        "Pre-trained only on drug-like ChEMBL 36 chemistry; may not generalize to "
        "natural products, agrochemicals, fragments, or other under-represented "
        "chemical space. Performance depends on the downstream task and "
        "adaptation strategy. No access to 3D/conformer information.\n"
    )

    return (
        frontmatter
        + f"\n# {repo_id}\n\n"
        + "ModernMolBERT is a compact ModernBERT encoder pre-trained from scratch "
        "with masked language modeling on ~2.4M SELFIES strings from ChEMBL 36, "
        "using a chemically aware Atom Pair Encoding (APE) tokenizer. It expects "
        "SELFIES input and produces general-purpose molecular embeddings.\n\n"
        + model_details
        + "\n"
        + quickstart
        + "\n"
        + uses
    )


def build_staging_dir(
    source_dir: Path, run_dir: Path, repo_id: str, tmp: Path, masking_strategy: str = "standard"
) -> None:
    shutil.copy(source_dir / "model.safetensors", tmp / "model.safetensors")

    vocab_path = find_tokenizer_vocab(source_dir, run_dir)
    vocab_size = read_vocab_size(vocab_path)
    config = load_and_patch_config(source_dir, run_dir, vocab_size=vocab_size)
    (tmp / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stage_tokenizer_files(source_dir, run_dir, tmp, vocab_path)
    tokenizer_tmp = tmp / "ape_tokenizer"
    tokenizer_tmp.mkdir(parents=True, exist_ok=True)
    stage_tokenizer_files(source_dir, run_dir, tokenizer_tmp, vocab_path)

    for name in (
        "run_args.json",
        "trainer_state.json",
        "tokenizer_metadata.json",
        "ape_tokenizer_metadata.json",
        "eval_results.json",
        "train_results.json",
        "all_results.json",
        "best_span_run.json",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, tmp / name)

    write_collator_config(tmp, masking_strategy)

    (tmp / "README.md").write_text(
        build_readme(
            tmp,
            run_dir,
            repo_id,
            vocab_size=vocab_size,
        ),
        encoding="utf-8",
    )
    remove_pycache_dirs(tmp)


def validate_staged_files(tmp: Path) -> None:
    required = [
        "config.json",
        "model.safetensors",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_ape.py",
        "README.md",
    ]

    missing = [name for name in required if not (tmp / name).exists()]
    if missing:
        raise FileNotFoundError(f"Staging directory is missing files: {missing}")


def write_collator_config(tmp: Path, masking_strategy: str) -> None:
    """Write collator_config.json with the training-time defaults for this strategy."""
    config = dict(MASKING_DEFAULTS[masking_strategy])
    config["_note"] = (
        "Collator parameters used during pre-training. "
        "Change masking_strategy to 'standard', 'span', or 'hetero_span' "
        "and adjust the corresponding parameters when fine-tuning."
    )
    (tmp / "collator_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def remove_pycache_dirs(path: Path) -> None:
    for pycache in path.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def validate_tokenizer_config(tmp: Path) -> None:
    tokenizer_config_path = tmp / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

    if "tokenizer_class" in tokenizer_config:
        raise ValueError(
            "tokenizer_config.json still contains tokenizer_class; "
            "this can force Transformers down the wrong tokenizer path."
        )

    expected_auto_map = {
        "AutoTokenizer": [
            "tokenization_ape.APEPreTrainedTokenizer",
            None,
        ],
    }

    if tokenizer_config.get("auto_map") != expected_auto_map:
        raise ValueError(f"Unexpected tokenizer auto_map: {tokenizer_config.get('auto_map')!r}")

    if tokenizer_config.get("model_max_length") != MODEL_MAX_LENGTH:
        raise ValueError(
            f"Unexpected model_max_length={tokenizer_config.get('model_max_length')!r}"
        )

    if tokenizer_config.get("use_fast") is not False:
        raise ValueError("tokenizer_config.json should contain use_fast=false")


def validate_staged_model(tmp: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    print("[validate] loading config", flush=True)
    config = AutoConfig.from_pretrained(tmp, local_files_only=True)

    print("[validate] loading direct APE tokenizer", flush=True)
    tokenizer = load_direct_ape_tokenizer(tmp)

    print("[validate] loading ape_tokenizer AutoTokenizer", flush=True)
    auto_tokenizer = AutoTokenizer.from_pretrained(
        tmp / "ape_tokenizer",
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )

    if auto_tokenizer.__class__.__name__ != "APEPreTrainedTokenizer":
        raise TypeError(
            f"Expected ape_tokenizer AutoTokenizer to load APEPreTrainedTokenizer, got "
            f"{type(auto_tokenizer)!r}"
        )

    if tokenizer.model_max_length != MODEL_MAX_LENGTH:
        raise ValueError(
            f"Tokenizer max length mismatch: tokenizer={tokenizer.model_max_length}, "
            f"expected={MODEL_MAX_LENGTH}"
        )

    if tokenizer.bos_token_id != EXPECTED_SPECIAL_IDS["bos_token_id"]:
        raise ValueError(f"bos_token_id mismatch: {tokenizer.bos_token_id}")

    if tokenizer.eos_token_id != EXPECTED_SPECIAL_IDS["eos_token_id"]:
        raise ValueError(f"eos_token_id mismatch: {tokenizer.eos_token_id}")

    if tokenizer.unk_token_id != EXPECTED_SPECIAL_IDS["unk_token_id"]:
        raise ValueError(f"unk_token_id mismatch: {tokenizer.unk_token_id}")

    if tokenizer.mask_token_id != EXPECTED_SPECIAL_IDS["mask_token_id"]:
        raise ValueError(f"mask_token_id mismatch: {tokenizer.mask_token_id}")

    print("[validate] loading model", flush=True)
    model = AutoModelForMaskedLM.from_pretrained(tmp, local_files_only=True)
    model.eval()

    if config.model_type != "modernbert":
        raise ValueError(f"Unexpected model_type={config.model_type!r}")

    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: tokenizer={tokenizer.vocab_size}, "
            f"config={config.vocab_size}"
        )

    if auto_tokenizer.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"ape_tokenizer AutoTokenizer vocab mismatch: auto={auto_tokenizer.vocab_size}, "
            f"direct={tokenizer.vocab_size}"
        )

    if tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError(
            f"pad_token_id mismatch: tokenizer={tokenizer.pad_token_id}, "
            f"config={config.pad_token_id}"
        )

    example = (
        "[C][C][=C][C][Branch1][=N][N][N][=C][C][=Branch1][C][=O][NH1][C][Ring1][#Branch1][=O]"
    )

    batch = tokenizer(
        [example],
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    print("[validate] running forward pass", flush=True)
    with torch.no_grad():
        out = model(**batch)

    if not torch.isfinite(out.logits).all():
        raise ValueError("Model forward pass produced non-finite logits.")

    n_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"[validate] OK model: {n_params / 1e6:.1f}M parameters", flush=True)
    print(
        f"[validate] OK tokenizer: {type(tokenizer)}; "
        f"vocab_size={tokenizer.vocab_size}; "
        f"max_length={tokenizer.model_max_length}",
        flush=True,
    )
    print(f"[validate] OK logits shape={tuple(out.logits.shape)}", flush=True)


def prepare_staging_dir(keep_staging_dir: Path | None) -> tuple[Path, Callable[[], None]]:
    if keep_staging_dir is not None:
        tmp = keep_staging_dir

        if tmp.exists():
            shutil.rmtree(tmp)

        tmp.mkdir(parents=True, exist_ok=True)

        def cleanup() -> None:
            return None

        return tmp, cleanup

    temp_dir = tempfile.TemporaryDirectory()
    tmp = Path(temp_dir.name)

    def cleanup() -> None:
        temp_dir.cleanup()

    return tmp, cleanup


def upload_model_to_hub(
    run_dir: Path,
    repo_id: str,
    checkpoint: str = "final",
    private: bool = False,
    commit_message: str = "Upload trained ModernMolBERT checkpoint",
    token: str | None = None,
    dry_run: bool = False,
    keep_staging_dir: Path | None = None,
    api: HfApi | None = None,
    masking_strategy: str = "standard",
) -> dict[str, Any]:
    if not run_dir.is_absolute():
        run_dir = repo_root() / run_dir

    source_dir = resolve_source_dir(run_dir, checkpoint)
    print(f"Source: {source_dir}", flush=True)

    tmp, cleanup = prepare_staging_dir(keep_staging_dir)
    staged_names: list[str] = []

    try:
        print(f"[upload] staging directory: {tmp}", flush=True)

        build_staging_dir(source_dir, run_dir, repo_id, tmp, masking_strategy=masking_strategy)
        validate_staged_files(tmp)

        validate_staged_model(tmp)
        remove_pycache_dirs(tmp)

        staged = sorted(tmp.iterdir())
        staged_names = [path.name for path in staged]
        print(f"[upload] staged {len(staged)} files: {staged_names}", flush=True)

        if dry_run:
            print(f"Dry run: skipped upload to https://huggingface.co/{repo_id}", flush=True)
        else:
            if api is None:
                api = HfApi(token=token)
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )

    finally:
        cleanup()
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "run_dir": str(run_dir),
        "source_dir": str(source_dir),
        "checkpoint": checkpoint,
        "private": private,
        "uploaded": not dry_run,
        "staged_files": staged_names,
        "staging_dir": str(tmp) if keep_staging_dir is not None else None,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None

    if args.hf_login:
        from huggingface_hub import login

        login(token=token)
        token = None

    result = upload_model_to_hub(
        run_dir=args.run_dir,
        repo_id=args.repo_id,
        checkpoint=args.checkpoint,
        private=args.private,
        commit_message=args.commit_message,
        token=token,
        dry_run=args.dry_run,
        keep_staging_dir=args.keep_staging_dir,
        masking_strategy=args.masking_strategy,
    )

    print(f"Done — {result['url']}", flush=True)


if __name__ == "__main__":
    main()
