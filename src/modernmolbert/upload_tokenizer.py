"""Stage and upload an APE tokenizer to Hugging Face Hub."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


DEFAULT_REPO_ID = "HauserGroup/ApeTokenizer-SELFIES"
DEFAULT_TMP = Path("./tmp-hf-tokenizer")
DEFAULT_VOCAB_PATH = Path("tokenizer/chembl36_selfies_2m_ape_max2_min3000.json")
TOKENIZER_CODE = Path("src/modernmolbert/tokenization_ape.py")
DEFAULT_MODEL_MAX_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--vocab_path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("--staging_dir", type=Path, default=DEFAULT_TMP)
    parser.add_argument("--model_max_length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--commit_message")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the HuggingFace repo as private.",
    )
    parser.add_argument(
        "--hf_login",
        action="store_true",
        help="Call huggingface_hub.login() before uploading. Reads HF_TOKEN_ORG or HF_TOKEN from env or .env.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Stage and validate files without creating or uploading to the Hub.",
    )
    parser.add_argument(
        "--keep_staging_dir",
        action="store_true",
        help="Do not remove the staging directory after a successful run.",
    )
    return parser.parse_args()


def clean_tmp(staging_dir: Path) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)


def resolve_metadata_path(vocab_path: Path, metadata_path: Path | None) -> Path:
    return metadata_path or vocab_path.with_name(vocab_path.stem + ".metadata.json")


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing tokenizer metadata: {metadata_path}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_metadata(metadata: dict, vocab_path: Path) -> str:
    representation = str(metadata.get("representation", "")).upper()
    if representation not in {"SELFIES", "SMILES"}:
        raise ValueError(f"Unexpected tokenizer representation: {representation!r}")

    for key in [
        "max_merge_pieces",
        "min_freq_for_merge",
        "tokenizer_train_size",
        "vocab_size",
    ]:
        if key not in metadata:
            raise ValueError(f"Missing metadata value: {key}")

    recorded_sha = str(metadata.get("tokenizer_sha256", ""))
    observed_sha = file_sha256(vocab_path)
    if recorded_sha != observed_sha:
        raise ValueError(
            f"Tokenizer SHA256 mismatch: metadata={recorded_sha}; observed={observed_sha}"
        )

    expected_special_ids = {
        "bos_token": 0,
        "pad_token": 1,
        "eos_token": 2,
        "unk_token": 3,
        "mask_token": 4,
    }
    special_ids = metadata.get("special_ids", {})
    if special_ids != expected_special_ids:
        raise ValueError(
            f"Unexpected special_ids: {special_ids!r}; expected {expected_special_ids!r}"
        )

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    observed_vocab_size = len(vocab)
    expected_vocab_size = int(metadata["vocab_size"])
    if observed_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Unexpected vocab_size: {observed_vocab_size}; expected {expected_vocab_size}"
        )

    for token, token_id in {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
    }.items():
        if vocab.get(token) != token_id:
            raise ValueError(
                f"Unexpected id for {token}: {vocab.get(token)!r}; expected {token_id}"
            )

    return representation


def validation_example(representation: str) -> str:
    if representation == "SMILES":
        return "CC(=O)Nc1ccc(O)cc1"
    return "[C][C][=C][C][Branch1][=N][N][N][=C][C][=Branch1][C][=O][NH1][C][Ring1][#Branch1][=O]"


def verify_saved_tokenizer(
    staging_dir: Path,
    *,
    representation: str,
    expected_vocab_size: int,
    model_max_length: int,
) -> None:
    config_path = staging_dir / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"save_pretrained did not create {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    saved_representation = str(config.get("representation", "")).upper()
    if saved_representation != representation:
        raise ValueError(
            f"Wrong tokenizer_config.json representation: "
            f"{saved_representation!r}; expected {representation!r}"
        )

    saved_max_length = config.get("model_max_length")
    if saved_max_length != model_max_length:
        raise ValueError(
            f"Wrong tokenizer_config.json model_max_length: "
            f"{saved_max_length!r}; expected {model_max_length}"
        )

    loaded = AutoTokenizer.from_pretrained(
        str(staging_dir),
        trust_remote_code=True,
    )

    if loaded.representation != representation:
        raise ValueError(
            f"Reloaded tokenizer has representation={loaded.representation}; "
            f"expected {representation}"
        )

    if loaded.model_max_length != model_max_length:
        raise ValueError(
            f"Reloaded tokenizer has model_max_length={loaded.model_max_length}; "
            f"expected {model_max_length}"
        )

    if loaded.vocab_size != expected_vocab_size:
        raise ValueError(
            f"Unexpected loaded vocab_size={loaded.vocab_size}; expected {expected_vocab_size}"
        )

    example = validation_example(representation)
    encoded = loaded(
        example,
        truncation=True,
        max_length=model_max_length,
    )

    if len(encoded["input_ids"]) > model_max_length:
        raise ValueError(
            f"Tokenizer produced {len(encoded['input_ids'])} tokens despite "
            f"max_length={model_max_length}"
        )

    print(f"Verified representation={loaded.representation}")
    print(f"Verified model_max_length={loaded.model_max_length}")
    print(f"Verified vocab_size={loaded.vocab_size}")
    print(f"Verified example length={len(encoded['input_ids'])}")


def _write_readme(staging_dir: Path, repo_id: str) -> None:
    """Write README.md using the canonical card generator in write_model_cards.py."""
    root = Path(__file__).resolve().parents[2]
    write_model_cards = root / "scripts" / "write_model_cards.py"
    if not write_model_cards.exists():
        print(f"WARNING: {write_model_cards} not found; skipping README.md", file=sys.stderr)
        return

    import importlib.util

    spec = importlib.util.spec_from_file_location("write_model_cards", write_model_cards)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    (staging_dir / "README.md").write_text(mod.tokenizer_card(), encoding="utf-8")
    print(f"wrote README.md to {staging_dir}")


def upload_tokenizer_to_hub(
    *,
    repo_id: str,
    vocab_path: Path,
    metadata_path: Path | None = None,
    staging_dir: Path = DEFAULT_TMP,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    keep_staging_dir: bool = False,
) -> None:
    vocab_path = vocab_path.resolve()
    metadata_path = resolve_metadata_path(vocab_path, metadata_path)
    clean_tmp(staging_dir)

    metadata = load_metadata(metadata_path)
    representation = verify_metadata(metadata, vocab_path)
    tokenizer = APEPreTrainedTokenizer(
        representation=representation,
        model_max_length=model_max_length,
    )
    tokenizer.load_vocabulary_file(str(vocab_path))

    tokenizer.save_pretrained(str(staging_dir))

    shutil.copy(TOKENIZER_CODE, staging_dir / "tokenization_ape.py")

    # Keep your training metadata in the HF repo as documentation.
    shutil.copy(metadata_path, staging_dir / "metadata.json")
    shutil.copy(metadata_path, staging_dir / "tokenizer_metadata.json")
    shutil.copy(metadata_path, staging_dir / "ape_tokenizer_metadata.json")

    # Write the model card using the canonical generator in write_model_cards.py.
    _write_readme(staging_dir, repo_id)

    verify_saved_tokenizer(
        staging_dir,
        representation=representation,
        expected_vocab_size=int(metadata["vocab_size"]),
        model_max_length=model_max_length,
    )

    if dry_run:
        print(f"Dry run staged tokenizer at {staging_dir}")
        return

    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
        or f"Add APE {representation} tokenizer with max length {model_max_length}",
    )

    if not keep_staging_dir:
        shutil.rmtree(staging_dir)
    print(f"Done: https://huggingface.co/{repo_id}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None

    if args.hf_login:
        from huggingface_hub import login

        login(token=token)
        token = None

    upload_tokenizer_to_hub(
        repo_id=args.repo_id,
        vocab_path=args.vocab_path,
        metadata_path=args.metadata_path,
        staging_dir=args.staging_dir,
        model_max_length=args.model_max_length,
        private=args.private,
        commit_message=args.commit_message,
        token=token,
        dry_run=args.dry_run,
        keep_staging_dir=args.keep_staging_dir,
    )


if __name__ == "__main__":
    main()
