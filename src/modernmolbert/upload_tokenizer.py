# upload_tokenizer.py — run from repo root

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


REPO_ID = "HauserGroup/ApeTokenizer-SELFIES"
TMP = Path("./tmp-hf-tokenizer")

VOCAB_PATH = Path("tokenizer/chembl36_selfies_2m_ape_max2_min3000.json")
METADATA_PATH = VOCAB_PATH.with_name(VOCAB_PATH.stem + ".metadata.json")
TOKENIZER_CODE = Path("src/modernmolbert/tokenization_ape.py")

MODEL_MAX_LENGTH = 256


def clean_tmp() -> None:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing tokenizer metadata: {METADATA_PATH}")

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def verify_metadata(metadata: dict) -> None:
    expected = {
        "representation": "SELFIES",
        "max_merge_pieces": 2,
        "min_freq_for_merge": 3000,
        "tokenizer_train_size": 2_000_000,
        "vocab_size": 631,
    }

    for key, value in expected.items():
        observed = metadata.get(key)
        if observed != value:
            raise ValueError(
                f"Unexpected metadata value for {key}: {observed!r}; expected {value!r}"
            )

    special_ids = metadata.get("special_ids", {})
    expected_special_ids = {
        "bos_token": 0,
        "pad_token": 1,
        "eos_token": 2,
        "unk_token": 3,
        "mask_token": 4,
    }

    if special_ids != expected_special_ids:
        raise ValueError(
            f"Unexpected special_ids: {special_ids!r}; expected {expected_special_ids!r}"
        )


def verify_saved_tokenizer() -> None:
    config_path = TMP / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"save_pretrained did not create {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    saved_max_length = config.get("model_max_length")
    if saved_max_length != MODEL_MAX_LENGTH:
        raise ValueError(
            f"Wrong tokenizer_config.json model_max_length: "
            f"{saved_max_length!r}; expected {MODEL_MAX_LENGTH}"
        )

    loaded = AutoTokenizer.from_pretrained(
        str(TMP),
        trust_remote_code=True,
    )

    if loaded.model_max_length != MODEL_MAX_LENGTH:
        raise ValueError(
            f"Reloaded tokenizer has model_max_length={loaded.model_max_length}; "
            f"expected {MODEL_MAX_LENGTH}"
        )

    if loaded.vocab_size != 631:
        raise ValueError(f"Unexpected loaded vocab_size={loaded.vocab_size}; expected 631")

    example = (
        "[C][C][=C][C][Branch1][=N][N][N][=C][C][=Branch1][C][=O][NH1][C][Ring1][#Branch1][=O]"
    )

    encoded = loaded(
        example,
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
    )

    if len(encoded["input_ids"]) > MODEL_MAX_LENGTH:
        raise ValueError(
            f"Tokenizer produced {len(encoded['input_ids'])} tokens despite "
            f"max_length={MODEL_MAX_LENGTH}"
        )

    print(f"Verified model_max_length={loaded.model_max_length}")
    print(f"Verified vocab_size={loaded.vocab_size}")
    print(f"Verified example length={len(encoded['input_ids'])}")


def main() -> None:
    clean_tmp()

    metadata = load_metadata()
    verify_metadata(metadata)

    tokenizer = APEPreTrainedTokenizer(
        representation="SELFIES",
        model_max_length=MODEL_MAX_LENGTH,
    )
    tokenizer.load_vocabulary_file(str(VOCAB_PATH))

    tokenizer.save_pretrained(str(TMP))

    shutil.copy(TOKENIZER_CODE, TMP / "tokenization_ape.py")

    # Keep your training metadata in the HF repo as documentation.
    shutil.copy(METADATA_PATH, TMP / "metadata.json")

    verify_saved_tokenizer()

    api = HfApi()

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(TMP),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Add APE SELFIES tokenizer with max length {MODEL_MAX_LENGTH}",
    )

    shutil.rmtree(TMP)
    print(f"Done — https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
