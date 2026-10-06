"""Load molecular tokenizers from tokenizer files or saved checkpoints.

A tokenizer file is an APE vocabulary JSON or a BPE ``tokenizer.json``, with a
``.metadata.json`` next to it. The metadata records the algorithm and the
representation (SELFIES or SMILES); files written before BPE support have no
``algorithm`` field and are APE.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modernmolbert.utils import (
    SELFIES_REPRESENTATION,
    SMILES_REPRESENTATION,
    file_sha256,
    load_tokenizer_metadata,
    metadata_path_for_vocab,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

APE = "APE"
BPE = "BPE"
SMIRK = "SMIRK"
ALGORITHMS = (APE, BPE, SMIRK)
REPRESENTATIONS = (SELFIES_REPRESENTATION, SMILES_REPRESENTATION)


def tokenizer_algorithm(metadata: dict[str, Any]) -> str:
    algorithm = str(metadata.get("algorithm", APE)).upper()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown tokenizer algorithm {algorithm!r}; expected one of {ALGORITHMS}")
    return algorithm


def tokenizer_representation(metadata: dict[str, Any]) -> str:
    representation = str(metadata.get("representation", "")).upper()
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Tokenizer metadata representation {representation or '<missing>'!r} "
            f"is not one of {REPRESENTATIONS}"
        )
    return representation


def load_tokenizer(
    vocab_path: str | Path,
    metadata: dict[str, Any],
    *,
    model_max_length: int | None = None,
) -> "PreTrainedTokenizerBase":
    """Build the tokenizer described by ``metadata`` from its file."""
    representation = tokenizer_representation(metadata)
    algorithm = tokenizer_algorithm(metadata)
    if algorithm == SMIRK:
        if representation != SMILES_REPRESENTATION:
            raise ValueError("SMIRK tokenizer requires SMILES representation")
        try:
            from smirk import SmirkTokenizerFast
        except ImportError as exc:
            raise ImportError("Install smirk==0.3.0 for experimental SMIRK runs") from exc

        options = {} if model_max_length is None else {"model_max_length": model_max_length}
        return SmirkTokenizerFast(
            tokenizer_file=Path(vocab_path),
            template="[BOS] $0 [EOS]",
            representation=representation,
            **options,
        )
    if algorithm == BPE:
        from modernmolbert.tokenization.bpe import load_bpe_tokenizer

        return load_bpe_tokenizer(
            vocab_path, representation=representation, model_max_length=model_max_length
        )

    from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

    tokenizer = (
        APEPreTrainedTokenizer(representation=representation)
        if model_max_length is None
        else APEPreTrainedTokenizer(
            representation=representation, model_max_length=model_max_length
        )
    )
    tokenizer.load_vocabulary_file(vocab_path)
    return tokenizer


def load_verified_tokenizer(
    vocab_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    log: Callable[[str], None] = print,
) -> tuple["PreTrainedTokenizerBase", dict[str, Any], Path, Path]:
    """Load a tokenizer file after checking it against its metadata.

    Returns the tokenizer, its metadata, and both paths. A hash mismatch always
    fails; metadata without a recorded hash only warns.
    """
    vocab_path = Path(vocab_path)
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file not found: {vocab_path}\n"
            "Train one first with: python -m modernmolbert.train_tokenizer"
        )
    metadata_path = Path(metadata_path) if metadata_path else metadata_path_for_vocab(vocab_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Tokenizer metadata not found: {metadata_path}")
    metadata = load_tokenizer_metadata(metadata_path)

    recorded_sha = str(metadata.get("tokenizer_sha256", ""))
    actual_sha = file_sha256(vocab_path)
    if not recorded_sha:
        log("WARNING: tokenizer metadata has no tokenizer_sha256; skipping integrity check.")
    elif recorded_sha != actual_sha:
        raise ValueError(
            "Tokenizer hash mismatch between metadata and file. "
            f"metadata={recorded_sha} file={actual_sha}"
        )
    return load_tokenizer(vocab_path, metadata), metadata, vocab_path, metadata_path


def load_checkpoint_tokenizer(path: str | Path) -> tuple["PreTrainedTokenizerBase", str]:
    """Load the tokenizer saved with a checkpoint and the representation its model reads.

    ``path`` may be a checkpoint directory or a tokenizer file. APE checkpoints keep
    their tokenizer in ``ape_tokenizer/`` (custom code); BPE checkpoints keep a
    standard ``tokenizer.json`` at the root. A directory with only ``vocab.json``,
    or a bare APE vocabulary without metadata, as in the earliest runs, is APE.
    """
    path = Path(path)
    if path.is_file():
        metadata_path = metadata_path_for_vocab(path)
        metadata = (
            load_tokenizer_metadata(metadata_path)
            if metadata_path.is_file()
            else {"representation": SELFIES_REPRESENTATION}
        )
        return load_tokenizer(path, metadata), tokenizer_representation(metadata)
    if not path.is_dir():
        raise FileNotFoundError(f"Tokenizer path does not exist: {path}")

    ape_dir = path / "ape_tokenizer"
    if ape_dir.is_dir():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(ape_dir), trust_remote_code=True)
        if type(tokenizer).__name__ != "APEPreTrainedTokenizer":
            raise TypeError(f"Expected APEPreTrainedTokenizer, got {type(tokenizer)!r}")
        representation = getattr(tokenizer, "representation", SELFIES_REPRESENTATION)
        return tokenizer, tokenizer_representation({"representation": representation})

    if (path / "tokenizer.json").is_file():
        config_path = path / "tokenizer_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("tokenizer_class") == "SmirkTokenizerFast":
                from smirk import SmirkTokenizerFast

                # Transformers 5's AutoTokenizer falls back to TokenizersBackend,
                # which cannot deserialize SMIRK's custom pre-tokenizer. The
                # package's own loader restores that component correctly.
                tokenizer = SmirkTokenizerFast.from_pretrained(str(path))
                representation = tokenizer.init_kwargs.get("representation")
                if representation is None:
                    raise ValueError(f"{config_path} does not record a representation")
                return tokenizer, tokenizer_representation({"representation": representation})

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        representation = tokenizer.init_kwargs.get("representation")
        if representation is None:
            raise ValueError(f"{path / 'tokenizer_config.json'} does not record a representation")
        return tokenizer, tokenizer_representation({"representation": representation})

    vocab_json = path / "vocab.json"
    if vocab_json.is_file():
        config_path = path / "tokenizer_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        )
        metadata = {"representation": config.get("representation", SELFIES_REPRESENTATION)}
        return load_tokenizer(vocab_json, metadata), tokenizer_representation(metadata)

    raise FileNotFoundError(
        f"No tokenizer found in {path}. Expected ape_tokenizer/, tokenizer.json or vocab.json."
    )
