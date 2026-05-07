"""
Shared utilities for APETokenizer interaction and dataset loading.
"""

from typing import Any

import torch
from datasets import IterableDataset, load_dataset
from tqdm.auto import tqdm

from modernmolbert.ape_tokenizer import APETokenizer


SPECIAL_TOKENS: dict[str, str] = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def normalize_sequence(example: dict[str, Any], representation: str) -> str | None:
    seq = example.get(representation)
    if seq is None:
        return None
    seq = str(seq).strip()
    return seq if seq else None


def get_streaming_dataset(
    dataset_name: str, seed: int, buffer_size: int
) -> IterableDataset:
    ds = load_dataset(dataset_name, split="train", streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer_size)


def collect_corpus_for_tokenizer(
    dataset_name: str,
    representation: str,
    n: int,
    seed: int,
    buffer_size: int,
) -> list[str]:
    ds = get_streaming_dataset(dataset_name, seed=seed, buffer_size=buffer_size)
    corpus: list[str] = []

    pbar = tqdm(total=n, desc=f"Collecting {representation} corpus for APE tokenizer")
    for row in ds:
        seq = normalize_sequence(row, representation)
        if seq is None:
            continue
        corpus.append(seq)
        pbar.update(1)
        if len(corpus) >= n:
            break
    pbar.close()

    if not corpus:
        raise RuntimeError("Tokenizer corpus is empty. Check dataset column names.")

    return corpus


# ---------------------------------------------------------------------------
# Tokenizer utilities
# ---------------------------------------------------------------------------


def tokenizer_vocab_size(tokenizer: APETokenizer) -> int:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict):
            return len(vocab)

    for attr in ["vocab", "vocabulary", "token_to_id", "token2id"]:
        if hasattr(tokenizer, attr):
            value = getattr(tokenizer, attr)
            if isinstance(value, dict):
                return len(value)

    raise AttributeError(
        "Could not infer APETokenizer vocabulary size. "
        "Inspect the tokenizer object and adjust tokenizer_vocab_size()."
    )


def token_id(tokenizer: APETokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        out = tokenizer.convert_tokens_to_ids([token])
        return int(out[0] if isinstance(out, list) else out)

    encoded = tokenizer(token, add_special_tokens=False)
    ids = encoded["input_ids"]

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} resolved to {ids}, expected one ID.")

    return int(ids[0])


def resolve_special_ids(tokenizer: APETokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for name, token in SPECIAL_TOKENS.items():
        try:
            ids[name] = token_id(tokenizer, token)
        except Exception:
            attr_name = name + "_id"
            if hasattr(tokenizer, attr_name):
                ids[name] = int(getattr(tokenizer, attr_name))
            else:
                raise RuntimeError(
                    f"Could not resolve ID for special token {token!r}. "
                    "Check APETokenizer special-token names."
                )
    return ids


def encode_sequence(
    tokenizer: APETokenizer, seq: str, max_seq_length: int
) -> dict[str, list[int]]:
    encoded = tokenizer(
        seq,
        padding=False,
        max_length=max_seq_length,
        add_special_tokens=True,
        return_tensors=None,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.tolist()

    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]

    return {
        "input_ids": list(map(int, input_ids[:max_seq_length])),
        "attention_mask": list(map(int, attention_mask[:max_seq_length])),
    }
