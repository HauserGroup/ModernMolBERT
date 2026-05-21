import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


EXAMPLES = [
    "[C][C][=C][C][Branch1][=N][N][N][=C][C][=Branch1][C][=O][NH1][C][Ring1][#Branch1][=O]",
    "[O][=C][Branch1][N][C][=C][C][=C][Branch1][C][Cl][C][=C][Ring1][#Branch1]",
    "[C][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C]",
]


def load_local_tokenizer(vocab_path: Path, model_max_length: int):
    tok = APEPreTrainedTokenizer(
        representation="SELFIES",
        model_max_length=model_max_length,
    )
    tok.load_vocabulary_file(vocab_path)
    return tok


def encode_batch(tok, examples, max_length):
    return tok(
        examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-tokenizer",
        default="HauserGroup/ApeTokenizer-SELFIES",
    )
    parser.add_argument(
        "--local-vocab",
        type=Path,
        default=Path("tokenizer/chembl36_selfies_2m_ape_max2_min3000.json"),
    )
    parser.add_argument("--model-max-length", type=int, default=256)
    args = parser.parse_args()

    local_tok = load_local_tokenizer(args.local_vocab, args.model_max_length)

    hf_tok = AutoTokenizer.from_pretrained(
        args.hf_tokenizer,
        trust_remote_code=True,
    )

    checks = {
        "model_max_length": (local_tok.model_max_length, hf_tok.model_max_length),
        "vocab_size": (local_tok.vocab_size, hf_tok.vocab_size),
        "pad_token_id": (local_tok.pad_token_id, hf_tok.pad_token_id),
        "bos_token_id": (local_tok.bos_token_id, hf_tok.bos_token_id),
        "eos_token_id": (local_tok.eos_token_id, hf_tok.eos_token_id),
        "unk_token_id": (local_tok.unk_token_id, hf_tok.unk_token_id),
        "mask_token_id": (local_tok.mask_token_id, hf_tok.mask_token_id),
    }

    for name, (local_value, hf_value) in checks.items():
        if local_value != hf_value:
            raise AssertionError(f"{name} mismatch: local={local_value!r}, hf={hf_value!r}")

    local_vocab = local_tok.get_vocab()
    hf_vocab = hf_tok.get_vocab()
    if local_vocab != hf_vocab:
        local_only = set(local_vocab) - set(hf_vocab)
        hf_only = set(hf_vocab) - set(local_vocab)
        raise AssertionError(
            f"Vocabulary mismatch. local_only={list(local_only)[:10]}, hf_only={list(hf_only)[:10]}"
        )

    local_batch = encode_batch(local_tok, EXAMPLES, args.model_max_length)
    hf_batch = encode_batch(hf_tok, EXAMPLES, args.model_max_length)

    for key in ["input_ids", "attention_mask"]:
        if not torch.equal(local_batch[key], hf_batch[key]):
            raise AssertionError(f"{key} mismatch between local and HF tokenizer")

    print("OK: HF tokenizer matches local tokenizer")
    print(f"model_max_length={hf_tok.model_max_length}")
    print(f"vocab_size={hf_tok.vocab_size}")
    print(f"input shape={tuple(hf_batch['input_ids'].shape)}")


if __name__ == "__main__":
    main()
