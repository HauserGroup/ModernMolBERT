import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


EXAMPLES = [
    "[C][C][=C][C][Branch1][=N][N][N][=C][C][=Branch1][C][=O][NH1][C][Ring1][#Branch1][=O]",
    "[O][=C][Branch1][N][C][=C][C][=C][Branch1][C][Cl][C][=C][Ring1][#Branch1]",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/modernmolbert_best_span"),
    )
    parser.add_argument(
        "--tokenizer",
        default="HauserGroup/ApeTokenizer-SELFIES",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    config = AutoConfig.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    print("Tokenizer:")
    print(f"  vocab_size:       {tok.vocab_size}")
    print(f"  model_max_length: {tok.model_max_length}")
    print(f"  pad_token_id:     {tok.pad_token_id}")
    print(f"  mask_token_id:    {tok.mask_token_id}")

    print("Model config:")
    print(f"  vocab_size:       {config.vocab_size}")
    print(f"  pad_token_id:     {getattr(config, 'pad_token_id', None)}")
    print(f"  bos_token_id:     {getattr(config, 'bos_token_id', None)}")
    print(f"  eos_token_id:     {getattr(config, 'eos_token_id', None)}")
    print(f"  hidden_size:      {getattr(config, 'hidden_size', None)}")

    if tok.vocab_size != config.vocab_size:
        raise AssertionError(
            f"vocab_size mismatch: tokenizer={tok.vocab_size}, model={config.vocab_size}"
        )

    if tok.pad_token_id != getattr(config, "pad_token_id", tok.pad_token_id):
        raise AssertionError(
            f"pad_token_id mismatch: tokenizer={tok.pad_token_id}, "
            f"model={getattr(config, 'pad_token_id', None)}"
        )

    batch = tok(
        EXAMPLES,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(args.device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    hidden = out.last_hidden_state

    if not torch.isfinite(hidden).all():
        raise AssertionError("Model output contains non-finite values")

    print("OK: tokenizer is compatible with model")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"hidden shape:    {tuple(hidden.shape)}")


if __name__ == "__main__":
    main()
