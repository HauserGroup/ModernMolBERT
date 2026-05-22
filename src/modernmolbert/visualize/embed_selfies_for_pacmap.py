# scripts/visualize_embeddings/embed_selfies_for_pacmap.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def embed_texts(
    *,
    texts: list[str],
    model_path: str | Path,
    tokenizer_path: str | Path | None,
    batch_size: int,
    max_length: int,
    pooling: str,
    device: str,
) -> np.ndarray:
    resolved_device = choose_device(device)

    tokenizer_source = tokenizer_path if tokenizer_path is not None else model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        subfolder="ape_tokenizer",
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.to(resolved_device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding SELFIES"):
            batch_texts = texts[start : start + batch_size]

            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(resolved_device) for key, value in batch.items()}

            output = model(**batch)

            if not hasattr(output, "last_hidden_state"):
                raise ValueError(
                    "Model output does not expose last_hidden_state. "
                    "Use an encoder-style checkpoint or adapt this script."
                )

            hidden = output.last_hidden_state

            if pooling == "mean":
                pooled = mean_pool(hidden, batch["attention_mask"])
            elif pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling: {pooling!r}")

            embeddings.append(pooled.detach().cpu().float().numpy())

    return np.concatenate(embeddings, axis=0)


def load_input_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input suffix: {path.suffix!r}")


def save_outputs(
    *,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: str | Path,
    metadata: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)

    df_out = df.copy()
    df_out["embedding_row"] = np.arange(len(df_out))
    df_out.to_parquet(output_dir / "metadata.parquet", index=False)

    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote embeddings: {output_dir / 'embeddings.npy'}")
    print(f"Wrote metadata:   {output_dir / 'metadata.parquet'}")
    print(f"Wrote config:     {output_dir / 'embedding_metadata.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a SELFIES column from a parquet/CSV file for PaCMAP visualization."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet/CSV produced by load_chembl_for_umap.py.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Hugging Face-compatible model checkpoint directory.",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional tokenizer path. Defaults to --model-path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where embeddings.npy and metadata.parquet are written.",
    )

    parser.add_argument(
        "--selfies-column",
        default="selfies",
        help="Column containing SELFIES strings.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or mps.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_input_frame(args.input)

    if args.selfies_column not in df.columns:
        raise ValueError(
            f"Missing SELFIES column {args.selfies_column!r}. Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.selfies_column]).reset_index(drop=True)
    texts = df[args.selfies_column].astype(str).tolist()

    if not texts:
        raise ValueError("No SELFIES strings left after dropping missing values.")

    embeddings = embed_texts(
        texts=texts,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        device=args.device,
    )

    metadata = {
        "input": str(args.input),
        "model_path": str(args.model_path),
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "selfies_column": args.selfies_column,
        "n_rows": int(len(df)),
        "embedding_shape": list(embeddings.shape),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pooling": args.pooling,
        "device": args.device,
    }

    save_outputs(
        df=df,
        embeddings=embeddings,
        output_dir=args.output_dir,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
