from pathlib import Path

import numpy as np
import selfies as sf
import torch
from tqdm.auto import tqdm
from transformers import AutoModel

from modernmolbert.ape_tokenizer import APETokenizer


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def _batch_tokenize(
    tokenizer: APETokenizer,
    selfies_list: list[str],
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    encoded = [
        tokenizer(s, add_special_tokens=True, max_length=max_seq_length)
        for s in selfies_list
    ]

    max_len = max(len(x["input_ids"]) for x in encoded)

    input_ids = []
    attention_mask = []

    for item in encoded:
        ids = item["input_ids"][:max_seq_length]
        mask = item["attention_mask"][:max_seq_length]

        pad_len = max_len - len(ids)
        ids = ids + [tokenizer.pad_token_id] * pad_len
        mask = mask + [0] * pad_len

        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom


def embed_smiles(
    smiles: list[str],
    model_dir: str | Path,
    tokenizer_path: str | Path,
    max_seq_length: int = 256,
    batch_size: int = 64,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed SMILES via SMILES->SELFIES->ModernMolBERT mean-pooled encoder.

    Returns:
        embeddings: [n_valid, hidden_size]
        valid_mask: boolean mask over input smiles indicating successfully embedded rows.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    tokenizer = APETokenizer()
    tokenizer.load_vocabulary(str(tokenizer_path))

    model = AutoModel.from_pretrained(str(model_dir))
    model.to(device_obj)
    model.eval()

    selfies_list: list[str] = []
    valid_indices: list[int] = []

    for i, smi in enumerate(smiles):
        s = smiles_to_selfies(smi)
        if s is None:
            continue
        selfies_list.append(s)
        valid_indices.append(i)

    embeddings = []

    for start in tqdm(
        range(0, len(selfies_list), batch_size), desc="Embedding molecules"
    ):
        batch_selfies = selfies_list[start : start + batch_size]
        batch = _batch_tokenize(tokenizer, batch_selfies, max_seq_length)
        batch = {k: v.to(device_obj) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch)
            pooled = mean_pool(out.last_hidden_state, batch["attention_mask"])

        embeddings.append(pooled.cpu().numpy())

    if embeddings:
        X = np.concatenate(embeddings, axis=0)
    else:
        X = np.empty((0, model.config.hidden_size), dtype=np.float32)

    valid_mask = np.zeros(len(smiles), dtype=bool)
    valid_mask[valid_indices] = True

    return X, valid_mask
