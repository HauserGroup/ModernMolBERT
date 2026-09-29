#!/usr/bin/env python3
"""Time one feature-extraction configuration for ``measure_embedding_cost.py``.

Runs in its own process and imports only what the measured pipeline needs, so
that its peak resident memory belongs to that pipeline. Prints one JSON object
on the last line of standard output.

For a ModernMolBERT checkpoint the timed path is one call of the benchmark
featurizer, ``ModernMolBERTSelfiesFeaturizer.featurize_smiles``. A full
untimed pass comes first, so that per-shape kernel compilation on
accelerators is excluded; its time is kept as ``first_pass_seconds``. A final
pass splits the same work into SMILES-to-SELFIES conversion, tokenisation
(including the lossless-tokenisation checks), and the encoder forward pass
with mean pooling. For a Hugging Face encoder the path is optional
SMILES-to-SELFIES conversion, the model's own tokeniser, the forward pass and
attention-mask mean pooling. For ECFP4 the path is RDKit SMILES parsing plus a
Morgan fingerprint, binary or count.

Usage (normally called by measure_embedding_cost.py):
    uv run python scripts/paper/embedding_cost_worker.py SPEC_JSON MOLECULES_TXT
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ECFP_RADIUS = 2
ECFP_BITS = 2048


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure_checkpoint(spec: dict, smiles: list[str]) -> dict[str, object]:
    start = time.perf_counter()
    import selfies as sf
    import torch

    from modernmolbert.eval.featurizers.modernmolbert_selfies import (
        ModernMolBERTSelfiesFeaturizer,
    )
    from modernmolbert.eval.pooling import mean_pool_excluding_token_ids

    import_seconds = time.perf_counter() - start
    device = spec["device"]
    batch_size = spec["batch_size"]

    def synchronize() -> None:
        if device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

    start = time.perf_counter()
    featurizer = ModernMolBERTSelfiesFeaturizer(
        model_dir=spec["path"],
        max_seq_length=spec["max_seq_length"],
        pooling="mean",
        device=device,
        batch_size=batch_size,
    )
    synchronize()
    load_seconds = time.perf_counter() - start
    rss_after_load = rss_bytes()

    start = time.perf_counter()
    featurizer.featurize_smiles(smiles)
    synchronize()
    first_pass = time.perf_counter() - start

    start = time.perf_counter()
    features = featurizer.featurize_smiles(smiles)
    synchronize()
    end_to_end = time.perf_counter() - start
    peak_rss = peak_rss_bytes()

    # The stage split mirrors featurize_smiles; the reported cost is the call above.
    start = time.perf_counter()
    encoded: list[str | None] = []
    for text in smiles:
        try:
            encoded.append(sf.encoder(text.strip()) or None)
        except Exception:
            encoded.append(None)
    conversion = time.perf_counter() - start

    tokenizer = featurizer.tokenizer
    special = featurizer._special_token_ids()
    start = time.perf_counter()
    accepted = []
    for text in encoded:
        if text is None or "".join(tokenizer.tokenize(text)) != text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if ids and special.isdisjoint(ids):
            accepted.append(text)
    batches = [
        featurizer._tokenize_selfies_batch(accepted[i : i + batch_size])
        for i in range(0, len(accepted), batch_size)
    ]
    tokenisation = time.perf_counter() - start

    peak_mps = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            inputs = {key: value.to(featurizer._device) for key, value in batch.items()}
            hidden = featurizer.model(**inputs).last_hidden_state
            pooled = mean_pool_excluding_token_ids(
                last_hidden_state=hidden,
                attention_mask=inputs["attention_mask"],
                input_ids=inputs["input_ids"],
                excluded_token_ids=special,
            )
            pooled.cpu().numpy()
            if device == "mps":
                peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())
    synchronize()
    features_seconds = time.perf_counter() - start

    config = featurizer.model.config
    return {
        "n_molecules": len(smiles),
        "n_valid": int(features.valid_mask.sum()),
        "n_valid_stage_split": len(accepted),
        "n_truncated": int(features.metadata["n_truncated"]),
        "import_seconds": import_seconds,
        "load_seconds": load_seconds,
        "first_pass_seconds": first_pass,
        "end_to_end_seconds": end_to_end,
        "conversion_seconds": conversion,
        "tokenisation_seconds": tokenisation,
        "features_seconds": features_seconds,
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": peak_rss,
        "peak_mps_driver_bytes": peak_mps if device == "mps" else None,
        "n_parameters": sum(p.numel() for p in featurizer.model.parameters()),
        "feature_dim": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "torch_threads": torch.get_num_threads(),
    }


def measure_transformer(spec: dict, smiles: list[str]) -> dict[str, object]:
    """Time a Hugging Face encoder on SMILES, or on SELFIES converted from them."""
    start = time.perf_counter()
    import selfies as sf
    import torch
    from transformers import AutoModel, AutoTokenizer

    import_seconds = time.perf_counter() - start
    device = spec["device"]
    batch_size = spec["batch_size"]
    trust = bool(spec["trust_remote_code"])

    def synchronize() -> None:
        if device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=trust)
    model = AutoModel.from_pretrained(spec["path"], trust_remote_code=trust).to(device).eval()
    synchronize()
    load_seconds = time.perf_counter() - start
    rss_after_load = rss_bytes()
    max_length = min(int(tokenizer.model_max_length), 512)

    def convert(batch: list[str]) -> list[str]:
        if spec["input"] == "smiles":
            return batch
        converted = []
        for text in batch:
            try:
                encoded = sf.encoder(text.strip())
            except Exception:
                continue
            if encoded:
                converted.append(encoded)
        return converted

    def tokenize(texts: list[str]) -> list:
        return [
            tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            for i in range(0, len(texts), batch_size)
        ]

    def forward(batches: list) -> tuple[np.ndarray, int]:
        rows = []
        peak = 0
        with torch.no_grad():
            for batch in batches:
                inputs = {key: value.to(device) for key, value in batch.items()}
                hidden = model(**inputs).last_hidden_state
                # Attention-mask mean; the pooling choice barely changes the cost.
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                rows.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
                if device == "mps":
                    peak = max(peak, torch.mps.driver_allocated_memory())
        synchronize()
        return np.concatenate(rows), peak

    def featurize(batch: list[str]) -> np.ndarray:
        return forward(tokenize(convert(batch)))[0]

    start = time.perf_counter()
    featurize(smiles)
    first_pass = time.perf_counter() - start

    start = time.perf_counter()
    features = featurize(smiles)
    end_to_end = time.perf_counter() - start
    peak_rss = peak_rss_bytes()

    start = time.perf_counter()
    texts = convert(smiles)
    conversion = time.perf_counter() - start
    start = time.perf_counter()
    batches = tokenize(texts)
    tokenisation = time.perf_counter() - start
    start = time.perf_counter()
    _, peak_mps = forward(batches)
    features_seconds = time.perf_counter() - start

    config = model.config
    return {
        "n_molecules": len(smiles),
        "n_valid": int(features.shape[0]),
        "n_valid_stage_split": len(texts),
        # Sequences that reached the length cap.
        "n_truncated": sum(
            int(batch["attention_mask"].sum(1).eq(max_length).sum()) for batch in batches
        ),
        "import_seconds": import_seconds,
        "load_seconds": load_seconds,
        "first_pass_seconds": first_pass,
        "end_to_end_seconds": end_to_end,
        "conversion_seconds": conversion,
        "tokenisation_seconds": tokenisation,
        "features_seconds": features_seconds,
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": peak_rss,
        "peak_mps_driver_bytes": peak_mps if device == "mps" else None,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "feature_dim": int(features.shape[1]),
        "vocab_size": getattr(config, "vocab_size", None),
        "torch_threads": torch.get_num_threads(),
        "model_revision": getattr(config, "_commit_hash", None),
    }


def measure_fingerprint(spec: dict, smiles: list[str]) -> dict[str, object]:
    start = time.perf_counter()
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    import_seconds = time.perf_counter() - start
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    start = time.perf_counter()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
    fingerprint = (
        generator.GetCountFingerprintAsNumPy
        if spec["name"] == "ECFP4-count"
        else generator.GetFingerprintAsNumPy
    )
    load_seconds = time.perf_counter() - start
    rss_after_load = rss_bytes()

    def featurize(batch: list[str]) -> np.ndarray:
        mols = [Chem.MolFromSmiles(text) for text in batch]
        return np.stack([fingerprint(mol) for mol in mols if mol is not None]).astype(np.float32)

    start = time.perf_counter()
    featurize(smiles)
    first_pass = time.perf_counter() - start

    start = time.perf_counter()
    features = featurize(smiles)
    end_to_end = time.perf_counter() - start
    peak_rss = peak_rss_bytes()

    start = time.perf_counter()
    mols = [Chem.MolFromSmiles(text) for text in smiles]
    conversion = time.perf_counter() - start
    start = time.perf_counter()
    np.stack([fingerprint(mol) for mol in mols if mol is not None]).astype(np.float32)
    features_seconds = time.perf_counter() - start

    return {
        "n_molecules": len(smiles),
        "n_valid": int(features.shape[0]),
        "n_valid_stage_split": sum(mol is not None for mol in mols),
        "n_truncated": 0,
        "import_seconds": import_seconds,
        "load_seconds": load_seconds,
        "first_pass_seconds": first_pass,
        "end_to_end_seconds": end_to_end,
        "conversion_seconds": conversion,
        "tokenisation_seconds": None,
        "features_seconds": features_seconds,
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": peak_rss,
        "peak_mps_driver_bytes": None,
        "n_parameters": 0,
        "feature_dim": ECFP_BITS,
        "vocab_size": None,
        "torch_threads": None,
    }


def main(argv: list[str] | None = None) -> None:
    rss_start = rss_bytes()
    spec_json, molecules_file = sys.argv[1:] if argv is None else argv
    spec = json.loads(spec_json)
    smiles = Path(molecules_file).read_text().splitlines()
    measure = {
        "checkpoint": measure_checkpoint,
        "transformer": measure_transformer,
        "fingerprint": measure_fingerprint,
    }[spec["kind"]]
    print(json.dumps({"rss_start_bytes": rss_start, **measure(spec, smiles)}))


if __name__ == "__main__":
    main()
