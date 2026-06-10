"""Optional end-to-end smoke tests for ModernMolBERT training and encoding.

Skipped by default. Enable with:
    MODERNMOLBERT_RUN_SMOKE=1 uv run pytest -m smoke

The MPS test additionally requires:
    MODERNMOLBERT_RUN_MPS=1

Example:
    MODERNMOLBERT_RUN_SMOKE=1 MODERNMOLBERT_RUN_MPS=1 \\
      uv run pytest -m "smoke and mps" -s
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import ROOT, find_existing_minimal_model


def _repo_root() -> Path:
    return ROOT


def _smoke_enabled() -> bool:
    return os.environ.get("MODERNMOLBERT_RUN_SMOKE") == "1"


def _mps_enabled() -> bool:
    return os.environ.get("MODERNMOLBERT_RUN_MPS") == "1"


def _find_existing_tokenizer_vocab() -> Path | None:
    """Find an existing tokenizer vocab for local encode checks."""
    candidates = [
        _repo_root() / "tokenizer" / "selfies_symbol_tokenizer.json",
        _repo_root() / "tokenizer" / "selfies_ape_tokenizer.json",
        _repo_root() / "tokenizer" / "selfies_ape_tokenizer_1m.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def test_local_tokenizer_encode_selfies_examples() -> None:
    """Fast encode-only check (no model forward)."""
    tokenizer_path = _find_existing_tokenizer_vocab()
    if tokenizer_path is None:
        pytest.skip("No local tokenizer vocabulary found under tokenizer/.")

    from modernmolbert.tokenization_ape import APEPreTrainedTokenizer

    tok = APEPreTrainedTokenizer()
    tok.load_vocabulary_file(tokenizer_path)

    examples = [
        "[C]",
        "[O]",
        "[C][C][O]",
        "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    ]

    unk_id = tok.vocabulary[str(tok.unk_token)]
    bos_id = tok.vocabulary[str(tok.bos_token)]
    eos_id = tok.vocabulary[str(tok.eos_token)]
    verbose = os.environ.get("MODERNMOLBERT_TEST_VERBOSE") == "1"
    summaries: list[str] = []

    if verbose:
        print(f"[encode-test] tokenizer={tokenizer_path}")
        print(f"[encode-test] special_ids bos={bos_id} eos={eos_id} unk={unk_id}")

    for text in examples:
        encoded = tok.encode(text, add_special_tokens=True)
        token_strings = tok.convert_ids_to_tokens(encoded)
        unk_positions = [
            i for i, token_id in enumerate(encoded[1:-1], start=1) if token_id == unk_id
        ]

        if verbose:
            summaries.append(
                " | ".join(
                    [
                        f"text={text}",
                        f"len={len(encoded)}",
                        f"ids={encoded}",
                        f"tokens={token_strings}",
                        f"unk_positions={unk_positions}",
                    ]
                )
            )

        assert encoded[0] == bos_id, (text, encoded, token_strings)
        assert encoded[-1] == eos_id, (text, encoded, token_strings)
        assert len(encoded) > 2, (text, encoded, token_strings)
        assert not unk_positions, (text, encoded, token_strings, unk_positions)

        # __call__ should preserve the same encoded ids.
        batch = tok(text, add_special_tokens=True, return_tensors="pt")
        ids = batch["input_ids"].tolist()
        assert ids == encoded, (text, ids, encoded, token_strings)

    if verbose:
        for summary in summaries:
            print(f"[encode-test] {summary}")


@pytest.mark.smoke
def test_existing_minimal_model_selfies_encoding() -> None:
    """Verify that a minimal trained model can encode SELFIES and produce finite logits.

    This does not train a model. It uses the first available minimal/debug model in
    runs/*/final_model. If no model exists, it skips.
    """
    if not _smoke_enabled():
        pytest.skip("Set MODERNMOLBERT_RUN_SMOKE=1 to run smoke tests.")

    final_model = find_existing_minimal_model()
    if final_model is None:
        pytest.skip(
            "No existing minimal trained model found. Run the MPS smoke training test "
            "or a debug training command first."
        )

    code = f"""
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
import torch

model_dir = {str(final_model)!r}
tokenizer_path = {str(final_model / "vocab.json")!r}

model = AutoModelForMaskedLM.from_pretrained(model_dir)
model.eval()

tokenizer_dir = Path(model_dir) / "ape_tokenizer"
if tokenizer_dir.exists():
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
else:
    tok = APEPreTrainedTokenizer()
    tok.load_vocabulary_file(tokenizer_path)

examples = [
    "[C]",
    "[O]",
    "[C][C][O]",
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
]

special = {{
    "unk_token": tok.vocabulary[tok.unk_token],
    "bos_token": tok.vocabulary[tok.bos_token],
    "eos_token": tok.vocabulary[tok.eos_token],
    "pad_token": tok.vocabulary[tok.pad_token],
    "mask_token": tok.vocabulary[tok.mask_token],
}}

for text in examples:
    batch = tok(text, add_special_tokens=True, return_tensors="pt")

    for key in ["input_ids", "attention_mask"]:
        if batch[key].ndim == 1:
            batch[key] = batch[key].unsqueeze(0)

    unk_positions = [
        i for i, token_id in enumerate(batch["input_ids"][0].tolist())
        if token_id == special["unk_token"]
    ]

    assert not unk_positions, (text, batch["input_ids"][0].tolist(), unk_positions)

    with torch.no_grad():
        out = model(**batch)

    assert torch.isfinite(out.logits).all(), text
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == batch["input_ids"].shape[1]
    assert out.logits.shape[2] == model.config.vocab_size

print("encoding ok", model_dir)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_repo_root(),
        env={**os.environ.copy(), "TOKENIZERS_PARALLELISM": "false"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "encoding ok" in result.stdout


@pytest.mark.smoke
@pytest.mark.mps
def test_mps_base_smoke_training(tmp_path: Path) -> None:
    """Tiny MPS smoke training job: exercises the full pipeline and verifies reload."""
    if not _smoke_enabled():
        pytest.skip("Set MODERNMOLBERT_RUN_SMOKE=1 to run smoke tests.")
    if not _mps_enabled():
        pytest.skip("Set MODERNMOLBERT_RUN_MPS=1 to run MPS smoke tests.")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this machine.")

    output_dir = tmp_path / "mps_base_smoke"
    cmd = [
        sys.executable,
        "-m",
        "modernmolbert.train_selfies_ape_modernbert",
        "--output_dir",
        str(output_dir),
        "--device_backend",
        "mps",
        "--model_size",
        "base",
        "--max_seq_length",
        "128",
        "--max_steps",
        "5",
        "--eval_size",
        "4",
        "--max_eval_batches",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "2",
        "--mlm_probability",
        "0.30",
        "--learning_rate",
        "1e-4",
        "--logging_steps",
        "1",
        "--eval_steps",
        "5",
        "--save_steps",
        "5",
        "--save_total_limit",
        "1",
        "--num_workers",
        "0",
        "--report_to",
        "none",
    ]

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    final_model = output_dir / "final_model"
    assert final_model.exists()
    assert (final_model / "config.json").exists()
    assert any(final_model.glob("*.safetensors")) or (final_model / "pytorch_model.bin").exists()
    assert (final_model / "vocab.json").exists()

    reload_code = f"""
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained({str(final_model)!r})
model.eval()

tok = AutoTokenizer.from_pretrained(
    {str(final_model / "ape_tokenizer")!r},
    trust_remote_code=True,
)

batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")

for key in ["input_ids", "attention_mask"]:
    if batch[key].ndim == 1:
        batch[key] = batch[key].unsqueeze(0)

with torch.no_grad():
    out = model(**batch)

assert torch.isfinite(out.logits).all()
assert out.logits.shape[0] == 1
assert out.logits.shape[1] == batch["input_ids"].shape[1]
assert out.logits.shape[2] == model.config.vocab_size

print("reload ok", tuple(out.logits.shape))
"""
    reload_result = subprocess.run(
        [sys.executable, "-c", reload_code],
        cwd=_repo_root(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    assert reload_result.returncode == 0, reload_result.stdout
    assert "reload ok" in reload_result.stdout
