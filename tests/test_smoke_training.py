"""Optional end-to-end smoke tests for ModernMolBERT training.

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _smoke_enabled() -> bool:
    return os.environ.get("MODERNMOLBERT_RUN_SMOKE") == "1"


def _mps_enabled() -> bool:
    return os.environ.get("MODERNMOLBERT_RUN_MPS") == "1"


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
    assert (
        any(final_model.glob("*.safetensors"))
        or (final_model / "pytorch_model.bin").exists()
    )
    assert (final_model / "tokenizer.json").exists()

    reload_code = f"""
from transformers import AutoModelForMaskedLM
from modernmolbert.ape_tokenizer import APETokenizer
import torch
model = AutoModelForMaskedLM.from_pretrained({str(final_model)!r})
tok = APETokenizer()
tok.load_vocabulary({str(final_model / "tokenizer.json")!r})
batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")
with torch.no_grad():
    out = model(**batch)
assert torch.isfinite(out.logits).all()
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
