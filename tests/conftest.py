from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Paper-analysis scripts live in scripts/paper/ and are imported by name in tests.
for scripts_dir in (ROOT / "scripts", ROOT / "scripts" / "paper"):
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def find_existing_minimal_model() -> Path | None:
    """Return the first available tiny trained model directory, or None."""
    candidates = [
        ROOT / "runs" / "mps_base_minimal_pubchem10m" / "final_model",
        ROOT / "runs" / "mps_debug" / "final_model",
        ROOT / "runs" / "mps_base_smoke_512_symbol_tokenizer" / "final_model",
        ROOT / "runs" / "mps_base_smoke_256_symbol_tokenizer" / "final_model",
        ROOT / "runs" / "zinc20_debug" / "final_model",
    ]
    for path in candidates:
        if (
            path.exists()
            and (path / "config.json").exists()
            and ((path / "ape_tokenizer").exists() or (path / "vocab.json").exists())
            and (any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").exists())
        ):
            return path
    return None


@pytest.fixture
def existing_minimal_model() -> Path:
    """Pytest fixture that skips if no trained model is available."""
    model = find_existing_minimal_model()
    if model is None:
        pytest.skip(
            "No existing minimal trained model found. Run a debug/smoke training command first."
        )
    return model
