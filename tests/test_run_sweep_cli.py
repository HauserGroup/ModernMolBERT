# tests/test_run_sweep_cli.py
"""Dry-run coverage for the representation-aware sweep launcher.

Dry-run skips preflight and only touches stdlib, so the script runs with the bare
interpreter (no uv / heavy deps).
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sweeps" / "run_sweep.py"


def _dry_run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--model-size", "small", "--dry-run", *extra],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_selfies_dry_run_is_unchanged():
    res = _dry_run()
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "Total grid: 27" in out
    assert "chembl36_selfies_2m_ape_max2_min3000.json" in out
    assert "runs/chembl36_small_mask_mlm_lr_sweep/" in out
    assert "chembl36_smiles_" not in out


def test_smiles_dry_run_uses_smiles_tokenizer_and_column():
    res = _dry_run("--representation", "SMILES")
    assert res.returncode == 0, res.stderr
    out = res.stdout
    # 2 masking x 3 mlm x 3 lr.
    assert "Total grid: 18" in out
    assert "chembl36_smiles_2m_ape_max6_mf3000.json" in out
    assert "--molecule_column smiles_canonical_clean" in out
    assert "--representation SMILES" in out
    assert "runs/chembl36_smiles_small_mask_mlm_lr_sweep/" in out
    assert "hetero_span" not in out


def test_smiles_rejects_hetero_span():
    res = _dry_run("--representation", "SMILES", "--masking", "hetero_span")
    assert res.returncode != 0
    assert "not valid for representation SMILES" in (res.stderr + res.stdout)
