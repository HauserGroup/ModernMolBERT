import pandas as pd

from make_cost_table import render

MANIFEST = {
    "hardware": {"processor": "Test CPU", "physical_cores": 4, "memory_bytes": 16 * 2**30},
    "software": {"torch": "2.0"},
    "molecules": {"n_sampled": 1000},
    "repeats": 3,
    "max_seq_length": 128,
    "ecfp": {"radius": 2, "bits": 2048},
    "checkpoints": {"ModernMolBERT-small": {"weights_bytes": 136_603_308}},
}


def summary_row(name: str, kind: str, device: str, batch_size: float, rate: float) -> dict:
    return {
        "name": name,
        "kind": kind,
        "device": device,
        "batch_size": batch_size,
        "median_molecules_per_second": rate,
        "max_peak_rss_bytes": 780_000_000 if kind == "checkpoint" else 74_000_000,
        "n_parameters": 33_886_208 if kind == "checkpoint" else 0,
        "n_valid": 970 if kind == "checkpoint" else 1000,
        "feature_dim": 512 if kind == "checkpoint" else 2048,
    }


def test_table_uses_cpu_batch_32_and_both_mps_batches() -> None:
    summary = pd.DataFrame(
        [
            summary_row("ECFP4-binary", "fingerprint", "cpu", float("nan"), 10_489.8),
            summary_row("ModernMolBERT-small", "checkpoint", "cpu", 32, 339.9),
            summary_row("ModernMolBERT-small", "checkpoint", "cpu", 128, 328.3),
            summary_row("ModernMolBERT-small", "checkpoint", "mps", 32, 691.4),
            summary_row("ModernMolBERT-small", "checkpoint", "mps", 128, 752.1),
            summary_row("debug", "checkpoint", "cpu", 32, 1.0),
        ]
    )
    measurements = pd.DataFrame(
        {"torch_threads": [None, 8.0, 8.0], "load_average_before": [5.2, 9.9, 15.9]}
    )
    table = render(summary, MANIFEST, measurements, exclude=["debug"])

    assert r"ECFP4, binary & --- & --- & 2,048 & 10,490 & --- & --- & 0.07 \\" in table
    assert r"\model{}-small & 33.9 & 136.6 & 512 & 340 & 691 & 752 & 0.78 \\" in table
    assert "debug" not in table
    assert "Test CPU, 4 CPU cores, 16\\,GB" in table
    assert "median of 3 interleaved repeats" in table
    assert "capped at 128 tokens" in table and "8 PyTorch threads" in table
    assert r"\model{} embedded 970 of the molecules" in table
    assert "load average 5--16" in table
    assert r"\label{tab:extraction-cost}" in table
