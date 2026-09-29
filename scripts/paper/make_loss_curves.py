#!/usr/bin/env python3
"""
make_loss_curves.py

Training/validation curves for the released small ModernMolBERT (standard
masking). Reads HuggingFace Trainer log_history from trainer_state.json.

NOTE: per-step logs are only retained for the small standard and span runs.
The base run's log_history was not kept locally (only final metrics survive in
results/sweep_results_base.csv); base curves cannot be reconstructed here.

For a newly trained model, pass its run directory with --run-dir. Runs that
did not log validation masked-token accuracy (the revised trainer logs only
losses) get the loss panel alone.

Output: Supplementary_2.pdf (Appendix A).
"""

import argparse
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Plot pretraining curves from trainer_state.json.")
parser.add_argument(
    "--run-dir",
    type=Path,
    default=ROOT / "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_standard",
    help="Run directory containing trainer_state.json.",
)
parser.add_argument("--figure-dir", type=Path, default=ROOT / "paper/figures")
ARGS = parser.parse_args()
FIGDIR = ARGS.figure_dir
FIGDIR.mkdir(parents=True, exist_ok=True)
RUN = ARGS.run_dir

ts = json.loads((RUN / "trainer_state.json").read_text())
lh = ts["log_history"]
train = [(e["step"], e["loss"]) for e in lh if "loss" in e and "eval_loss" not in e]
ev_loss = [(e["step"], e["eval_loss"]) for e in lh if "eval_loss" in e]
ev_acc = [(e["step"], e["eval_masked_accuracy"]) for e in lh if "eval_masked_accuracy" in e]

if ev_acc:
    fig, (axA, axL) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): validation masked-token accuracy
    axA.plot(*zip(*ev_acc, strict=False), "o-", color="#2a7", lw=1.6, ms=5)
    axA.set_xlabel("Training step")
    axA.set_ylabel("Validation masked-token accuracy")
    axA.set_title("(a) Validation masked accuracy", fontsize=10)
    axA.grid(ls=":", c="0.85")
else:
    fig, axL = plt.subplots(1, 1, figsize=(5.5, 4.2))

# Panel (b): training loss (dense) + validation loss (sparse)
axL.plot(*zip(*train, strict=False), "-", color="#888", lw=1.0, label="train loss")
axL.plot(*zip(*ev_loss, strict=False), "o-", color="#c33", lw=1.6, ms=5, label="val loss")
axL.set_xlabel("Training step")
axL.set_ylabel("MLM loss")
axL.set_title(
    "(b) Training and validation loss" if ev_acc else "Training and validation loss", fontsize=10
)
axL.legend(frameon=False, fontsize=9)
axL.grid(ls=":", c="0.85")

fig.tight_layout()
fig.savefig(FIGDIR / "Supplementary_2.pdf", bbox_inches="tight")
print("Wrote Supplementary_2.pdf to", FIGDIR)
