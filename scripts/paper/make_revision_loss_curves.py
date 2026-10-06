#!/usr/bin/env python3
"""Plot within-configuration MLM loss histories for the 25 revision runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = (
    "small_ape_selfies",
    "small_ape_smiles",
    "small_bpe_selfies",
    "small_bpe_smiles",
    "base_ape_selfies",
)
SEEDS = range(42, 47)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    figure, axes = plt.subplots(len(MODELS), 1, figsize=(8, 12), sharex=True)
    colours = plt.cm.viridis([0.1, 0.3, 0.5, 0.7, 0.9])
    for axis, model in zip(axes, MODELS, strict=True):
        for seed, colour in zip(SEEDS, colours, strict=True):
            source = args.run_root / model / f"seed{seed}" / "trainer_state.json"
            data = json.loads(source.read_text())
            if data["global_step"] != 30000:
                raise ValueError(f"Incomplete trainer history: {source}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            train, validation = [], []
            for index, event in enumerate(data["log_history"]):
                for key, series in (("loss", "train_loss"), ("eval_loss", "validation_loss")):
                    if key not in event:
                        continue
                    point = (int(event["step"]), float(event[key]))
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "history_sha256": digest,
                            "log_history_index": index,
                            "step": point[0],
                            "series": series,
                            "value": point[1],
                        }
                    )
                    (train if series == "train_loss" else validation).append(point)
            if len(train) != 300 or len(validation) != 7:
                raise ValueError(f"Unexpected loss record count: {source}")
            axis.plot(*zip(*train, strict=True), color=colour, alpha=0.35, linewidth=0.7)
            axis.plot(
                *zip(*validation, strict=True),
                color=colour,
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=f"seed {seed}",
            )
        axis.set_title(model.replace("_", " "))
        axis.set_ylabel("MLM loss")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Optimizer step")
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    plt.close(figure)

    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    with args.source_data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
