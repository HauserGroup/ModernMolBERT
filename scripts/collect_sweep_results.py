"""Collect MLM sweep results into a CSV.

Usage:
    python scripts/collect_sweep_results.py
    python scripts/collect_sweep_results.py --sweep runs/other_sweep --out results/other.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SWEEP_DIR = Path("runs/chembl36_small_mask_mlm_lr_sweep")
OUT_FILE = SWEEP_DIR / "sweep_results.csv"

# Columns pulled from all_results.json (in order)
METRIC_KEYS = [
    "eval_loss",
    "eval_masked_accuracy",
    "eval_perplexity",
    "train_loss",
    "epoch",
    "num_parameters",
    "train_runtime",
    "eval_runtime",
]

# Regex to parse dir names like: mask_hetero_span__mlm_0p15__lr_1e-4
_DIR_RE = re.compile(r"mask_(?P<strategy>.+?)__mlm_(?P<mlm_prob>[0-9p]+)__lr_(?P<lr>[0-9e\-+.]+)$")


def _parse_mlm_prob(raw: str) -> float:
    return float(raw.replace("p", "."))


def _parse_dir_name(name: str) -> dict | None:
    m = _DIR_RE.match(name)
    if not m:
        return None
    return {
        "strategy": m.group("strategy"),
        "mlm_prob": _parse_mlm_prob(m.group("mlm_prob")),
        "lr_from_name": m.group("lr"),
    }


def collect(sweep_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _parse_dir_name(run_dir.name)
        if parsed is None:
            continue

        results_path = run_dir / "all_results.json"
        args_path = run_dir / "run_args.json"
        if not results_path.exists():
            print(f"  skip {run_dir.name}: no all_results.json", file=sys.stderr)
            continue

        results = json.loads(results_path.read_text())
        args = json.loads(args_path.read_text()) if args_path.exists() else {}

        row = {
            "run": run_dir.name,
            "size": args.get("model_size", ""),
            "strategy": parsed["strategy"],
            "mlm_prob": parsed["mlm_prob"],
            "learning_rate": args.get("learning_rate", parsed["lr_from_name"]),
        }
        for key in METRIC_KEYS:
            row[key] = results.get(key, "")

        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default=str(SWEEP_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep)
    out_file = Path(args.out) if args.out else sweep_dir / "sweep_results.csv"

    rows = collect(sweep_dir)
    if not rows:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()
