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

# Columns pulled from all_results.json (in order). These legacy names are kept
# for backwards compatibility with existing notebooks. The final_eval_* columns
# below make the source explicit for new analysis.
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

FINAL_EVAL_KEYS = [
    "eval_loss",
    "eval_masked_accuracy",
    "eval_perplexity",
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _eval_history(trainer_state: dict) -> list[dict]:
    history = trainer_state.get("log_history", [])
    if not isinstance(history, list):
        return []
    return [
        entry
        for entry in history
        if isinstance(entry, dict) and _numeric(entry.get("eval_loss")) is not None
    ]


def _best_logged_eval(history: list[dict]) -> dict:
    if not history:
        return {}
    return min(history, key=lambda entry: float(entry["eval_loss"]))


def _last_logged_eval(history: list[dict]) -> dict:
    return history[-1] if history else {}


def _checkpoint_step(checkpoint: object) -> int | str:
    if not isinstance(checkpoint, str) or "-" not in checkpoint:
        return ""
    raw_step = checkpoint.rsplit("-", 1)[-1]
    try:
        return int(raw_step)
    except ValueError:
        return raw_step


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

        results = _load_json(results_path)
        args = _load_json(args_path)
        trainer_state = _load_json(run_dir / "trainer_state.json")
        eval_history = _eval_history(trainer_state)
        best_logged = _best_logged_eval(eval_history)
        last_logged = _last_logged_eval(eval_history)
        best_checkpoint = trainer_state.get("best_model_checkpoint", "")
        load_best_model_at_end = args.get("load_best_model_at_end", "")
        metric_source = (
            "best_model_final_eval"
            if load_best_model_at_end is not False and best_checkpoint
            else "final_model_eval"
        )

        row = {
            "run": run_dir.name,
            "size": args.get("model_size", ""),
            "strategy": parsed["strategy"],
            "mlm_prob": parsed["mlm_prob"],
            "learning_rate": args.get("learning_rate", parsed["lr_from_name"]),
            "eval_masking_strategy": args.get("masking_strategy", parsed["strategy"]),
            "eval_mlm_probability": args.get("mlm_probability", parsed["mlm_prob"]),
            "load_best_model_at_end": load_best_model_at_end,
            "metric_source": metric_source,
            "metric_note": (
                "Final eval after training; if load_best_model_at_end was enabled, "
                "the best checkpoint was loaded first. Validation masking uses this "
                "run-specific masking_strategy and mlm_probability."
            ),
            "best_checkpoint": best_checkpoint,
            "best_step": trainer_state.get("best_global_step", "")
            or _checkpoint_step(best_checkpoint),
            "best_logged_eval_loss": best_logged.get("eval_loss", ""),
            "best_logged_eval_masked_accuracy": best_logged.get("eval_masked_accuracy", ""),
            "last_logged_eval_step": last_logged.get("step", ""),
            "last_logged_eval_loss": last_logged.get("eval_loss", ""),
            "last_logged_eval_masked_accuracy": last_logged.get("eval_masked_accuracy", ""),
        }
        for key in FINAL_EVAL_KEYS:
            row[f"final_{key}"] = results.get(key, "")
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
