#!/usr/bin/env python3
"""Summarize and rank ModernMolBERT pretraining sweep runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank pretraining runs in a sweep directory by a selection metric."
    )
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--metric", default="eval_loss")
    parser.add_argument(
        "--lower_is_better",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument(
        "--output_report",
        type=Path,
        default=None,
        help="Write a Markdown summary report to this path.",
    )
    parser.add_argument("--copy_best_to", type=Path, default=None)
    parser.add_argument(
        "--require_complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only rank runs that reached max_steps.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_scalar_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return only scalar-valued entries (keys already carry their prefix)."""
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, int | float | str | bool) or value is None
    }


def best_eval_from_log_history(
    trainer_state: dict[str, Any],
    metric: str,
    lower_is_better: bool,
) -> tuple[float | None, int | None]:
    log_history = trainer_state.get("log_history", [])
    candidates = []

    for event in log_history:
        if metric in event:
            value = event[metric]
        elif f"eval_{metric}" in event:
            value = event[f"eval_{metric}"]
        else:
            continue

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        step = event.get("step")
        candidates.append((value, step))

    if not candidates:
        return None, None

    if lower_is_better:
        best_value, best_step = min(candidates, key=lambda x: x[0])
    else:
        best_value, best_step = max(candidates, key=lambda x: x[0])

    return best_value, best_step


def summarize_run(run_dir: Path, metric: str, lower_is_better: bool) -> dict[str, Any]:
    run_args = read_json(run_dir / "run_args.json")
    eval_results = read_json(run_dir / "eval_results.json")
    train_results = read_json(run_dir / "train_results.json")
    trainer_state = read_json(run_dir / "trainer_state.json")
    metadata = read_json(run_dir / "ape_tokenizer_metadata.json")

    final_model = run_dir / "final_model"
    has_final_model = (
        final_model.exists()
        and (final_model / "config.json").exists()
        and (any(final_model.glob("*.safetensors")) or (final_model / "pytorch_model.bin").exists())
    )

    max_steps = run_args.get("max_steps")
    global_step = trainer_state.get("global_step")
    completed = bool(
        has_final_model
        and max_steps is not None
        and global_step is not None
        and global_step >= max_steps
    )

    best_metric = trainer_state.get("best_metric")
    best_checkpoint = trainer_state.get("best_model_checkpoint")

    best_from_history, best_step_from_history = best_eval_from_log_history(
        trainer_state,
        metric=metric,
        lower_is_better=lower_is_better,
    )

    if best_metric is None:
        best_metric = best_from_history

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "status": (
            "complete" if completed else ("has_final_model" if has_final_model else "incomplete")
        ),
        "has_final_model": has_final_model,
        "completed_max_steps": completed,
        "max_steps": max_steps,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_step_from_history": best_step_from_history,
        "best_model_checkpoint": best_checkpoint,
        "final_model": str(final_model) if has_final_model else None,
        "model_size": run_args.get("model_size"),
        "mlm_probability": run_args.get("mlm_probability"),
        "masking_strategy": run_args.get("masking_strategy"),
        "learning_rate": run_args.get("learning_rate"),
        "warmup_steps": run_args.get("warmup_steps"),
        "weight_decay": run_args.get("weight_decay"),
        "max_seq_length": run_args.get("max_seq_length"),
        "per_device_train_batch_size": run_args.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": run_args.get("gradient_accumulation_steps"),
        "dataset_name": run_args.get("dataset_name"),
        "tokenizer_vocab_path": run_args.get("tokenizer_vocab_path"),
    }

    # Eval and train result keys already carry their own prefix (e.g. eval_loss, train_loss).
    row.update(flatten_scalar_dict(eval_results))
    row.update(flatten_scalar_dict(train_results))

    row["num_parameters"] = metadata.get("num_parameters", row.get("train_num_parameters"))
    row["metadata_best_checkpoint"] = (metadata.get("trainer_state_summary", {}) or {}).get(
        "best_model_checkpoint"
    )

    # Resolve selection metric: prefer direct key, then eval_ prefix, then log-history fallback.
    if metric in row:
        selection_metric = row[metric]
    elif f"eval_{metric}" in row:
        selection_metric = row[f"eval_{metric}"]
    else:
        selection_metric = best_metric

    try:
        row["selection_metric"] = (
            float(selection_metric) if selection_metric is not None else math.nan
        )
    except (TypeError, ValueError):
        row["selection_metric"] = math.nan

    return row


def discover_runs(run_root: Path) -> list[Path]:
    runs = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / "run_args.json").exists() or (child / "trainer_state.json").exists():
            runs.append(child)
    return runs


def copy_best_model(best_row: dict[str, Any], destination: Path) -> None:
    src = Path(best_row["final_model"])
    if not src.exists():
        raise FileNotFoundError(f"Best final_model does not exist: {src}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Remove it or choose a different path."
        )
    shutil.copytree(src, destination)


# ── report helpers ────────────────────────────────────────────────────────────


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.4g}"
    return "—" if v is None else str(v)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = list(df.itertuples(index=False, name=None))
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def write_report(df: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    best = df.iloc[0]
    lines: list[str] = []

    lines += [
        f"# Sweep report: {args.run_root.name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Root: `{args.run_root}`  ",
        f"Metric: `{args.metric}` ({'lower' if args.lower_is_better else 'higher'} is better)  ",
        f"Runs ranked: {len(df)}",
        "",
        "## Ranked runs",
        "",
    ]

    overview_cols = [
        "rank",
        "run_name",
        "status",
        "selection_metric",
        "learning_rate",
        "masking_strategy",
        "mlm_probability",
        "global_step",
        "max_steps",
    ]
    lines += [_markdown_table(df[[c for c in overview_cols if c in df.columns]]), ""]

    lines += [f"## Best run: `{best['run_name']}`", ""]

    hp_cols = [
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "masking_strategy",
        "mlm_probability",
        "max_seq_length",
        "model_size",
        "max_steps",
        "global_step",
    ]
    hp_items = [(c, best[c]) for c in hp_cols if c in best.index and pd.notna(best[c])]
    if hp_items:
        lines += ["### Hyperparameters", ""]
        lines += [f"- **{k}:** {_fmt(v)}" for k, v in hp_items]
        lines.append("")

    eval_items = [
        (c, best[c])
        for c in df.columns
        if c.startswith("eval_") and c in best.index and pd.notna(best[c])
    ]
    if eval_items:
        lines += ["### Evaluation metrics", ""]
        lines += [f"- **{k}:** {_fmt(v)}" for k, v in eval_items]
        lines.append("")

    if best.get("final_model"):
        lines += [f"**Final model:** `{best['final_model']}`", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    run_dirs = discover_runs(args.run_root)
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.run_root}")

    rows = [
        summarize_run(run_dir, metric=args.metric, lower_is_better=args.lower_is_better)
        for run_dir in run_dirs
    ]

    df = pd.DataFrame(rows)

    if args.require_complete:
        df = df[df["completed_max_steps"]]

    df = df[df["selection_metric"].notna()].copy()
    if df.empty:
        raise SystemExit("No runs had a usable selection metric.")

    df = df.sort_values("selection_metric", ascending=args.lower_is_better).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    display_cols = [
        "rank",
        "run_name",
        "status",
        "selection_metric",
        "best_metric",
        "model_size",
        "mlm_probability",
        "masking_strategy",
        "learning_rate",
        "max_steps",
        "global_step",
        "eval_loss",
        "eval_perplexity",
        "eval_masked_accuracy",
        "train_loss",
        "train_runtime",
        "best_model_checkpoint",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nwrote CSV: {args.output_csv}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote JSON: {args.output_json}")

    if args.output_report:
        write_report(df, args, args.output_report)
        print(f"wrote report: {args.output_report}")

    if args.copy_best_to:
        best: dict[str, Any] = {str(k): v for k, v in df.iloc[0].to_dict().items()}
        copy_best_model(best, args.copy_best_to)
        print(f"copied best final_model to: {args.copy_best_to}")


if __name__ == "__main__":
    main()
