#!/usr/bin/env python3
"""Upload every completed ModernMolBERT model in a sweep to Hugging Face Hub."""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi

from modernmolbert.upload_model import resolve_source_dir, upload_model_to_hub
from modernmolbert.utils import repo_root


@dataclass(frozen=True)
class UploadPlan:
    run_dir: Path
    source_dir: Path
    repo_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload all completed ModernMolBERT final_model directories in a sweep."
    )
    parser.add_argument(
        "--run_root",
        type=Path,
        required=True,
        help="Sweep directory containing one subdirectory per training run.",
    )
    parser.add_argument(
        "--repo_prefix",
        required=True,
        help=(
            "Hub repo prefix, usually namespace/base-name. "
            "Each run slug is appended, e.g. org/model-prefix-mask-standard-mlm-0p15-lr-1e-4."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="'final' (default), 'best', or a step number such as '25000'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create Hugging Face model repos as private.",
    )
    parser.add_argument(
        "--commit_message",
        default="Upload ModernMolBERT sweep checkpoint",
        help="Commit message for each Hub upload.",
    )
    parser.add_argument(
        "--hf_login",
        action="store_true",
        help="Call huggingface_hub.login() before uploading (reads HF_TOKEN from env / .env).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the upload plan and stage each model without touching the Hub.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path recording planned and completed uploads.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any discovered run is missing the requested checkpoint.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue uploading later runs if one upload fails.",
    )
    return parser.parse_args()


def slugify_run_name(name: str) -> str:
    slug = name.lower().replace("__", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9.-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-.")
    if not slug:
        raise ValueError(f"Cannot derive a Hub repo slug from run name: {name!r}")
    return slug


def repo_id_for_run(repo_prefix: str, run_name: str) -> str:
    prefix = repo_prefix.rstrip("-")
    return f"{prefix}-{slugify_run_name(run_name)}"


def discover_run_dirs(run_root: Path) -> list[Path]:
    runs = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (
            (child / "run_args.json").exists()
            or (child / "trainer_state.json").exists()
            or (child / "final_model").exists()
        ):
            runs.append(child)
    return runs


def build_upload_plans(
    run_root: Path,
    repo_prefix: str,
    checkpoint: str = "final",
    strict: bool = False,
) -> tuple[list[UploadPlan], list[dict[str, str]]]:
    plans: list[UploadPlan] = []
    skipped: list[dict[str, str]] = []

    for run_dir in discover_run_dirs(run_root):
        try:
            source_dir = resolve_source_dir(run_dir, checkpoint)
        except (FileNotFoundError, ValueError) as exc:
            message = str(exc)
            if strict:
                raise
            skipped.append({"run_dir": str(run_dir), "reason": message})
            continue

        plans.append(
            UploadPlan(
                run_dir=run_dir,
                source_dir=source_dir,
                repo_id=repo_id_for_run(repo_prefix, run_dir.name),
            )
        )

    return plans, skipped


def write_manifest(
    path: Path,
    run_root: Path,
    checkpoint: str,
    dry_run: bool,
    results: list[dict[str, Any]],
    skipped: list[dict[str, str]],
) -> None:
    payload = {
        "run_root": str(run_root),
        "checkpoint": checkpoint,
        "dry_run": dry_run,
        "results": results,
        "skipped": skipped,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()

    run_root = args.run_root
    if not run_root.is_absolute():
        run_root = repo_root() / run_root

    plans, skipped = build_upload_plans(
        run_root=run_root,
        repo_prefix=args.repo_prefix,
        checkpoint=args.checkpoint,
        strict=args.strict,
    )
    if not plans:
        raise SystemExit(f"No uploadable models found under {run_root}")

    print(f"Uploadable models: {len(plans)}")
    for plan in plans:
        print(f"- {plan.run_dir.name} -> {plan.repo_id}")
    if skipped:
        print(f"Skipped runs without requested checkpoint: {len(skipped)}")

    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None
    if args.hf_login:
        from huggingface_hub import login

        login(token=token)
        token = None

    api = None if args.dry_run else HfApi(token=token)
    results: list[dict[str, Any]] = []

    for plan in plans:
        try:
            result = upload_model_to_hub(
                run_dir=plan.run_dir,
                repo_id=plan.repo_id,
                checkpoint=args.checkpoint,
                private=args.private,
                commit_message=args.commit_message,
                token=token,
                dry_run=args.dry_run,
                api=api,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = {
                "repo_id": plan.repo_id,
                "run_dir": str(plan.run_dir),
                "source_dir": str(plan.source_dir),
                "checkpoint": args.checkpoint,
                "uploaded": False,
                "error": str(exc),
            }
        results.append(result)

    if args.manifest:
        manifest_path = args.manifest
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
        write_manifest(
            path=manifest_path,
            run_root=run_root,
            checkpoint=args.checkpoint,
            dry_run=args.dry_run,
            results=results,
            skipped=skipped,
        )
        print(f"Wrote manifest: {manifest_path}")

    done = sum(1 for result in results if result.get("uploaded"))
    planned = len(results) - done
    if args.dry_run:
        print(f"Dry run complete: staged {planned} planned uploads.")
    else:
        print(f"Uploaded {done} models.")


if __name__ == "__main__":
    main()
