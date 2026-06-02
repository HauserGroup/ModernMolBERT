import argparse
import gc
import logging as log
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    write_dataset_checkpoint,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
    load_embedding_config,
    load_yaml_config,
)
from modernmolbert.eval.benchmarking_molecular_models.src.common.types import (
    EmbeddedDataset,
    EmbeddingConfig,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.models import (
    AVAILABLE_HEADS,
)
from modernmolbert.eval.benchmarking_molecular_models.src.eval.supervised.procedure import (
    eval_procedure,
    load_embedded_dataset,
)


@dataclass(frozen=True)
class DatasetItem:
    """Resolved dataset entry used by the scoring loop.

    config_name:
        Name returned by expand_dataset_selection(...), e.g. clf_ogbg-molhiv.
        This is only a config selector.

    name:
        Canonical prepared dataset name from dataset_info.name, e.g. ogbg-molhiv.
        This is the identity used for logging, results, skips, and checkpoints.

    info:
        Fully loaded dataset config object passed into eval_procedure(...).
    """

    config_name: str
    name: str
    info: Any


@dataclass(frozen=True)
class SkippedItem:
    """Dataset skipped before scoring, with a human-readable reason."""

    name: str
    reason: str


@dataclass(frozen=True)
class SubsampleConfig:
    """Scoring-time dataset subsampling options."""

    max_samples: int
    scope: str
    seed: int


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a config value from either dict-like or attribute-like configs."""
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass

    return getattr(cfg, key, default)


def as_list(value: Any) -> list[Any]:
    """Return value as a list while treating None as an empty list."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return list(value)


def normalize_name(name: str | Path) -> str:
    """Normalize dataset selectors and names for matching.

    Handles:
      ogbg-molhiv
      clf_ogbg-molhiv
      reg_esol
      ogbg-molhiv.yaml
      config/datasets/clf_ogbg-molhiv.yaml

    The prepared dataset identity remains dataset_info.name. This function is
    only for user-facing matching, especially --skip_datasets.
    """
    stem = Path(str(name)).stem

    for prefix in ("clf_", "reg_"):
        if stem.startswith(prefix):
            stem = stem.removeprefix(prefix)

    return stem


def safe_file_component(value: str) -> str:
    """Make a conservative filename component."""
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def dataset_checkpoint_path(
    *,
    checkpoint_dir: str | Path,
    dataset: str,
    embedder: str,
) -> Path:
    """Return the expected per-dataset checkpoint path.

    This path must match write_dataset_checkpoint(...). If checkpoint writing is
    later moved fully into praski_export.py, this function should be moved there
    too and imported here.
    """
    safe_dataset = safe_file_component(dataset)
    safe_embedder = safe_file_component(embedder)
    return Path(checkpoint_dir) / f"{safe_dataset}__{safe_embedder}.csv"


def checkpoint_exists(
    *,
    checkpoint_dir: Path | None,
    dataset: str,
    embedder: str,
) -> bool:
    """Check whether a dataset/embedder checkpoint already exists."""
    if checkpoint_dir is None:
        return False

    path = dataset_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        embedder=embedder,
    )
    return path.exists() and path.stat().st_size > 0


def make_short_model_name(model_name: str) -> str:
    """Convert model/embedder path or identifier to the stored embedding name.

    Existing embedding files are expected under:

        data/embedded/<dataset>/<short_model_name>.joblib

    Examples:
      runs/modernmolbert_best_span -> modernmolbert_best_span
      modernmolbert_best_span      -> modernmolbert_best_span
    """
    final_component = model_name.split("/")[-1]

    if "gpt" in model_name.lower():
        return final_component

    return final_component.split(".")[0]


def resolve_model_name(cfg: Any, args: argparse.Namespace) -> str | None:
    """Resolve embedder/model name from CLI, compatibility overrides, or config."""
    model_name = args.model_name

    for override in args.overrides:
        if override.startswith("model_name=") or override.startswith("embedder="):
            model_name = override.split("=", 1)[1]

    model_cfg = cfg_get(cfg, "model", {})

    return (
        model_name
        or cfg_get(model_cfg, "embedding_name", None)
        or cfg_get(cfg, "model_name", None)
        or cfg_get(model_cfg, "model_name", None)
    )


def resolve_dataset_selections(cfg: Any, args: argparse.Namespace) -> list[str]:
    """Return dataset selections from CLI or score.yaml."""
    return as_list(args.datasets or cfg_get(cfg, "datasets", ["all"]))


def resolve_skip_set(cfg: Any, args: argparse.Namespace) -> set[str]:
    """Return normalized skip names from CLI or score.yaml."""
    skip_values = (
        args.skip_datasets
        or cfg_get(cfg, "skip_datasets", None)
        or cfg_get(cfg, "skip_dataset", [])
    )

    return {normalize_name(name) for name in as_list(skip_values)}


def resolve_subsample_config(cfg: Any, args: argparse.Namespace) -> SubsampleConfig | None:
    """Return scoring-time subsampling options from CLI or score.yaml."""
    max_samples = args.subsample_size
    if max_samples is None:
        max_samples = cfg_get(cfg, "subsample_size", cfg_get(cfg, "subsample", None))

    if max_samples is None:
        return None

    try:
        max_samples = int(max_samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("--subsample must be a positive integer") from exc

    if max_samples <= 0:
        raise ValueError("--subsample must be a positive integer")

    scope = args.subsample_scope or cfg_get(cfg, "subsample_scope", "train")
    if scope not in {"train", "all"}:
        raise ValueError("--subsample-scope must be one of: train, all")

    seed = args.subsample_seed
    if seed is None:
        seed = cfg_get(cfg, "subsample_seed", 13)

    return SubsampleConfig(max_samples=max_samples, scope=str(scope), seed=int(seed))


def make_scoring_model_name(model_name: str, subsample: SubsampleConfig | None) -> str:
    """Return the result/checkpoint identity for this scoring run."""
    if subsample is None:
        return model_name

    return f"{model_name}__subsample_{subsample.scope}{subsample.max_samples}_seed{subsample.seed}"


def split_indices(dataset: EmbeddedDataset) -> dict[str, np.ndarray]:
    """Normalize split index containers to integer numpy arrays."""
    return {
        str(split_name): np.asarray(indices, dtype=int)
        for split_name, indices in dataset.splits.items()
    }


def allocate_subsample_counts(
    split_sizes: dict[str, int],
    max_samples: int,
) -> dict[str, int]:
    """Allocate a total sample budget across splits proportional to split size."""
    total = sum(split_sizes.values())
    if total <= max_samples:
        return dict(split_sizes)

    nonempty = {name: size for name, size in split_sizes.items() if size > 0}
    if not nonempty:
        return {name: 0 for name in split_sizes}

    counts = {name: 0 for name in split_sizes}
    if max_samples >= len(nonempty):
        for name in nonempty:
            counts[name] = 1

    remaining = max_samples - sum(counts.values())
    weights = {name: max(0, size - counts[name]) for name, size in nonempty.items()}
    remaining_weight = sum(weights.values())

    if remaining > 0 and remaining_weight > 0:
        fractional: list[tuple[float, str]] = []
        for name, weight in weights.items():
            exact = remaining * weight / remaining_weight
            add = int(np.floor(exact))
            add = min(add, split_sizes[name] - counts[name])
            counts[name] += add
            fractional.append((exact - add, name))

        spare = max_samples - sum(counts.values())
        for _, name in sorted(fractional, reverse=True):
            if spare <= 0:
                break
            capacity = split_sizes[name] - counts[name]
            if capacity <= 0:
                continue
            counts[name] += 1
            spare -= 1

    if sum(counts.values()) < max_samples:
        for name, size in sorted(nonempty.items(), key=lambda item: item[1], reverse=True):
            while counts[name] < size and sum(counts.values()) < max_samples:
                counts[name] += 1

    return counts


def sample_split_indices(
    *,
    splits: dict[str, np.ndarray],
    split_names: set[str],
    max_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample selected splits and leave unselected splits unchanged."""
    rng = np.random.default_rng(seed)
    target_sizes = {
        name: int(len(indices)) for name, indices in splits.items() if name in split_names
    }
    target_counts = allocate_subsample_counts(target_sizes, max_samples)

    sampled: dict[str, np.ndarray] = {}
    for name, indices in splits.items():
        if name not in split_names:
            sampled[name] = indices
            continue

        count = target_counts.get(name, 0)
        if count >= len(indices):
            sampled[name] = indices
        elif count <= 0:
            sampled[name] = np.asarray([], dtype=int)
        else:
            sampled[name] = np.sort(rng.choice(indices, size=count, replace=False))

    return sampled


def subsample_embedded_dataset(
    dataset: EmbeddedDataset,
    *,
    subsample: SubsampleConfig,
    embedder_name: str,
) -> EmbeddedDataset:
    """Return a split-aware scoring subset without mutating the loaded embedding."""
    splits = split_indices(dataset)
    if subsample.scope == "train":
        sampled_split_names = {"train", "valid"}
    elif subsample.scope == "all":
        sampled_split_names = set(splits)
    else:
        raise ValueError(f"Unknown subsample scope: {subsample.scope}")

    sampled_splits = sample_split_indices(
        splits=splits,
        split_names=sampled_split_names,
        max_samples=subsample.max_samples,
        seed=subsample.seed,
    )

    kept_arrays = [indices for indices in sampled_splits.values() if len(indices) > 0]
    selected = np.unique(np.concatenate(kept_arrays)) if kept_arrays else np.asarray([], dtype=int)
    old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(selected.tolist())}

    remapped_splits = {
        name: [
            old_to_new[int(old_idx)] for old_idx in indices.tolist() if int(old_idx) in old_to_new
        ]
        for name, indices in sampled_splits.items()
    }

    y = dataset.y.iloc[selected].reset_index(drop=True)
    subset = EmbeddedDataset(
        name=dataset.name,
        task=dataset.task,
        embedder=embedder_name,
        splits=remapped_splits,
        X=dataset.X[selected],
        y=y,
    )

    log.info(
        "Subsampled dataset %s scope=%s max_samples=%s seed=%s: X %s -> %s; splits=%s",
        dataset.name,
        subsample.scope,
        subsample.max_samples,
        subsample.seed,
        dataset.X.shape,
        subset.X.shape,
        {name: len(indices) for name, indices in remapped_splits.items()},
    )
    return subset


def load_dataset_items(
    *,
    config_dir: Path,
    selections: list[str],
) -> list[DatasetItem]:
    """Expand dataset selections and load configs exactly once.

    The returned DatasetItem.name is the canonical prepared dataset name.
    This is the name used for skip logic, checkpoint logic, logs, and results.
    """
    config_names = expand_dataset_selection(config_dir, selections)

    items: list[DatasetItem] = []

    for config_name in config_names:
        dataset_info = load_dataset_config(config_dir, config_name)

        items.append(
            DatasetItem(
                config_name=str(config_name),
                name=str(dataset_info.name),
                info=dataset_info,
            )
        )

    return items


def should_skip_item(item: DatasetItem, skip_set: set[str]) -> bool:
    """Return True if a dataset item matches user-requested skips."""
    if not skip_set:
        return False

    candidate_names = {
        normalize_name(item.config_name),
        normalize_name(item.name),
    }

    return bool(candidate_names & skip_set)


def build_run_plan(
    *,
    items: list[DatasetItem],
    skip_set: set[str],
    checkpoint_dir: Path | None,
    embedder: str,
    resume: bool,
) -> tuple[list[DatasetItem], list[SkippedItem]]:
    """Apply all pre-run decisions once.

    This is the only place where requested skips and checkpoint-resume skips are
    applied. The scoring loop should only iterate over the returned run_items.
    """
    run_items: list[DatasetItem] = []
    skipped_items: list[SkippedItem] = []

    for item in items:
        if should_skip_item(item, skip_set):
            skipped_items.append(SkippedItem(item.name, "requested skip"))
            continue

        if resume and checkpoint_exists(
            checkpoint_dir=checkpoint_dir,
            dataset=item.name,
            embedder=embedder,
        ):
            skipped_items.append(SkippedItem(item.name, "checkpoint exists"))
            continue

        run_items.append(item)

    return run_items, skipped_items


def print_run_plan(
    *,
    items: list[DatasetItem],
    run_items: list[DatasetItem],
    skipped_items: list[SkippedItem],
    skip_set: set[str],
    embedder: str,
    heads: list[str],
    override: bool,
    safe: bool,
    resume: bool,
    source_embedder: str | None = None,
    subsample: SubsampleConfig | None = None,
) -> None:
    """Print the complete plan before scoring starts."""
    print(
        (
            f"[score] embedder={embedder}  "
            f"expanded_datasets={len(items)}  "
            f"datasets_to_run={len(run_items)}  "
            f"heads={heads}  "
            f"override={override}  "
            f"safe={safe}  "
            f"resume={resume}  "
            f"subsample={subsample or 'disabled'}"
        ),
        flush=True,
    )

    if source_embedder is not None and source_embedder != embedder:
        print(f"[score] source_embedder={source_embedder}", flush=True)

    if skip_set:
        print(f"[score] requested skips: {sorted(skip_set)}", flush=True)

    if skipped_items:
        print("[score] skipped datasets:", flush=True)
        for skipped in skipped_items:
            print(f"  - {skipped.name}: {skipped.reason}", flush=True)

    print("[score] run plan:", flush=True)
    for idx, item in enumerate(run_items, start=1):
        print(f"  [{idx:>2}/{len(run_items)}] {item.name}", flush=True)


def run_eval(
    *,
    safe: bool,
    embed_config: EmbeddingConfig,
    full_model_name: str,
    short_model_name: str,
    dataset_info: Any,
    model_head: str,
    output_csv: Path,
    override: bool,
    preloaded: Any = None,
) -> bool:
    """Run one dataset/head evaluation.

    Returns True if the head completed successfully.

    In safe mode:
      exceptions are logged and False is returned.

    In non-safe mode:
      exceptions propagate.
    """
    log.info(
        "Evaluating model %s on dataset %s with metric %s using head %s",
        full_model_name,
        dataset_info.name,
        dataset_info.metric,
        model_head,
    )

    try:
        eval_procedure(
            dataset_info=dataset_info,
            embedded_dir=embed_config.embedded_directory,
            predictions_dir=embed_config.predictions_directory,
            model_name=short_model_name,
            model_head=model_head,
            output_csv=output_csv,
            override=override,
            preloaded=preloaded,
        )
    except Exception as exc:
        if not safe:
            raise

        import traceback

        log.error(
            "Error during evaluation for dataset=%s head=%s: %s",
            dataset_info.name,
            model_head,
            exc,
        )
        log.error(traceback.format_exc())
        return False

    return True


def write_checkpoint_if_successful(
    *,
    results_csv: Path,
    checkpoint_dir: Path | None,
    dataset: str,
    embedder: str,
    dataset_success: bool,
) -> None:
    """Write a per-dataset checkpoint if at least one head succeeded."""
    if checkpoint_dir is None:
        return

    if not dataset_success:
        print(
            f"[score] {dataset}: no successful heads; not writing checkpoint",
            flush=True,
        )
        return

    write_dataset_checkpoint(
        results_csv=results_csv,
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        embedder=embedder,
    )

    checkpoint = dataset_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        embedder=embedder,
    )

    print(f"[score] wrote checkpoint: {checkpoint}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing molecular embedding files.")

    parser.add_argument(
        "--embedder",
        "--model-name",
        dest="model_name",
        required=False,
        help=(
            "Embedding/model name. Expected embeddings at "
            "data/embedded/<dataset>/<embedder>.joblib."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset config stems, globs, or 'all'. Defaults to config/score.yaml.",
    )

    parser.add_argument(
        "--heads",
        nargs="+",
        choices=AVAILABLE_HEADS,
        default=AVAILABLE_HEADS,
        help="Scoring heads to run.",
    )

    parser.add_argument(
        "--skip_datasets",
        "--skip-datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Skip one or more datasets by name. Accepts canonical dataset names "
            "such as 'ogbg-molhiv' and config names such as 'clf_ogbg-molhiv'."
        ),
    )

    parser.add_argument(
        "--config-dir",
        default="config",
        help="Config directory relative to this script directory.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/benchmark_results.csv"),
        help="Main accumulated results CSV.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for per-dataset result checkpoint CSVs.",
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip dataset/embedder combinations with existing checkpoint CSVs.",
    )

    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use cached scoring results if supported by eval_procedure. "
            "Equivalent to override = not cache."
        ),
    )

    parser.add_argument(
        "--safe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log errors and continue instead of aborting on the first failed head.",
    )

    parser.add_argument(
        "--subsample",
        "--subsample-size",
        dest="subsample_size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Subsample at most N rows per dataset while scoring. By default this "
            "limits the train+valid rows and keeps the full test split."
        ),
    )

    parser.add_argument(
        "--subsample-scope",
        choices=["train", "all"],
        default=None,
        help=(
            "Which splits to subsample: 'train' samples train+valid only and "
            "keeps test intact; 'all' samples every split."
        ),
    )

    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="Random seed for scoring-time subsampling. Defaults to 42.",
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Compatibility support for key=value overrides such as model_name=my_embedder.",
    )

    return parser.parse_args()


def main() -> int:
    log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()

    root = Path(__file__).resolve().parent
    config_dir = root / args.config_dir

    cfg = load_yaml_config(config_dir / "score.yaml")
    embedding_cfg = load_embedding_config(config_dir)

    model_name = resolve_model_name(cfg, args)
    if not model_name:
        raise ValueError(
            "Scoring requires --embedder <name>. "
            "Expected embeddings at data/embedded/<dataset>/<embedder>.joblib."
        )

    short_model_name = make_short_model_name(model_name)

    cache = cfg_get(cfg, "cache", True) if args.cache is None else args.cache
    safe = cfg_get(cfg, "safe", False) if args.safe is None else args.safe
    override = not cache

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_selections = resolve_dataset_selections(cfg, args)
    skip_set = resolve_skip_set(cfg, args)
    subsample_config = resolve_subsample_config(cfg, args)
    scoring_model_name = make_scoring_model_name(short_model_name, subsample_config)

    items = load_dataset_items(
        config_dir=config_dir,
        selections=dataset_selections,
    )

    run_items, skipped_items = build_run_plan(
        items=items,
        skip_set=skip_set,
        checkpoint_dir=args.checkpoint_dir,
        embedder=scoring_model_name,
        resume=args.resume,
    )

    print_run_plan(
        items=items,
        run_items=run_items,
        skipped_items=skipped_items,
        skip_set=skip_set,
        embedder=scoring_model_name,
        heads=list(args.heads),
        override=override,
        safe=safe,
        resume=args.resume,
        source_embedder=short_model_name,
        subsample=subsample_config,
    )

    embed_config = EmbeddingConfig(**embedding_cfg)

    attempted = 0
    successful_datasets = 0
    failed_datasets = 0
    failures: list[str] = []

    for idx, item in enumerate(run_items, start=1):
        attempted += 1
        dataset_success = False

        embedded_data = load_embedded_dataset(
            embedded_dir=embed_config.embedded_directory,
            dataset_info=item.info,
            model_name=short_model_name,
        )
        if embedded_data is None:
            log.error(
                "Skipping dataset=%s because no embedding was loaded for source embedder=%s",
                item.name,
                short_model_name,
            )
            failed_datasets += 1
            failures.append(f"{item.name}: missing embedding")
            continue

        if subsample_config is not None:
            embedded_data = subsample_embedded_dataset(
                embedded_data,
                subsample=subsample_config,
                embedder_name=scoring_model_name,
            )

        for model_head in args.heads:
            print(
                f"[{idx:>2}/{len(run_items)}] {item.name}  head={model_head}",
                flush=True,
            )

            success = run_eval(
                safe=safe,
                embed_config=embed_config,
                full_model_name=model_name,
                short_model_name=scoring_model_name,
                dataset_info=item.info,
                model_head=model_head,
                output_csv=args.output_csv,
                override=override,
                preloaded=embedded_data,
            )
            if success is False:
                failures.append(f"{item.name}/{model_head}: evaluation failed")
            else:
                dataset_success = True

        del embedded_data
        gc.collect()

        write_checkpoint_if_successful(
            results_csv=args.output_csv,
            checkpoint_dir=args.checkpoint_dir,
            dataset=item.name,
            embedder=scoring_model_name,
            dataset_success=dataset_success,
        )

        if dataset_success:
            successful_datasets += 1
        else:
            failed_datasets += 1

    print(
        (
            "[score] complete: "
            f"expanded={len(items)}, "
            f"skipped={len(skipped_items)}, "
            f"attempted={attempted}, "
            f"successful_datasets={successful_datasets}, "
            f"failed_datasets={failed_datasets}"
        ),
        flush=True,
    )

    if failures:
        print("[score] failures:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)

    return len(failures)


if __name__ == "__main__":
    failures = main()
    if failures:
        raise SystemExit(1)
    print("All done", flush=True)
