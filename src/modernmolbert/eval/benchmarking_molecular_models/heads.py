from __future__ import annotations

from typing import Literal

from modernmolbert.eval.downstream import FrozenDownstreamConfig

HeadName = Literal[
    "auto",
    "logreg",
    "logistic_regression",
    "rf",
    "random_forest_classifier",
    "ridge",
    "ridge_cv",
    "random_forest_regressor",
]

LightweightParityHeadName = Literal["auto", "rf", "ridge", "knn"]


def normalize_head_name(head: str) -> str:
    aliases = {
        "logreg": "logistic_regression",
        "rf": "random_forest_classifier",
    }
    return aliases.get(head, head)


def downstream_configs_for_heads(
    heads: list[str],
    *,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    """Build suite downstream configs for selected lightweight heads."""

    if not heads or "auto" in heads:
        heads = ["logistic_regression", "ridge"]

    classification: list[dict[str, object]] = []
    regression: list[dict[str, object]] = []

    for raw_head in heads:
        head = normalize_head_name(raw_head)

        if head == "logistic_regression":
            classification.append(
                {
                    "name": "logistic_regression",
                    "model_type": "logistic_regression",
                    "standardize": True,
                    "random_state": seed,
                    "params": {
                        "class_weight": "balanced",
                        "max_iter": 5000,
                        "C": 1.0,
                    },
                }
            )
        elif head == "random_forest_classifier":
            classification.append(
                {
                    "name": "random_forest_classifier",
                    "model_type": "random_forest_classifier",
                    "standardize": False,
                    "random_state": seed,
                    "params": {
                        "n_estimators": 500,
                        "class_weight": "balanced",
                        "min_samples_leaf": 1,
                        "n_jobs": -1,
                    },
                }
            )
        elif head in {"ridge", "ridge_cv", "random_forest_regressor"}:
            regression.append(_regression_head_config(head=head, seed=seed))
        else:
            raise ValueError(
                f"Unknown head {raw_head!r}. "
                "Use auto, logreg, logistic_regression, rf, "
                "random_forest_classifier, ridge, ridge_cv, or random_forest_regressor."
            )

    return {
        "classification": classification,
        "regression": regression,
    }


def lightweight_parity_downstream_configs_for_heads(
    heads: list[str],
) -> dict[str, list[dict[str, object]]]:
    """Build placeholder suite configs for lightweight parity classification heads."""

    if not heads or "auto" in heads:
        heads = ["rf", "ridge", "knn"]

    classification: list[dict[str, object]] = []
    for head in heads:
        if head not in {"rf", "ridge", "knn"}:
            raise ValueError(
                f"Unsupported lightweight parity head {head!r}. Use auto, rf, ridge, or knn."
            )
        classification.append(
            {
                "name": head,
                "model_type": "lightweight_parity_classifier",
                "standardize": False,
                "params": {},
            }
        )

    return {
        "classification": classification,
        "regression": [],
    }


def _regression_head_config(*, head: str, seed: int) -> dict[str, object]:
    if head == "ridge":
        return {
            "name": "ridge",
            "model_type": "ridge",
            "standardize": True,
            "random_state": seed,
            "params": {"alpha": 1.0},
        }

    if head == "ridge_cv":
        return {
            "name": "ridge_cv",
            "model_type": "ridge_cv",
            "standardize": True,
            "random_state": seed,
            "params": {"alphas": [0.01, 0.1, 1.0, 10.0, 100.0]},
        }

    return {
        "name": "random_forest_regressor",
        "model_type": "random_forest_regressor",
        "standardize": False,
        "random_state": seed,
        "params": {
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        },
    }


def frozen_downstream_config(head: str, *, seed: int = 13) -> FrozenDownstreamConfig:
    """Return a single existing eval downstream config for direct callers."""

    configs = downstream_configs_for_heads([head], seed=seed)
    entries = configs["classification"] + configs["regression"]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one config for head {head!r}")

    entry = entries[0]
    return FrozenDownstreamConfig(
        model_type=str(entry["model_type"]),
        params=dict(entry.get("params", {})),
        random_state=seed,
        standardize=bool(entry.get("standardize", True)),
    )
