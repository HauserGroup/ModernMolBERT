# %% [markdown]
# # Ideal Masking Probability — Standard Masking LR Sweep
#
# 1. Find optimal LR per masking probability (lowest best eval loss)
# 2. Benchmark each optimal model on MoleculeNet datasets
# 3. Plot ROC curves per dataset, one curve per masking probability

# %%
import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modernmolbert.eval.benchmarking_molecular_models.embed_modernmolbert import (
    embed_dataset,
    load_prepared_dataset,
    make_featurizer,
)
from modernmolbert.eval.benchmarking_molecular_models.common.config import (
    load_dataset_config,
    load_dataset_registry,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.const import (
    DEFAULT_MEMORY_WEIGHT,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.eval_metrics import (
    get_skfp_roc_auc,
    log_predictions,
    multioutput_auroc_score,
)
from modernmolbert.eval.benchmarking_molecular_models.supervised.train import (
    fit_and_eval_embedding,
)
from sklearn.metrics import auc, roc_curve


def repo_root() -> Path:
    """Repository root — parent of the notebooks/ directory."""
    return Path(__file__).resolve().parent.parent


ROOT = repo_root()
BENCH_ROOT = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models"
CONFIG_DIR = BENCH_ROOT / "config"

RUNS_ROOT = ROOT / "runs/chembl36_small_mask_mlm_lr_sweep"
DATA_DIR = ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared"
EMBEDDED_DIR = DATA_DIR / "embedded"
PREDICTIONS_DIR = DATA_DIR / "predictions"

for _d in [EMBEDDED_DIR, PREDICTIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Optimal LR per Masking Probability

# %%
records = []
for run_dir in sorted(RUNS_ROOT.iterdir()):
    if not run_dir.is_dir():
        continue
    m = re.match(r"mask_standard__mlm_([\dp]+)__lr_([\de\-]+)", run_dir.name)
    if m is None:
        continue
    mask_prob = float(m.group(1).replace("p", "."))
    lr_str = m.group(2)

    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        continue
    state = json.loads(state_path.read_text())

    records.append(
        {
            "mask_prob": mask_prob,
            "lr": lr_str,
            "best_eval_loss": state["best_metric"],
            "best_step": state["best_global_step"],
            "run_dir": str(run_dir),
        }
    )

df_sweep = pd.DataFrame(records).sort_values(["mask_prob", "best_eval_loss"])
print(df_sweep[["mask_prob", "lr", "best_eval_loss", "best_step"]].to_string(index=False))

# %%
df_optimal = (
    df_sweep.sort_values("best_eval_loss")
    .groupby("mask_prob", sort=True)
    .first()
    .reset_index()[["mask_prob", "lr", "best_eval_loss", "best_step", "run_dir"]]
)

print("\nOptimal LR per masking probability:")
print(df_optimal[["mask_prob", "lr", "best_eval_loss"]].to_string(index=False))

# %%
mask_colors = {0.15: "#1f77b4", 0.20: "#ff7f0e", 0.25: "#2ca02c"}
lr_order = sorted(df_sweep["lr"].unique(), key=float)

fig, ax = plt.subplots(figsize=(7, 4))
for mask_prob, grp in df_sweep.groupby("mask_prob"):
    grp = grp.set_index("lr").reindex(lr_order)
    ax.plot(
        lr_order,
        grp["best_eval_loss"],
        marker="o",
        label=f"mask={mask_prob}",
        color=mask_colors[mask_prob],  # type: ignore
    )

ax.set_xlabel("Learning rate")
ax.set_ylabel("Best eval loss (MLM)")
ax.set_title("LR sweep — standard masking")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "notebooks/01A_lr_sweep.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Benchmark Optimal Models on MoleculeNet

HEAD = "ridge"

# Discover which MoleculeNet datasets are already prepared
registry = load_dataset_registry(CONFIG_DIR)
MOLNET_CONFIG_KEYS = [
    key
    for key, cfg in registry.items()
    if "ogbg-mol" in cfg.name and (PREPARED_DIR / f"{cfg.name}.joblib").exists()
]
print(f"Found {len(MOLNET_CONFIG_KEYS)} prepared MoleculeNet datasets: {MOLNET_CONFIG_KEYS}")

# %%
# Steps 2–3: embed then score each optimal model; both steps are cached on disk
all_results = []

for _, opt_row in df_optimal.iterrows():
    mask_prob = opt_row["mask_prob"]
    run_dir = Path(opt_row["run_dir"])
    embedder_name = f"mask_{mask_prob}"

    print(f"\n=== mask_prob={mask_prob}, lr={opt_row['lr']} ===")

    # Build featurizer using the existing make_featurizer helper
    featurizer_args = argparse.Namespace(
        model_dir=run_dir / "final_model",
        tokenizer_path=run_dir / "ape_tokenizer",
        embedder=embedder_name,
        max_seq_length=256,
        pooling="mean",
        device="auto",
        batch_size=128,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        featurizer = make_featurizer(featurizer_args)

    for config_key in MOLNET_CONFIG_KEYS:
        cfg = load_dataset_config(CONFIG_DIR, config_key)
        ds_name = cfg.name

        # Step 2: embed (cached as joblib)
        embedded_path = EMBEDDED_DIR / ds_name / f"{embedder_name}.joblib"
        embedded_path.parent.mkdir(parents=True, exist_ok=True)

        if embedded_path.exists():
            embedded = joblib.load(embedded_path)
        else:
            prepared = load_prepared_dataset(PREPARED_DIR / f"{ds_name}.joblib")  # pre-existing
            print(f"  {ds_name}: embedding ...", end=" ", flush=True)
            embedded = embed_dataset(
                prepared, featurizer=featurizer, embedder_name=embedder_name, batch_size=128
            )
            joblib.dump(embedded, embedded_path)
            print("done")

        # Step 3: score (cached — .npz file acts as sentinel)
        # log_predictions writes: <pred_dir>/<dataset>/<embedder>/<head>.npz
        # Passing an absolute PREDICTIONS_DIR makes os.path.join ignore cwd
        pred_npz = PREDICTIONS_DIR / ds_name / embedder_name / f"{HEAD}.npz"
        pred_npz.parent.mkdir(parents=True, exist_ok=True)

        if not pred_npz.exists():
            print(f"  {ds_name}: fitting {HEAD} ...", end=" ", flush=True)
            head_result = fit_and_eval_embedding(
                dataset=embedded,
                model_head=HEAD,
                memory_weight=cfg.get("memory_weight", DEFAULT_MEMORY_WEIGHT),
            )
            log_predictions(head_result, str(PREDICTIONS_DIR))
            roc_auc = get_skfp_roc_auc(head_result.y_test_pred, head_result.y_test_true)
            print(f"ROC-AUC={roc_auc:.4f}")
        else:
            with np.load(pred_npz) as npz:
                roc_auc = float(multioutput_auroc_score(npz["y_true"], npz["y_score"]))
            print(f"  {ds_name}: cached ROC-AUC={roc_auc:.4f}")

        all_results.append(
            {"dataset": ds_name, "mask_prob": mask_prob, "lr": opt_row["lr"], "roc_auc": roc_auc}
        )

# %%
df_results = pd.DataFrame(all_results)
pivot = df_results.groupby(["dataset", "mask_prob"])["roc_auc"].mean().unstack("mask_prob").round(4)
pivot.columns = [f"mask={c}" for c in pivot.columns]
print("\nMean ROC-AUC per dataset and masking probability:")
print(pivot.to_string())

# %% [markdown]
# ## 3. ROC Curves per MoleculeNet Dataset


# %%
def macro_roc_from_npz(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """Macro-average ROC curve from a prediction .npz (y_true, y_score)."""
    with np.load(npz_path) as npz:
        y_true = np.asarray(npz["y_true"], dtype=float)
        y_score = np.asarray(npz["y_score"], dtype=float)

    grid = np.linspace(0, 1, 200)
    tprs, aucs = [], []

    cols = (
        [(y_true, y_score)]
        if y_true.ndim == 1
        else [(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
    )

    for yt, ys in cols:
        mask = np.isfinite(yt) & np.isfinite(ys)
        yt, ys = yt[mask], ys[mask]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt.astype(int), ys)
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(auc(fpr, tpr))

    if not tprs:
        return None, None, None
    return grid, np.mean(tprs, axis=0), float(np.mean(aucs))


ds_names = [load_dataset_config(CONFIG_DIR, k).name for k in MOLNET_CONFIG_KEYS]
n_ds = len(ds_names)
ncols = 3
nrows = (n_ds + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for ax_idx, ds_name in enumerate(ds_names):
    ax = axes[ax_idx]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    for mask_prob in sorted(df_optimal["mask_prob"]):
        pred_npz = PREDICTIONS_DIR / ds_name / f"mask_{mask_prob}" / f"{HEAD}.npz"
        if not pred_npz.exists():
            continue
        fpr, tpr, roc_auc = macro_roc_from_npz(pred_npz)
        if fpr is None:
            continue
        ax.plot(
            fpr,
            tpr,
            color=mask_colors[mask_prob],
            lw=1.8,
            label=f"mask={mask_prob} (AUC={roc_auc:.3f})",
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.set_title(ds_name, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.2)

for ax in axes[n_ds:]:
    ax.set_visible(False)

fig.suptitle(
    "MoleculeNet ROC Curves — Standard Masking, Optimal LR per Mask Probability",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.savefig(ROOT / "notebooks/01A_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {ROOT / 'notebooks/01A_roc_curves.png'}")
