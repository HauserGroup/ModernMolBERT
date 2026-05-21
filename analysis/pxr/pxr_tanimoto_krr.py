"""
pxr_tanimoto_krr.py
====================
Selectivity-weighted heteroskedastic Tanimoto Kernel Ridge Regression
for hPXR pEC50 prediction — PXR Challenge 2025, Activity Track.

Method summary
--------------
  yhat(x*) = k*^T @ alpha + y_mean
  alpha     = (K + lambda * D)^{-1} @ (y - y_mean)

  K_ij   = binary ECFP4 Tanimoto kernel
  D_ii   = (sigma_i^2 + sigma_floor^2) / w_i     [noise-inflated diagonal]
  w_i    = w_selectivity_i * w_emax_i            [reliability weight]
  y_mean = reliability-weighted mean of training pEC50 (prevents shrinkage to zero)

CV finding: ECFP6 added nothing (best rho=1.0 in all mix modes).
            Count fingerprints give larger but harder CV neighborhoods
            than binary at the same threshold and are not directly
            comparable.  Binary ECFP4 is the principled default —
            the challenge analog set was built from binary ECFP4 Tanimoto.

Dependencies
------------
  pip install numpy scipy pandas rdkit-pypi pyarrow
"""

# %%
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------


class Config:
    """
    Minimal, principled configuration.

    Fixed constants (from challenge geometry — not tunable):
        ANCHOR_MIN_PEC50, ANCHOR_MIN_SEL, NEIGHBORHOOD_SIM encode the
        challenge design (EC50 ≤ 1 µM seed criterion, 1.5 log-unit
        selectivity margin, ECFP4 Tanimoto ≥ 0.4 analog expansion).

    Model-choice parameter (tested by CV):
        FINGERPRINT_MODE — "ecfp4_binary" (default) or "ecfp4_count".
        CV confirmed ECFP6 adds nothing (ρ=1.0 won in all mix modes).

    Tuned regularisation parameter:
        LAMBDA — selected by analog-neighbourhood CV on MAE.
        Grid search resolved λ=2.0 as the interior optimum for binary.

    Fixed assay-noise safeguard:
        SIGMA_FLOOR — prevents near-zero reported SE from dominating.
    """

    # -- Fingerprint representation ----------------------------------------
    FINGERPRINT_MODE: str = "ecfp4_binary"  # or "ecfp4_count"
    FP_BITS: int = 2048

    # -- Observation noise (fixed) -----------------------------------------
    DEFAULT_SIGMA: float = 0.25
    SIGMA_FLOOR: float = 0.20

    # -- Hyperparameter grid (tuned by analog-neighbourhood CV) -----------
    LAMBDA_GRID: list = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    # -- Challenge geometry (fixed — encode the analog-set design) ---------
    ANCHOR_MIN_PEC50: float = 6.0
    ANCHOR_MIN_SEL: float = 1.5
    NEIGHBORHOOD_SIM: float = 0.4  # diagnostic / prediction neighbourhood

    # CV uses a slightly relaxed threshold to generate enough folds.
    # 0.35 is still local and avoids cross-series contamination.
    CV_NEIGHBORHOOD_SIM: float = 0.35

    # -- Diagnostics -------------------------------------------------------
    RUN_LOCAL_MEDIAN_DIAGNOSTIC: bool = True
    DIAGNOSTIC_DISAGREEMENT: float = 0.8
    DIAGNOSTIC_MAX_DISPLAY: int = 20  # max flagged compounds to print
    DIAGNOSTIC_TOP_K: int = 10  # neighbours shown per flagged compound

    # -- Phase 2 cluster-offset (off by default; enable after AS1 reveal) --
    USE_PHASE2_CLUSTER_OFFSET: bool = False
    OFFSET_KAPPA: float = 3.0
    MIN_OFFSET: float = 0.3


# ---------------------------------------------------------------------------
# 1.  SMILES utilities
# ---------------------------------------------------------------------------


def canonicalize(smi: str):
    """Return RDKit canonical SMILES, or None if unparsable."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


# ---------------------------------------------------------------------------
# 2.  Fingerprint computation
# ---------------------------------------------------------------------------


def _morgan_gen(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def compute_ecfp(
    smiles_list: list, radius: int, n_bits: int = 2048, use_counts: bool = False
) -> np.ndarray:
    """
    Morgan (ECFP) fingerprints for a list of canonical SMILES.
    Returns float32 array (n, n_bits). Failed SMILES → zero vectors + warning.
    """
    gen = _morgan_gen(radius, n_bits)
    fps, failed = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            failed.append(i)
            continue
        arr = (
            gen.GetCountFingerprintAsNumPy(mol) if use_counts else gen.GetFingerprintAsNumPy(mol)
        ).astype(np.float32)
        fps.append(arr)
    if failed:
        warnings.warn(
            f"Could not parse {len(failed)} SMILES "
            f"(indices: {failed[:5]}{'...' if len(failed) > 5 else ''}). "
            "Zero vectors used.",
            UserWarning,
            stacklevel=2,
        )
    return np.array(fps, dtype=np.float32)


def build_fingerprints(smiles: list, cfg: Config) -> np.ndarray:
    """
    Build fingerprint array according to FINGERPRINT_MODE.
    Returns (n, FP_BITS) float32 array.
    """
    mode = cfg.FINGERPRINT_MODE
    bits = cfg.FP_BITS
    if mode == "ecfp4_binary":
        return compute_ecfp(smiles, 2, bits, use_counts=False)
    if mode == "ecfp4_count":
        return compute_ecfp(smiles, 2, bits, use_counts=True)
    raise ValueError(f"Unknown FINGERPRINT_MODE: {mode!r}")


def _uses_counts(mode: str) -> bool:
    return "count" in mode


# ---------------------------------------------------------------------------
# 3.  Tanimoto kernels
# ---------------------------------------------------------------------------


def binary_tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    T(a,b) = |a∩b| / |a∪b| = dot(a,b) / (||a||_1 + ||b||_1 − dot(a,b))
    Vectorised via matmul. A:(n,d), B:(m,d) → (n,m).
    """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    inter = A @ B.T
    norm_A = A.sum(axis=1)
    norm_B = B.sum(axis=1)
    union = norm_A[:, None] + norm_B[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def count_tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Generalised Tanimoto for count fps:
    T(a,b) = Σ min / Σ max = (||a||_1+||b||_1−L1) / (||a||_1+||b||_1+L1)
    A:(n,d), B:(m,d) → (n,m).
    """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    norm_A = A.sum(axis=1)
    norm_B = B.sum(axis=1)
    L1 = cdist(A, B, metric="cityblock")
    numerator = norm_A[:, None] + norm_B[None, :] - L1
    denom = norm_A[:, None] + norm_B[None, :] + L1
    return np.where(denom > 0, numerator / denom, 0.0)


def tanimoto(A: np.ndarray, B: np.ndarray, use_counts: bool) -> np.ndarray:
    return count_tanimoto(A, B) if use_counts else binary_tanimoto(A, B)


# ---------------------------------------------------------------------------
# 4.  Training reliability weights  (fixed functions, not config knobs)
# ---------------------------------------------------------------------------


def selectivity_weight(delta: float, has_counter: bool) -> float:
    """
    Reliability weight from primary-minus-counter pEC50 selectivity delta.
    Thresholds match the challenge seed-selection criterion (delta ≥ 1.5).
    """
    if not has_counter or np.isnan(delta):
        return 0.75  # no counter data: neutral
    if delta >= 1.5:
        return 1.00  # genuinely selective
    if delta >= 0.5:
        return 0.60  # moderate selectivity
    return 0.30  # non-specific signal


def emax_weight(emax: float) -> float:
    """
    Mild reliability modifier from Emax vs positive control.
    Full agonists give the cleanest EC50 fits.
    """
    if pd.isna(emax):
        return 0.85
    if emax >= 0.7:
        return 1.00
    if emax >= 0.4:
        return 0.85
    return 0.70


def compute_noise_diagonal(df: pd.DataFrame, cfg: Config = None) -> np.ndarray:
    """
    D_ii = (sigma_i^2 + sigma_floor^2) / w_i   — vectorised.
    Required columns: pEC50_se, delta_sel, has_counter, emax_vs_ctrl.
    """
    if cfg is None:
        cfg = Config()
    sigma = (
        df["pEC50_se"]
        .fillna(cfg.DEFAULT_SIGMA)
        .clip(lower=cfg.SIGMA_FLOOR)
        .to_numpy(dtype=np.float64)
    )
    delta = df["delta_sel"].to_numpy(dtype=np.float64)
    has_counter = df["has_counter"].to_numpy(dtype=bool)
    emax = df["emax_vs_ctrl"].to_numpy(dtype=np.float64)
    w_sel = np.where(
        ~has_counter | np.isnan(delta),
        0.75,
        np.where(delta >= 1.5, 1.00, np.where(delta >= 0.5, 0.60, 0.30)),
    )
    w_emax = np.where(
        np.isnan(emax), 0.85, np.where(emax >= 0.7, 1.00, np.where(emax >= 0.4, 0.85, 0.70))
    )
    w = w_sel * w_emax
    return (sigma**2 + cfg.SIGMA_FLOOR**2) / w


# ---------------------------------------------------------------------------
# 5.  Tanimoto KRR model
# ---------------------------------------------------------------------------


class TanimotoKRR:
    """
    Tanimoto Kernel Ridge Regression with heteroskedastic regularisation
    and reliability-weighted mean centering.

    Fit:     alpha  = (K + lambda * D)^{-1} @ (y - y_mean)
    Predict: yhat   = K_star @ alpha + y_mean

    Mean centering is critical: without it, large lambda shrinks predictions
    toward zero rather than toward the training mean.
    """

    def __init__(self, lam: float = 2.0, cfg: Config = None):
        self.lam = lam
        self.cfg = cfg or Config()
        self.alpha_ = None
        self.y_mean_ = None
        self.fp_train = None

    def fit(self, fp_train: np.ndarray, y_train: np.ndarray, D_diag: np.ndarray):
        self.fp_train = fp_train
        cfg = self.cfg
        use_counts = _uses_counts(cfg.FINGERPRINT_MODE)
        weights = 1.0 / D_diag
        self.y_mean_ = float(np.average(y_train, weights=weights))
        y_c = y_train - self.y_mean_
        K = tanimoto(fp_train, fp_train, use_counts)
        A = K + self.lam * np.diag(D_diag)
        try:
            c, low = cho_factor(A, lower=True)
            self.alpha_ = cho_solve((c, low), y_c)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Cholesky failed; falling back to np.linalg.solve.", UserWarning, stacklevel=2
            )
            self.alpha_ = np.linalg.solve(A, y_c)
        return self

    def predict(self, fp_test: np.ndarray) -> np.ndarray:
        use_counts = _uses_counts(self.cfg.FINGERPRINT_MODE)
        K_star = tanimoto(fp_test, self.fp_train, use_counts)
        return K_star @ self.alpha_ + self.y_mean_

    def predict_clipped(self, fp_test: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
        return np.clip(self.predict(fp_test), y_min, y_max)


# ---------------------------------------------------------------------------
# 6.  Sanity check — weighted local median
# ---------------------------------------------------------------------------


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    idx = np.argsort(values)
    cumw = np.cumsum(weights[idx])
    return float(values[idx[np.searchsorted(cumw, cumw[-1] / 2.0)]])


def local_median_predictions(
    fp_test: np.ndarray,
    fp_train: np.ndarray,
    y_train: np.ndarray,
    D_diag: np.ndarray,
    cfg: Config = None,
    k: int = 15,
    power: float = 2.0,
) -> np.ndarray:
    """Weighted nearest-neighbour median — diagnostic baseline."""
    if cfg is None:
        cfg = Config()
    use_counts = _uses_counts(cfg.FINGERPRINT_MODE)
    K_star = tanimoto(fp_test, fp_train, use_counts)
    w_base = 1.0 / D_diag
    y_med = np.empty(len(fp_test))
    for i in range(len(fp_test)):
        sims = K_star[i]
        top_k = np.argsort(sims)[-k:]
        w = (sims[top_k] ** power) * w_base[top_k]
        y_med[i] = _weighted_median(y_train[top_k], w) if w.sum() > 0 else float(np.mean(y_train))
    return y_med


# ---------------------------------------------------------------------------
# 7.  Analog-neighbourhood cross-validation
# ---------------------------------------------------------------------------


def identify_anchors(df_train: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Positional indices of potent/selective training anchors."""
    mask = (df_train["pEC50"] >= cfg.ANCHOR_MIN_PEC50) & (
        df_train["delta_sel"].fillna(0.0) >= cfg.ANCHOR_MIN_SEL
    )
    return np.where(mask.values)[0]


def analog_neighbourhood_cv(
    fp: np.ndarray, y: np.ndarray, D_diag: np.ndarray, df_train: pd.DataFrame, cfg: Config
) -> tuple:
    """
    Analog-neighbourhood cross-validation.

    Folds: for each potent/selective anchor, hold out nearby training
    compounds (Tanimoto ≥ CV_NEIGHBORHOOD_SIM) and evaluate MAE.
    Sweeps only LAMBDA_GRID — ECFP6/RHO removed (CV showed ρ=1.0 always wins).

    Returns (best_lam, results_dict).
    """
    use_counts = _uses_counts(cfg.FINGERPRINT_MODE)
    lam_grid = cfg.LAMBDA_GRID
    cv_sim = cfg.CV_NEIGHBORHOOD_SIM

    anchor_idx = identify_anchors(df_train, cfg)
    if len(anchor_idx) == 0:
        warnings.warn("No anchor compounds found; using default lambda.", UserWarning, stacklevel=2)
        return lam_grid[len(lam_grid) // 2], {}

    print(
        f"[CV] {len(anchor_idx)} anchors | {len(fp)} training compounds | {len(lam_grid)} λ values"
    )
    print("[CV] Precomputing full kernel matrix...")
    K_full = tanimoto(fp, fp, use_counts)

    # Identify valid folds
    folds = []
    for a_idx in anchor_idx:
        sims = K_full[:, a_idx]
        hood = np.where((sims >= cv_sim) & (np.arange(len(fp)) != a_idx))[0]
        if len(hood) >= 3:
            folds.append((a_idx, hood))

    if not folds:
        warnings.warn("No valid CV folds found.", UserWarning, stacklevel=2)
        return lam_grid[len(lam_grid) // 2], {}

    hood_sizes = [len(h) for _, h in folds]
    total_held = sum(hood_sizes)
    print(
        f"[CV] valid folds: {len(folds)}/{len(anchor_idx)} | "
        f"hood sizes: min={min(hood_sizes)} max={max(hood_sizes)} | "
        f"total held-out: {total_held} (threshold={cv_sim})"
    )

    results: dict = {}
    med_maes: list = []

    for f_idx, (a_idx, hood) in enumerate(folds):  # noqa: B007
        train_mask = np.ones(len(fp), dtype=bool)
        train_mask[hood] = False

        K_tr = K_full[np.ix_(np.where(train_mask)[0], np.where(train_mask)[0])]
        K_val = K_full[np.ix_(hood, np.where(train_mask)[0])]
        y_tr = y[train_mask]
        D_tr = D_diag[train_mask]
        y_val = y[hood]
        w_tr = 1.0 / D_tr
        y_mean = float(np.average(y_tr, weights=w_tr))
        y_c = y_tr - y_mean

        # Local-median baseline for this fold
        w_base = 1.0 / D_tr
        y_med_val = np.empty(len(hood))
        for vi in range(len(hood)):
            sims_vi = K_val[vi]
            top_k = np.argsort(sims_vi)[-15:]
            ww = (sims_vi[top_k] ** 2) * w_base[top_k]
            y_med_val[vi] = (
                _weighted_median(y_tr[top_k], ww) if ww.sum() > 0 else float(np.mean(y_tr))
            )
        med_maes.append(float(np.mean(np.abs(y_med_val - y_val))))

        for lam in lam_grid:
            A = K_tr + lam * np.diag(D_tr)
            try:
                c, low = cho_factor(A, lower=True)
                alpha = cho_solve((c, low), y_c)
            except np.linalg.LinAlgError:
                alpha = np.linalg.solve(A, y_c)
            yhat_val = K_val @ alpha + y_mean
            mae = float(np.mean(np.abs(yhat_val - y_val)))
            results.setdefault(lam, []).append(mae)

        print(f"[CV] fold {f_idx + 1}/{len(folds)} done  (hood size={len(hood)})")

    # Baseline summary
    med_mean = float(np.mean(med_maes))
    med_worst = float(np.max(med_maes))
    print(
        f"[CV] local-median baseline: "
        f"mean MAE={med_mean:.3f}  worst MAE={med_worst:.3f}  "
        f"folds={len(med_maes)}"
    )

    # Aggregate
    agg = {
        lam: {"mean_mae": float(np.mean(v)), "worst_mae": float(np.max(v)), "n_folds": len(v)}
        for lam, v in results.items()
    }
    agg["local_median_baseline"] = {
        "mean_mae": med_mean,
        "worst_mae": med_worst,
        "n_folds": len(med_maes),
    }

    width = len(str(len(lam_grid)))
    for idx, (lam, r) in enumerate(
        sorted((k, v) for k, v in agg.items() if k != "local_median_baseline")
    ):
        print(
            f"[CV {idx + 1:{width}}/{len(lam_grid)}] "
            f"λ={lam:<6}  "
            f"mean MAE={r['mean_mae']:.3f}  worst MAE={r['worst_mae']:.3f}  "
            f"folds={r['n_folds']}"
        )

    best_lam = min(
        (k for k in agg if k != "local_median_baseline"), key=lambda k: agg[k]["mean_mae"]
    )
    r = agg[best_lam]
    delta = r["mean_mae"] - med_mean
    print(
        f"[CV] ✓ best λ={best_lam} | mean MAE={r['mean_mae']:.3f}  worst MAE={r['worst_mae']:.3f}"
    )
    print(
        f"[CV]   vs local-median baseline: Δ={delta:+.3f} "
        f"({'↓ better' if delta < 0 else '↑ worse'} than median)"
    )

    return best_lam, agg


# ---------------------------------------------------------------------------
# 8.  Column names and data loading
# ---------------------------------------------------------------------------

_COL_SMILES = "SMILES"
_COL_NAME = "Molecule Name"
_COL_PEC50 = "pEC50"
_COL_PEC50_SE = "pEC50_std.error (-log10(molarity))"
_COL_EMAX_CTRL = "Emax.vs.pos.ctrl_estimate (dimensionless)"
_COL_TEST_SMILES = "CXSMILES (CDD Compatible)"

_HF_BASE = "hf://datasets/openadmet/pxr-challenge-train-test"
_HF_TRAIN = f"{_HF_BASE}/pxr-challenge_TRAIN.csv"
_HF_TEST = f"{_HF_BASE}/pxr-challenge_TEST_BLINDED.csv"
_HF_COUNTER = f"{_HF_BASE}/pxr-challenge_counter-assay_TRAIN.csv"

_N_TEST_EXPECTED = 513  # Activity Track: challenge requires exactly 513 rows


def load_training_data(
    train_path: str = _HF_TRAIN, counter_path: str = _HF_COUNTER, cfg: Config = None
) -> pd.DataFrame:
    """
    Load primary DRC training data and merge counter-assay selectivity delta.
    Merge key: 'Molecule Name'.
    """
    if cfg is None:
        cfg = Config()
    df = pd.read_csv(train_path)
    df = df.rename(columns={_COL_PEC50_SE: "pEC50_se", _COL_EMAX_CTRL: "emax_vs_ctrl"})
    dc = pd.read_csv(counter_path).rename(columns={_COL_PEC50: "pEC50_counter"})[
        [_COL_NAME, "pEC50_counter"]
    ]
    df = df.merge(dc, on=_COL_NAME, how="left")
    df["has_counter"] = df["pEC50_counter"].notna()
    df["delta_sel"] = (df[_COL_PEC50] - df["pEC50_counter"]).where(df["has_counter"])
    df["SMILES_canon"] = df[_COL_SMILES].apply(canonicalize)
    n_bad = df["SMILES_canon"].isna().sum()
    if n_bad:
        warnings.warn(f"{n_bad} training SMILES dropped (unparsable).", UserWarning, stacklevel=2)
    return df.dropna(subset=["SMILES_canon", _COL_PEC50]).reset_index(drop=True)


def load_test_data(test_path: str = _HF_TEST) -> pd.DataFrame:
    """Load test SMILES. Handles CDD-format SMILES column rename."""
    df = pd.read_csv(test_path)
    if _COL_TEST_SMILES in df.columns and _COL_SMILES not in df.columns:
        df = df.rename(columns={_COL_TEST_SMILES: _COL_SMILES})
    elif _COL_TEST_SMILES in df.columns:
        df = df.drop(columns=[_COL_SMILES]).rename(columns={_COL_TEST_SMILES: _COL_SMILES})
    df["SMILES_canon"] = df[_COL_SMILES].apply(canonicalize)
    n_bad = df["SMILES_canon"].isna().sum()
    if n_bad:
        warnings.warn(f"{n_bad} test SMILES could not be parsed.", UserWarning, stacklevel=2)
    return df.dropna(subset=["SMILES_canon"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 9.  Submission writer
# ---------------------------------------------------------------------------


def write_submission(
    df_test: pd.DataFrame, yhat: np.ndarray, output_path: str, fmt: str = "parquet"
) -> pd.DataFrame:
    """
    Write a challenge-compliant Activity Track submission file.

    Produces a file with exactly the three required columns:
        SMILES | Molecule Name | pEC50

    Parameters
    ----------
    df_test     : test DataFrame (must contain SMILES and Molecule Name)
    yhat        : predicted pEC50 array, same length as df_test
    output_path : destination path (use .parquet or .csv extension)
    fmt         : "parquet" (preferred by challenge) or "csv"

    Returns the submission DataFrame for inspection.

    Raises
    ------
    ValueError if row count != 513, NaN/inf present, or duplicate identifiers.
    """
    df = df_test[[_COL_SMILES, _COL_NAME]].copy().reset_index(drop=True)
    df[_COL_PEC50] = yhat.astype(np.float64)

    # --- Validation -------------------------------------------------------
    errors = []
    if len(df) != _N_TEST_EXPECTED:
        errors.append(f"Row count: got {len(df)}, expected {_N_TEST_EXPECTED}")
    if df[_COL_PEC50].isna().any():
        errors.append(f"NaN in pEC50 ({df[_COL_PEC50].isna().sum()} rows)")
    if np.isinf(df[_COL_PEC50].values).any():
        errors.append(f"Inf in pEC50 ({np.isinf(df[_COL_PEC50].values).sum()} rows)")
    if df[_COL_NAME].nunique() != len(df):
        n_dup = len(df) - df[_COL_NAME].nunique()
        errors.append(f"{n_dup} duplicate Molecule Names")
    if errors:
        raise ValueError("Submission validation failed:\n  " + "\n  ".join(errors))

    # --- Write ------------------------------------------------------------
    if fmt == "parquet":
        df.to_parquet(output_path, index=False)
    elif fmt == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"fmt must be 'parquet' or 'csv', got {fmt!r}")

    # --- Summary ----------------------------------------------------------
    p = df[_COL_PEC50]
    print(f"✓ Submission written: {output_path}")
    print(f"  Format   : {fmt}")
    print(f"  Rows     : {len(df)} of {_N_TEST_EXPECTED} required")
    print(f"  Columns  : {list(df.columns)}")
    print(
        f"  pEC50    : min={p.min():.3f}  p5={p.quantile(0.05):.3f}  "
        f"median={p.median():.3f}  p95={p.quantile(0.95):.3f}  max={p.max():.3f}"
    )
    n_floor = int((p < 4.5).sum())
    n_ceil = int((p > 8.0).sum())
    if n_floor:
        print(f"  ⚠  {n_floor} predictions below 4.5 — inspect before submitting")
    if n_ceil:
        print(f"  ⚠  {n_ceil} predictions above 8.0 — inspect before submitting")
    return df


# ---------------------------------------------------------------------------
# 10.  Phase 2 — cluster-offset correction  (gated by Config flag)
# ---------------------------------------------------------------------------


def compute_cluster_offsets(
    fp_as1: np.ndarray,
    y_as1: np.ndarray,
    yhat_as1_p1: np.ndarray,
    fp_anchors: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    use_counts = _uses_counts(cfg.FINGERPRINT_MODE)
    K = tanimoto(fp_as1, fp_anchors, use_counts)
    n_anchors = fp_anchors.shape[0]
    offsets = np.zeros(n_anchors)
    for a in range(n_anchors):
        cluster = np.where(K[:, a] >= cfg.NEIGHBORHOOD_SIM)[0]
        if len(cluster) < 2:
            continue
        n_s = len(cluster)
        residuals = y_as1[cluster] - yhat_as1_p1[cluster]
        offsets[a] = (n_s / (n_s + cfg.OFFSET_KAPPA)) * float(np.median(residuals))
    return offsets


def apply_cluster_offsets(
    fp_test: np.ndarray,
    yhat_test: np.ndarray,
    fp_anchors: np.ndarray,
    offsets: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    use_counts = _uses_counts(cfg.FINGERPRINT_MODE)
    significant = np.abs(offsets) >= cfg.MIN_OFFSET
    if not significant.any():
        print("[Phase 2] No cluster offsets exceed threshold.")
        return yhat_test.copy()
    K = tanimoto(fp_test, fp_anchors, use_counts)
    w = np.maximum(K, 0.0)
    ws = w.sum(axis=1, keepdims=True)
    pi = np.where(ws > 0, w / ws, 0.0)
    return yhat_test + pi @ (offsets * significant.astype(float))


# ---------------------------------------------------------------------------
# 11.  Phase 1 — main pipeline
# ---------------------------------------------------------------------------


def run_phase1(
    train_path: str = _HF_TRAIN,
    counter_path: str = _HF_COUNTER,
    test_path: str = _HF_TEST,
    output_path: str = "phase1_submission.parquet",
    fmt: str = "parquet",
    cfg: Config = None,
) -> dict:
    """
    Full Phase 1 pipeline. Returns artifacts dict for Phase 2.

    The submission file is written via write_submission() and validated
    before saving (513 rows, no NaN/inf, correct columns).
    """
    if cfg is None:
        cfg = Config()

    print("=" * 60)
    print(f"PXR Challenge — Phase 1  (mode={cfg.FINGERPRINT_MODE!r})")
    print("=" * 60)

    print("\nLoading data...")
    df_train = load_training_data(train_path, counter_path, cfg)
    df_test = load_test_data(test_path)
    print(f"  Training : {len(df_train):,} compounds")
    print(f"  Test     : {len(df_test):,} compounds")

    print(f"\nComputing fingerprints (mode={cfg.FINGERPRINT_MODE!r})...")
    fp_train = build_fingerprints(df_train["SMILES_canon"].tolist(), cfg)
    fp_test = build_fingerprints(df_test["SMILES_canon"].tolist(), cfg)

    print("\nComputing noise diagonal...")
    y_train = df_train[_COL_PEC50].values.astype(np.float64)
    D_diag = compute_noise_diagonal(df_train, cfg)

    # Data-derived clipping bounds (not hard-coded)
    Y_MIN = max(float(np.quantile(y_train, 0.005)), 3.5)
    Y_MAX = min(float(np.quantile(y_train, 0.995)), 9.0)
    print(f"  Clipping bounds (data-derived): [{Y_MIN:.2f}, {Y_MAX:.2f}]")

    print("\nRunning analog-neighbourhood CV...")
    best_lam, cv_results = analog_neighbourhood_cv(fp_train, y_train, D_diag, df_train, cfg)

    print(f"\nFitting final model (λ={best_lam})...")
    model = TanimotoKRR(lam=best_lam, cfg=cfg)
    model.fit(fp_train, y_train, D_diag)

    print("\nPredicting Phase 1 test set...")
    yhat = model.predict_clipped(fp_test, Y_MIN, Y_MAX)

    # Prediction distribution
    print(
        f"  min={yhat.min():.2f}  "
        f"p5={np.percentile(yhat, 5):.2f}  "
        f"median={np.median(yhat):.2f}  "
        f"p95={np.percentile(yhat, 95):.2f}  "
        f"max={yhat.max():.2f}"
    )
    n_floor = int((yhat <= Y_MIN + 0.01).sum())
    if n_floor:
        print(f"  ⚠  {n_floor} predictions at or near floor ({Y_MIN:.2f})")

    if cfg.RUN_LOCAL_MEDIAN_DIAGNOSTIC:
        print("\nSanity check: KRR vs local median...")
        yhat_med = local_median_predictions(fp_test, fp_train, y_train, D_diag, cfg=cfg)
        disc = np.abs(yhat - yhat_med)
        flagged = np.where(disc > cfg.DIAGNOSTIC_DISAGREEMENT)[0]
        if len(flagged):
            print(
                f"  {len(flagged)} predictions differ >{cfg.DIAGNOSTIC_DISAGREEMENT} "
                f"from local median — showing nearest-neighbour context:"
            )
            K_diag = tanimoto(fp_test, fp_train, _uses_counts(cfg.FINGERPRINT_MODE))
            w_base = 1.0 / D_diag
            train_names = df_train[_COL_NAME].values
            train_pec50 = y_train
            delta_arr = df_train["delta_sel"].to_numpy(dtype=np.float64)
            emax_arr = df_train["emax_vs_ctrl"].to_numpy(dtype=np.float64)
            topk = cfg.DIAGNOSTIC_TOP_K
            display = min(len(flagged), cfg.DIAGNOSTIC_MAX_DISPLAY)
            for i in flagged[np.argsort(-disc[flagged])][:display]:
                sims = K_diag[i]
                nbrs = np.argsort(sims)[-topk:][::-1]
                print(
                    f"\n  ── idx={i}  {df_test.at[i, _COL_NAME]}  "
                    f"KRR={yhat[i]:.2f}  median={yhat_med[i]:.2f}  "
                    f"Δ={disc[i]:+.2f}"
                )
                print(
                    f"     {'Neighbour':<20} {'sim':>5} {'pEC50':>6} "
                    f"{'Δsel':>5} {'Emax':>5} {'w_base':>7}"
                )
                for n in nbrs:
                    dsel = delta_arr[n]
                    emax = emax_arr[n]
                    dsel_s = "  n/a" if np.isnan(dsel) else f"{dsel:5.2f}"
                    emax_s = "  n/a" if np.isnan(emax) else f"{emax:5.2f}"
                    print(
                        f"     {str(train_names[n]):<20} {sims[n]:>5.3f} "
                        f"{train_pec50[n]:>6.2f} {dsel_s} {emax_s} "
                        f"{w_base[n]:>7.3f}"
                    )
            if len(flagged) > display:
                print(
                    f"  ... {len(flagged) - display} more "
                    f"(increase DIAGNOSTIC_MAX_DISPLAY to see all)"
                )
        else:
            print("  All predictions consistent with local median.")
    else:
        yhat_med = None

    print("\nWriting submission...")
    df_sub = write_submission(df_test, yhat, output_path, fmt=fmt)

    return dict(
        model=model,
        cfg=cfg,
        df_train=df_train,
        df_test=df_test,
        df_sub=df_sub,
        fp_train=fp_train,
        fp_test=fp_test,
        y_train=y_train,
        D_diag=D_diag,
        yhat_p1=yhat,
        yhat_med=yhat_med,
        best_lam=best_lam,
        Y_MIN=Y_MIN,
        Y_MAX=Y_MAX,
        cv_results=cv_results,
    )


# ---------------------------------------------------------------------------
# 12.  Phase 2 — incorporate Analog Set 1 labels
# ---------------------------------------------------------------------------


def run_phase2(
    artifacts: dict,
    as1_path: str,
    as2_path: str,
    output_path: str = "phase2_submission.parquet",
    fmt: str = "parquet",
) -> pd.DataFrame:
    """
    Phase 2: add revealed AS1 labels, refit, predict AS2.
    Cluster-offset correction only if cfg.USE_PHASE2_CLUSTER_OFFSET = True.
    """
    cfg = artifacts["cfg"]
    fp_train = artifacts["fp_train"]
    y_train = artifacts["y_train"]
    D_diag = artifacts["D_diag"]
    df_train = artifacts["df_train"]
    model_p1 = artifacts["model"]
    Y_MIN = artifacts["Y_MIN"]
    Y_MAX = artifacts["Y_MAX"]

    print("Loading Analog Set 1 labels...")
    df_as1 = pd.read_csv(as1_path)
    df_as1["SMILES_canon"] = df_as1[_COL_SMILES].apply(canonicalize)
    df_as1 = df_as1.dropna(subset=["SMILES_canon", _COL_PEC50]).reset_index(drop=True)
    print(f"  AS1: {len(df_as1):,} compounds")

    fp_as1 = build_fingerprints(df_as1["SMILES_canon"].tolist(), cfg)
    y_as1 = df_as1[_COL_PEC50].values.astype(np.float64)
    D_as1 = np.full(len(df_as1), 2.0 * cfg.SIGMA_FLOOR**2)

    fp_ext = np.vstack([fp_train, fp_as1])
    y_ext = np.concatenate([y_train, y_as1])
    D_ext = np.concatenate([D_diag, D_as1])
    print(f"  Extended training: {len(y_ext):,} compounds")

    print("Refitting model on extended training set...")
    model_p2 = TanimotoKRR(lam=artifacts["best_lam"], cfg=cfg)
    model_p2.fit(fp_ext, y_ext, D_ext)

    print("Loading Analog Set 2...")
    df_as2 = load_test_data(as2_path)
    fp_as2 = build_fingerprints(df_as2["SMILES_canon"].tolist(), cfg)

    yhat_as2 = model_p2.predict_clipped(fp_as2, Y_MIN, Y_MAX)

    if cfg.USE_PHASE2_CLUSTER_OFFSET:
        print("Computing cluster-offset corrections from AS1 residuals...")
        anchor_idx = identify_anchors(df_train, cfg)
        fp_anchors = fp_train[anchor_idx]
        yhat_as1_p1 = model_p1.predict_clipped(fp_as1, Y_MIN, Y_MAX)
        offsets = compute_cluster_offsets(fp_as1, y_as1, yhat_as1_p1, fp_anchors, cfg)
        yhat_as2 = apply_cluster_offsets(fp_as2, yhat_as2, fp_anchors, offsets, cfg)
        yhat_as2 = np.clip(yhat_as2, Y_MIN, Y_MAX)
        n_sig = int((np.abs(offsets) >= cfg.MIN_OFFSET).sum())
        print(f"  {n_sig} cluster offsets applied")

    print("\nWriting Phase 2 submission...")
    df_sub = write_submission(df_as2, yhat_as2, output_path, fmt=fmt)
    return df_sub


# %%
# Phase 1 — run

cfg = Config()

artifacts = run_phase1(cfg=cfg, output_path="phase1_submission.csv", fmt="csv")

# %%
# Phase 2 — uncomment after Analog Set 1 is revealed (May 26)

# cfg.USE_PHASE2_CLUSTER_OFFSET = True   # enable if AS1 residuals show series bias
# run_phase2(
#     artifacts    = artifacts,
#     as1_path     = "analog_set1_revealed.csv",
#     as2_path     = "analog_set2_test.csv",
#     output_path  = "phase2_submission.parquet",
#     fmt          = "parquet",
# )
