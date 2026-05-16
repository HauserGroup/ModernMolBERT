"""
https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/
"""

# %%
import numpy as np
import pandas as pd
from datasets import load_dataset

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR

# ============================================================
# CONSTANTS: edit only these
# ============================================================

RANDOM_STATE = 42
N_FOLDS = 10
FINGERPRINT_KIND = "morgan"  # "morgan", "rdkit", "atom_pair", "torsion"
FP_SIZE = 1024  # Number of bits in the fingerprint. ECFP4 with radius=2 typically uses 1024 bits.

# Only used for Morgan fingerprints.
# radius=2 -> ECFP4, radius=3 -> ECFP6
MORGAN_RADIUS = 2
USE_CHIRALITY = False
USE_HUBER_CALIBRATION = True
BLEND_WEIGHTS = np.linspace(0.75, 0.95, 41)


# %%
# -----------------------------
# Metrics
# -----------------------------


def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / denom


def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


# ============================================================
# Fingerprints
# ============================================================


def make_fingerprint_generator():
    if FINGERPRINT_KIND == "morgan":
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS,
            fpSize=FP_SIZE,
            includeChirality=USE_CHIRALITY,
        )

    if FINGERPRINT_KIND == "rdkit":
        return rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=FP_SIZE,
        )

    if FINGERPRINT_KIND == "atom_pair":
        return rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=FP_SIZE,
            includeChirality=USE_CHIRALITY,
        )

    if FINGERPRINT_KIND == "torsion":
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=FP_SIZE,
            includeChirality=USE_CHIRALITY,
        )

    raise ValueError(f"Unknown FINGERPRINT_KIND: {FINGERPRINT_KIND}")


def fingerprint_from_smiles(smiles):
    generator = make_fingerprint_generator()
    rows = []
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        arr = generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
        rows.append(arr)
        valid[i] = True
    if rows:
        X = np.vstack(rows).astype(np.float32, copy=False)
    else:
        X = np.empty((0, FP_SIZE), dtype=np.float32)

    return X, valid


# -----------------------------
# Tanimoto similarity
# -----------------------------


def tanimoto_similarity(X, Y):
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    intersection = X @ Y.T
    x_sum = X.sum(axis=1)[:, None]
    y_sum = Y.sum(axis=1)[None, :]
    union = x_sum + y_sum - intersection

    return intersection / np.maximum(union, 1e-8)


# -----------------------------
# Tanimoto kNN regressor
# -----------------------------


class TanimotoKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, power=1.0, eps=1e-8):
        self.n_neighbors = n_neighbors
        self.power = power
        self.eps = eps

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.float32)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        sim = tanimoto_similarity(X, self.X_train_)

        k = min(self.n_neighbors, self.X_train_.shape[0])
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

        rows = np.arange(X.shape[0])[:, None]
        sim_k = sim[rows, idx]
        y_k = self.y_train_[idx]

        weights = np.maximum(sim_k, self.eps) ** self.power
        return np.sum(weights * y_k, axis=1) / np.sum(weights, axis=1)


# -----------------------------
# Tanimoto SVR
# -----------------------------


class TanimotoSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=3.0, epsilon=0.1):
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float32)
        K = tanimoto_similarity(self.X_train_, self.X_train_)

        self.model_ = SVR(
            kernel="precomputed",
            C=self.C,
            epsilon=self.epsilon,
        )
        self.model_.fit(K, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        K = tanimoto_similarity(X, self.X_train_)
        return self.model_.predict(K)


# -----------------------------
# OOF helper
# -----------------------------


def oof_predict(model_factory, X, y, cv):
    pred = np.zeros(len(y), dtype=np.float32)

    for train_idx, val_idx in cv.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred[val_idx] = model.predict(X[val_idx])

    return pred


# %%
# -----------------------------
# Calibrated Blended Regressor
# -----------------------------


class CalibratedBlendedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, knn_model, svr_model, blend_weight=0.5, epsilon=1.35):
        self.knn_model = knn_model
        self.svr_model = svr_model
        self.blend_weight = blend_weight
        self.epsilon = epsilon
        self.calibrator_ = None

    def fit(self, X, y):
        # Fit underlying models if not already fitted
        if not hasattr(self.knn_model, "X_train_"):
            self.knn_model.fit(X, y)
        if not hasattr(self.svr_model, "X_train_"):
            self.svr_model.fit(X, y)

        # Get blended predictions
        knn_pred = self.knn_model.predict(X)
        svr_pred = self.svr_model.predict(X)
        blended_pred = self.blend_weight * svr_pred + (1.0 - self.blend_weight) * knn_pred

        # Fit calibration model
        self.calibrator_ = HuberRegressor(epsilon=self.epsilon)
        self.calibrator_.fit(blended_pred.reshape(-1, 1), y)

        return self

    def predict(self, X):
        knn_pred = self.knn_model.predict(X)
        svr_pred = self.svr_model.predict(X)
        blended_pred = self.blend_weight * svr_pred + (1.0 - self.blend_weight) * knn_pred

        if self.calibrator_ is None:
            return blended_pred

        return self.calibrator_.predict(blended_pred.reshape(-1, 1))


# %%
# -----------------------------
# Load data
# -----------------------------

df = load_dataset("openadmet/pxr-challenge-train-test")["train"].to_pandas()

smiles = df["SMILES"].tolist()
pEC50 = df["pEC50"].values.astype(np.float32)

X, valid = fingerprint_from_smiles(smiles)

X = X[valid]
y = pEC50[valid]

print("N valid:", len(y))
print("pEC50 range:", float(np.min(y)), float(np.max(y)))

# %%
# -----------------------------
# Cross-validation
# -----------------------------

cv = KFold(n_splits=10, shuffle=True, random_state=42)


# -----------------------------
# Tune kNN
# -----------------------------

knn_grid_fine = [
    {"n_neighbors": 5, "power": 1.5},
    {"n_neighbors": 7, "power": 1.5},
    {"n_neighbors": 9, "power": 1.5},
    {"n_neighbors": 5, "power": 2.0},
    {"n_neighbors": 7, "power": 2.0},
    {"n_neighbors": 9, "power": 2.0},
    {"n_neighbors": 5, "power": 2.5},
    {"n_neighbors": 7, "power": 2.5},
    {"n_neighbors": 9, "power": 2.5},
]

knn_results = []

for params in knn_grid_fine:
    pred = oof_predict(
        lambda params=params: TanimotoKNNRegressor(**params),
        X,
        y,
        cv,
    )

    score = rae(y, pred)
    knn_results.append((score, params, pred))

    print("kNN", params, "RAE:", round(score, 4), "MAE:", round(mae(y, pred), 4))

best_knn_score, best_knn_params, best_knn_oof = min(knn_results, key=lambda z: z[0])

print("\nBest kNN:")
print(best_knn_params)
print("RAE:", best_knn_score)


# %%
# -----------------------------
# Tune Tanimoto SVR
# -----------------------------

svr_grid_fine = [
    {"C": 0.5, "epsilon": 0.05},
    {"C": 0.7, "epsilon": 0.05},
    {"C": 1.0, "epsilon": 0.05},
    {"C": 1.3, "epsilon": 0.05},
    {"C": 1.6, "epsilon": 0.05},
    {"C": 0.5, "epsilon": 0.075},
    {"C": 0.7, "epsilon": 0.075},
    {"C": 1.0, "epsilon": 0.075},
    {"C": 1.3, "epsilon": 0.075},
    {"C": 1.6, "epsilon": 0.075},
    {"C": 0.5, "epsilon": 0.1},
    {"C": 0.7, "epsilon": 0.1},
    {"C": 1.0, "epsilon": 0.1},
    {"C": 1.3, "epsilon": 0.1},
    {"C": 1.6, "epsilon": 0.1},
    {"C": 0.5, "epsilon": 0.15},
    {"C": 0.7, "epsilon": 0.15},
    {"C": 1.0, "epsilon": 0.15},
    {"C": 1.3, "epsilon": 0.15},
    {"C": 1.6, "epsilon": 0.15},
]

svr_results = []

for params in svr_grid_fine:
    pred = oof_predict(
        lambda params=params: TanimotoSVR(**params),
        X,
        y,
        cv,
    )

    score = rae(y, pred)
    svr_results.append((score, params, pred))

    print("SVR", params, "RAE:", round(score, 4), "MAE:", round(mae(y, pred), 4))

best_svr_score, best_svr_params, best_svr_oof = min(svr_results, key=lambda z: z[0])

print("\nBest SVR:")
print(best_svr_params)
print("RAE:", best_svr_score)


# %%
# -----------------------------
# Tune blend weight
# -----------------------------

blend_results = []


for w_svr in BLEND_WEIGHTS:
    pred = w_svr * best_svr_oof + (1.0 - w_svr) * best_knn_oof
    score = rae(y, pred)
    blend_results.append((score, w_svr, pred))

best_blend_score, best_w_svr, best_blend_oof = min(blend_results, key=lambda z: z[0])

print("\nBest blend:")
print("SVR weight:", best_w_svr)
print("kNN weight:", 1.0 - best_w_svr)
print("RAE:", best_blend_score)
print("MAE:", mae(y, best_blend_oof))


# %%
# -----------------------------
# Final fit on all training data
# -----------------------------

final_knn = TanimotoKNNRegressor(**best_knn_params)
final_svr = TanimotoSVR(**best_svr_params)

final_knn.fit(X, y)
final_svr.fit(X, y)

# Create and fit calibrated blended model
final_model = CalibratedBlendedRegressor(
    knn_model=final_knn,
    svr_model=final_svr,
    blend_weight=best_w_svr,
    epsilon=1.35,
)
final_model.fit(X, y)

# Get final model performance on training data
final_pred = final_model.predict(X)

print(f"Training RAE: {rae(y, final_pred):.4f}")
print(f"Training MAE: {mae(y, final_pred):.4f}")


# -----------------------------
# Final performance Notes
# -----------------------------
# ECFP4
# Training RAE: 0.1923
# Training MAE: 0.1750
# ECFP4 with chirality
# Training RAE: 0.1835
# Training MAE: 0.1670
# ECFP6
# Training RAE: 0.1645
# Training MAE: 0.1497
# ECFP6 with chirality
# Training RAE: 0.1960
# Training MAE: 0.1783
# RDKit topological fingerprint
# Training RAE: 0.3142
# Training MAE: 0.2859
# Atom pair fingerprint
# Training RAE: 0.2619
# Training MAE: 0.2383
# Torsion fingerprint
# Training RAE: 0.2619
# Training MAE: 0.2383
# ECFP8
# Training RAE: 0.1882
# Training MAE: 0.1712


# %%
# -----------------------------
# Plot 1: observed vs predicted
# -----------------------------

plt.figure(figsize=(6, 6))

plt.scatter(y, best_blend_oof, alpha=0.35, s=18)

lo = min(np.min(y), np.min(best_blend_oof))
hi = max(np.max(y), np.max(best_blend_oof))

plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

plt.xlabel("Observed pEC50")
plt.ylabel("OOF predicted pEC50")
plt.title("OOF predictions: blended Tanimoto-SVR + kNN")

plt.text(
    0.05,
    0.95,
    f"RAE = {rae(y, best_blend_oof):.3f}\nMAE = {mae(y, best_blend_oof):.3f}",
    transform=plt.gca().transAxes,
    va="top",
)

plt.tight_layout()
plt.show()

# %%
# -----------------------------
# Plot 2: residuals
# -----------------------------

resid = best_blend_oof - y

plt.figure(figsize=(7, 4))

plt.scatter(y, resid, alpha=0.35, s=18)
plt.axhline(0, linestyle="--", linewidth=2)

plt.xlabel("Observed pEC50")
plt.ylabel("Residual: predicted - observed")
plt.title("Residuals vs observed pEC50")

plt.tight_layout()
plt.show()

# %%
# -----------------------------
# Optional Plot 3: model disagreement
# -----------------------------

disagreement = best_svr_oof - best_knn_oof
abs_error = np.abs(best_blend_oof - y)

plt.figure(figsize=(7, 4))

plt.scatter(disagreement, abs_error, alpha=0.35, s=18)
plt.axvline(0, linestyle="--", linewidth=2)

plt.xlabel("SVR prediction - kNN prediction")
plt.ylabel("Absolute OOF error")
plt.title("Do SVR/kNN disagreements flag hard compounds?")

plt.tight_layout()
plt.show()

# %%

# Get calibrated predictions from the final model
best_blend_oof_cal = final_model.predict(X)

print("Raw blend RAE:", rae(y, best_blend_oof))
print("Calibrated blend RAE:", rae(y, best_blend_oof_cal))
print("Raw blend MAE:", mae(y, best_blend_oof))
print("Calibrated blend MAE:", mae(y, best_blend_oof_cal))

print("Calibration intercept:", final_model.calibrator_.intercept_)
print("Calibration slope:", final_model.calibrator_.coef_[0])
# %%
resid = best_blend_oof - y

plt.figure(figsize=(7, 4))
plt.scatter(best_blend_oof, resid, alpha=0.35, s=18)
plt.axhline(0, linestyle="--", linewidth=2)
plt.xlabel("OOF predicted pEC50")
plt.ylabel("Residual: predicted - observed")
plt.title("Residuals vs predicted pEC50")
plt.tight_layout()
plt.show()

# %%
tmp = pd.DataFrame(
    {
        "y": y,
        "pred": best_blend_oof,
        "pred_cal": best_blend_oof_cal,
    }
)

tmp["bin"] = pd.qcut(tmp["pred"], 10, duplicates="drop")

summary = tmp.groupby("bin").apply(
    lambda d: pd.Series(
        {
            "n": len(d),
            "mean_y": d["y"].mean(),
            "mean_pred": d["pred"].mean(),
            "mean_pred_cal": d["pred_cal"].mean(),
            "mae_raw": np.mean(np.abs(d["y"] - d["pred"])),
            "mae_cal": np.mean(np.abs(d["y"] - d["pred_cal"])),
        }
    )
)

summary
