import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


RF_CLF = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["entropy"],
}

RF_REG = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["squared_error"],
}

RIDGE__MULTIOUTPUT_CLF = {
    "clf__estimator__C": 1 / np.logspace(-2, 3, 10),
    "clf__estimator__solver": ["lbfgs"],
    "clf__estimator__max_iter": [5000],
}

RIDGE_CLF = {
    "clf__C": 1 / np.logspace(-2, 3, 10),
    "clf__solver": ["lbfgs"],
    "clf__max_iter": [5000],
}


RIDGE_REG = {
    "clf__alpha": np.logspace(-2, 3, 10),
    "clf__max_iter": [5000],
    "clf__solver": ["lbfgs"],
}

KNN_CLF = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}

KNN_REG = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}


AVAILABLE_HEADS = ["rf", "ridge", "knn"]


class ConstantProbabilityClassifier(BaseEstimator, ClassifierMixin):
    """Classifier used when one endpoint has zero or one observed class."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p1 = np.full(X.shape[0], self.p, dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FiniteLabelMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """Fit one binary classifier per endpoint using only finite labels."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("FiniteLabelMultiOutputClassifier expects a 2D target array.")

        self.estimators_ = []
        self.n_outputs_ = y.shape[1]

        for col in range(self.n_outputs_):
            mask = np.isfinite(y[:, col])
            y_col = y[mask, col]

            if y_col.size == 0:
                estimator = ConstantProbabilityClassifier(p=0.5)
            elif np.unique(y_col).size < 2:
                estimator = ConstantProbabilityClassifier(p=float(y_col[0]))
            else:
                estimator = clone(self.base_estimator)
                estimator.fit(X[mask], y_col.astype(int))

            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Estimator has not been fitted.")

        columns = []
        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            if isinstance(proba, list):
                proba = proba[0]
            proba = np.asarray(proba)

            if proba.ndim == 2 and proba.shape[1] >= 2:
                columns.append(proba[:, 1])
            elif proba.ndim == 2 and proba.shape[1] == 1:
                columns.append(proba[:, 0])
            else:
                columns.append(proba.ravel())

        return np.column_stack(columns)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def get_knn_distance(embeddings_dtype):
    if np.issubdtype(embeddings_dtype, np.integer):
        return tanimoto_count_distance
    elif np.issubdtype(embeddings_dtype, np.floating):
        return "cosine"
    else:
        raise ValueError(
            f"Unsupported embeddings dtype: {embeddings_dtype}. Expected integer or floating point type."
        )


def tanimoto_count_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    denominator = np.maximum(x, y).sum()
    if denominator == 0:
        return 0.0
    return float(1.0 - np.minimum(x, y).sum() / denominator)


def get_clf_models(no_output: int, embeddings_dtype, n_jobs: int = -1):
    if no_output == 1:
        lr_clf = LogisticRegression()
        lr_params = RIDGE_CLF
    else:
        lr_clf = MultiOutputClassifier(LogisticRegression())
        lr_params = RIDGE__MULTIOUTPUT_CLF

    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestClassifier(n_jobs=n_jobs))]),
            "params": RF_CLF.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", lr_clf),
                ]
            ),
            "params": lr_params.copy(),
        },
        "knn": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_jobs=n_jobs, metric=get_knn_distance(embeddings_dtype)
                        ),
                    ),
                ]
            ),
            "params": KNN_CLF.copy(),
        },
    }


def get_reg_models(embeddings_dtype, n_jobs: int = -1):
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestRegressor(n_jobs=n_jobs))]),
            "params": RF_REG.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", Ridge()),
                ]
            ),
            "params": RIDGE_REG.copy(),
        },
        "knn": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsRegressor(
                            n_jobs=n_jobs, metric=get_knn_distance(embeddings_dtype)
                        ),
                    ),
                ]
            ),
            "params": KNN_REG.copy(),
        },
    }
