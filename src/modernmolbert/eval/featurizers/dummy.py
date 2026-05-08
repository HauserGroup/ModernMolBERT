from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modernmolbert.eval.featurizers.base import FeatureBatch


@dataclass(frozen=True)
class DummyFeaturizer:
    """Tiny deterministic featurizer for tests and CLI plumbing.

    It encodes each SMILES string into simple character-derived features. This
    should never be used for scientific results.
    """

    name: str = "dummy"
    n_features: int = 8

    def featurize_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> FeatureBatch:
        rows: list[np.ndarray] = []
        valid: list[bool] = []

        for value in smiles:
            if value is None:
                valid.append(False)
                continue

            text = str(value).strip()
            if not text:
                valid.append(False)
                continue

            vec = np.zeros(self.n_features, dtype=np.float32)
            for i, ch in enumerate(text):
                vec[i % self.n_features] += (ord(ch) % 31) / 31.0

            vec[0] += len(text) / 100.0
            rows.append(vec)
            valid.append(True)

        X = (
            np.vstack(rows).astype(np.float32)
            if rows
            else np.empty((0, self.n_features), dtype=np.float32)
        )
        out = FeatureBatch(X=X, valid_mask=np.asarray(valid, dtype=bool))
        out.check(len(smiles))
        return out
