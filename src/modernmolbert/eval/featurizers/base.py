from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureBatch:
    """Output from a representation featurizer.

    Attributes
    ----------
    X:
        Feature matrix with one row per valid molecule.

    valid_mask:
        Boolean mask over the original input sequence. If ``valid_mask[i]`` is
        false, the corresponding input molecule could not be featurized and has
        no row in ``X``.

    metadata:
        Optional diagnostic information. This is intentionally free-form so
        individual featurizers can report tokenization failure rates, device,
        model checkpoint, etc.
    """

    X: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def check(self, n_inputs: int) -> None:
        """Validate shape invariants."""
        if not isinstance(self.X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(self.X)!r}")

        if not isinstance(self.valid_mask, np.ndarray):
            raise TypeError(f"valid_mask must be a numpy array, got {type(self.valid_mask)!r}")

        if self.valid_mask.dtype != bool:
            raise TypeError(f"valid_mask must be boolean, got {self.valid_mask.dtype}")

        if self.valid_mask.shape != (n_inputs,):
            raise ValueError(
                f"valid_mask must have shape ({n_inputs},), got {self.valid_mask.shape}"
            )

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")

        n_valid = int(self.valid_mask.sum())
        if self.X.shape[0] != n_valid:
            raise ValueError(
                "Number of rows in X must equal number of valid inputs: "
                f"{self.X.shape[0]} != {n_valid}"
            )

        if not np.issubdtype(self.X.dtype, np.number):
            raise TypeError(f"X must be numeric, got dtype {self.X.dtype}")

        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values")


class RepresentationFeaturizer(Protocol):
    """Protocol implemented by all frozen molecular representation wrappers."""

    @property
    def name(self) -> str: ...

    def featurize_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> FeatureBatch:
        """Convert SMILES strings into a fixed-size feature matrix."""
        ...
