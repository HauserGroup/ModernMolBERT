from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modernmolbert.eval.featurizers.base import FeatureBatch


@dataclass(frozen=True)
class ECFP4Featurizer:
    """RDKit ECFP4/Morgan fingerprint featurizer.

    ECFP4 corresponds to a Morgan/circular fingerprint with radius 2.
    The default output is a 2048-bit hashed binary fingerprint.
    """

    name: str = "ecfp4"
    n_bits: int = 2048
    radius: int = 2
    include_chirality: bool = False
    use_bond_types: bool = True

    def featurize_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> FeatureBatch:
        """Convert SMILES strings to ECFP4 feature vectors.

        Invalid SMILES are marked False in valid_mask and omitted from X.
        """
        # Lazy import so the core package does not require RDKit unless this
        # featurizer is actually used.
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=self.include_chirality,
            useBondTypes=self.use_bond_types,
        )

        rows: list[np.ndarray] = []
        valid_mask = np.zeros(len(smiles), dtype=bool)

        for i, smi in enumerate(smiles):
            if smi is None:
                continue

            text = str(smi).strip()
            if not text:
                continue

            mol = Chem.MolFromSmiles(text)
            if mol is None:
                continue

            arr = generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)

            if arr.shape != (self.n_bits,):
                raise ValueError(
                    f"Expected fingerprint shape ({self.n_bits},), got {arr.shape}"
                )

            rows.append(arr)
            valid_mask[i] = True

        X = (
            np.vstack(rows).astype(np.float32, copy=False)
            if rows
            else np.empty((0, self.n_bits), dtype=np.float32)
        )

        out = FeatureBatch(
            X=X,
            valid_mask=valid_mask,
            metadata={
                "backend": "rdkit",
                "fingerprint": "morgan",
                "ecfp": "ECFP4",
                "radius": self.radius,
                "n_bits": self.n_bits,
                "include_chirality": self.include_chirality,
                "use_bond_types": self.use_bond_types,
            },
        )
        out.check(len(smiles))
        return out
