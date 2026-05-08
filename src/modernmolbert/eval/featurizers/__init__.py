"""Representation featurizers.

Do not import heavy optional featurizers here. Keep this module lightweight.
"""

from modernmolbert.eval.featurizers.base import FeatureBatch, RepresentationFeaturizer
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.featurizers.rdkit_ecfp import ECFP4Featurizer
from modernmolbert.eval.featurizers.hf_smiles import HuggingFaceSmilesFeaturizer

__all__ = [
    "FeatureBatch",
    "RepresentationFeaturizer",
    "DummyFeaturizer",
    "ECFP4Featurizer",
    "HuggingFaceSmilesFeaturizer",
]
