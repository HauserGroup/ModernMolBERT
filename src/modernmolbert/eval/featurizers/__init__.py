"""Representation featurizers.

Do not import heavy optional featurizers here. Keep this module lightweight.
"""

from modernmolbert.eval.featurizers.base import FeatureBatch, RepresentationFeaturizer
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer

__all__ = [
    "FeatureBatch",
    "RepresentationFeaturizer",
    "DummyFeaturizer",
]
