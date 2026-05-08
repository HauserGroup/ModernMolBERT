"""Evaluation utilities for ModernMolBERT.

The eval package provides infrastructure for frozen-representation

benchmarks:

    molecules -> representation featurizer -> shared downstream learner

Heavy optional model dependencies should not be imported here.

"""

from modernmolbert.eval.datasets import EvalDataset

# from modernmolbert.eval.runner import FrozenBenchmarkRunner

__all__ = [
    "EvalDataset",
    # "FrozenBenchmarkRunner",
]
