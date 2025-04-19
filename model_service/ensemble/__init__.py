"""
Initialize ensemble module
"""
from .base_ensemble import BaseEnsemble
from .weighted_average import WeightedAverageEnsemble
from .stacking import StackingEnsemble

__all__ = [
    'BaseEnsemble',
    'WeightedAverageEnsemble',
    'StackingEnsemble'
]
