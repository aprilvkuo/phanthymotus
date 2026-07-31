"""PhanthyMotus NOD benchmark solution package.

Nearest Obstacle Distance (正前方最近可影响通行的障碍物距离) predictor for the
embodied-intelligence spatial-perception leaderboard.

Public API:
    from nod.infer import NODInference
    from solution import NODSolution, predict
"""

from .config import NODConfig
from .interface import BaseNODPredictor, NODResult
from .geometric import GeometricNODPredictor

__all__ = [
    "NODConfig",
    "BaseNODPredictor",
    "NODResult",
    "GeometricNODPredictor",
]
__version__ = "1.0.0"
