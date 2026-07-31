"""Backwards-compatible re-export of the public model API.

Kept so older imports (`from model.infer import ObstacleDistancePredictor`)
keep working. The implementation now lives in `model.model`.
"""
from .model import (
    ModelConfig,
    MonocularDepthBackbone,
    ObstacleDistancePredictor,
    load_predictor,
)

__all__ = [
    "ModelConfig",
    "MonocularDepthBackbone",
    "ObstacleDistancePredictor",
    "load_predictor",
]
