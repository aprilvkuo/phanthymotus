"""Package public API.

Importing `from model import predict, ObstacleDistancePredictor, ModelConfig`
gives the judgeflow / plugin a single, stable entry surface.

The model uses an open-source pretrained monocular depth backbone (no training
needed). See model.py for details.
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


def predict(image_rgb, cfg: ModelConfig | None = None) -> float:
    """Convenience one-shot predictor (rebuilds model each call — use the class for batches)."""
    return ObstacleDistancePredictor(cfg=cfg).predict(image_rgb)
