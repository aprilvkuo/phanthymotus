"""Package public API.

Importing `from model import predict, ObstacleDistancePredictor` gives the
judgeflow / plugin a single, stable entry surface.
"""
from .config import ModelConfig, RuntimeConfig
from .infer import ObstacleDistancePredictor
from .model import build_model

__all__ = ["ModelConfig", "RuntimeConfig", "ObstacleDistancePredictor", "build_model"]


def predict(image_rgb, cfg: ModelConfig | None = None, rt: RuntimeConfig | None = None) -> float:
    """Convenience one-shot predictor (rebuilds model each call — use the class for batches)."""
    return ObstacleDistancePredictor(cfg=cfg, rt=rt).predict(image_rgb)
