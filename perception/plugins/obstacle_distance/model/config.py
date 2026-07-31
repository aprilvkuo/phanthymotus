"""Model + inference configuration for the Nearest Obstacle Distance perception model.

All tunables live here so the leaderboard submission has a single source of truth.
Resource budget (hard constraint from the requirement):
  - model params < 30M
  - GPU usage < 10% on Jetson Orin 16G
  - real-time inference
"""
from dataclasses import dataclass, field
import os


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _default_model_dir() -> str:
    import os
    # Prefer the phanthymotus mount; fall back to a local weights/ dir.
    for cand in ("/models/obstacle_distance", os.path.join(os.path.dirname(__file__), "..", "weights")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))


@dataclass
class ModelConfig:
    # Input resolution (H, W). Kept small for real-time Jetson inference.
    img_size: tuple = (256, 320)

    # Depth range in meters. Nearest Obstacle Distance is reported in [min_depth, max_depth].
    min_depth: float = 0.1
    max_depth: float = 10.0

    backbone: str = "mobilenet_v3_small"  # ~2.5M params, Jetson-friendly

    # Light decoder channel schedule (stride 16 -> full res via x4 bilinear upsample).
    decoder_channels: tuple = (128, 64, 32, 16)

    # Frontal ROI as fractions of (H, W). We only look straight ahead:
    #   exclude top (ceiling/sky) and very bottom (robot body / immediate floor),
    #   and restrict to a central horizontal band (the path ahead).
    roi_top: float = 0.35
    roi_bottom: float = 0.95
    roi_left: float = 0.30
    roi_right: float = 0.70

    # Robust nearest estimate: take this percentile of valid depths in the ROI
    # (5th percentile ignores a few outlier spikes while staying close to the true minimum).
    nearest_percentile: float = 5.0

    # ImageNet stats for the backbone preprocessor.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class RuntimeConfig:
    # Where weights live. The judgeflow image downloads from juicefs at runtime.
    # Per phanthymotus convention, production mounts models under /models.
    juicefs_base: str = "http://172.28.4.81:34567"
    weights_filename: str = "obstacle_distance.pt"
    model_dir: str = field(default_factory=lambda: _default_model_dir())
    device: str = "cuda" if _cuda_available() else "cpu"
